//! Self-service roles: a member picks roles for **themselves** and nothing
//! else.
//!
//! The rule this module encodes (operator, 2026-10-31):
//!
//! - *"the bot should be able to change roles for people though only for
//!   themselves — they can't change anyone else's color nor role. Only your
//!   own color. And only you can select your own roles in general as well."*
//! - *"If the role doesn't exist, tag the owner to ask for approval to
//!   creation of new roles other than color roles. Color roles require no
//!   approval."*
//!
//! So there is no target parameter anywhere in this file: the actor is the
//! target, always. A request for a role that does not exist is not created
//! silently — it is posted as an owner ping, unless it is a colour role,
//! which namecolor.rs creates on demand without approval.
//!
//! Safety defaults (pure, unit-tested below):
//!
//! - roles carrying moderation/admin permissions are never handed out from
//!   chat, even when someone asks nicely (trust tiers are not self-granted);
//! - colour roles are refused here and redirected to the colour lane;
//! - `[roles] allow` can narrow the shelf to an explicit list; `deny` always
//!   wins;
//! - a role above the bot's own top role is refused honestly (Discord's
//!   hierarchy, not a judgement call).

use std::collections::HashMap;

use poise::serenity_prelude as serenity;
use serde::Deserialize;

use crate::commands::Data;

type Error = anyhow::Error;
type Ctx<'a> = poise::Context<'a, Data, Error>;

/// `[roles]` in bot.toml. On by default: it only touches the caller's own
/// cosmetic/opt-in roles.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RolesConfig {
    /// Run the lane at all.
    pub enabled: bool,
    /// Scope to one guild (None = every guild the bot is in).
    pub guild_id: Option<u64>,
    /// Explicit self-assignable names. Empty = every safe role.
    pub allow: Vec<String>,
    /// Never self-assignable, matched case-insensitively.
    pub deny: Vec<String>,
}

impl Default for RolesConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            guild_id: None,
            allow: Vec::new(),
            deny: Vec::new(),
        }
    }
}

impl RolesConfig {
    pub fn allows_guild(&self, guild_id: u64) -> bool {
        self.enabled && self.guild_id.map(|g| g == guild_id).unwrap_or(true)
    }
}

/// Permission bits that make a role non-self-assignable. A role carrying any
/// of these is staff infrastructure, not a preference.
pub fn blocked_permissions() -> serenity::Permissions {
    serenity::Permissions::ADMINISTRATOR
        | serenity::Permissions::MANAGE_ROLES
        | serenity::Permissions::MANAGE_GUILD
        | serenity::Permissions::KICK_MEMBERS
        | serenity::Permissions::BAN_MEMBERS
        | serenity::Permissions::MODERATE_MEMBERS
        | serenity::Permissions::MANAGE_CHANNELS
        | serenity::Permissions::MANAGE_WEBHOOKS
        | serenity::Permissions::MANAGE_MESSAGES
        | serenity::Permissions::MANAGE_THREADS
        | serenity::Permissions::MANAGE_EVENTS
        | serenity::Permissions::MENTION_EVERYONE
}

/// One role row, boiled down so the planning logic is testable without a
/// Discord connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfRole {
    pub id: u64,
    pub name: String,
    pub position: u16,
    pub permissions: u64,
    pub managed: bool,
}

impl SelfRole {
    pub fn from_role(role: &serenity::Role) -> Self {
        Self {
            id: role.id.get(),
            name: role.name.clone(),
            position: role.position,
            permissions: role.permissions.bits(),
            managed: role.managed,
        }
    }

    /// Colour roles belong to the name-colour lane, not this one.
    pub fn is_colour(&self) -> bool {
        crate::namecolor::is_color_role_name(&self.name)
    }

    /// Staff/admin permissions → never self-assignable.
    pub fn carries_blocked_permissions(&self) -> bool {
        self.permissions & blocked_permissions().bits() != 0
    }

    fn named(&self, wanted: &str) -> bool {
        self.name.trim().eq_ignore_ascii_case(wanted)
    }
}

/// Is this role something a member may give themselves?
pub fn is_self_assignable(row: &SelfRole, everyone_id: u64, cfg: &RolesConfig) -> bool {
    if row.id == everyone_id || row.managed || row.is_colour() {
        return false;
    }
    if cfg.deny.iter().any(|d| row.named(d)) {
        return false;
    }
    if !cfg.allow.is_empty() && !cfg.allow.iter().any(|a| row.named(a)) {
        return false;
    }
    !row.carries_blocked_permissions()
}

/// The shelf for this guild, highest role first (Discord's own ordering).
pub fn assignable(rows: &[SelfRole], everyone_id: u64, cfg: &RolesConfig) -> Vec<SelfRole> {
    let mut out: Vec<SelfRole> = rows
        .iter()
        .filter(|r| is_self_assignable(r, everyone_id, cfg))
        .cloned()
        .collect();
    out.sort_by(|a, b| {
        b.position
            .cmp(&a.position)
            .then_with(|| a.name.cmp(&b.name))
    });
    out
}

/// Resolve a typed name: exact (case-insensitive) first, then the shortest
/// name containing it — "purple" finds "Royal Purple", but an exact
/// "Royal Purple" always wins.
pub fn resolve<'a>(rows: &'a [SelfRole], query: &str) -> Option<&'a SelfRole> {
    let q = query.trim();
    if q.is_empty() {
        return None;
    }
    if let Some(row) = rows.iter().find(|r| r.named(q)) {
        return Some(row);
    }
    let lower = q.to_lowercase();
    rows.iter()
        .filter(|r| r.name.to_lowercase().contains(&lower))
        .min_by_key(|r| r.name.len())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ask {
    /// Show what the caller may pick.
    List,
    /// Give the caller this role.
    Add(String),
    /// Take this role off the caller.
    Remove(String),
}

fn alnum(word: &str) -> String {
    word.trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
}

/// Drop the conversational scaffolding in front of a role name:
/// "give me the **blue** role" → "blue".
fn strip_leading_filler(text: &str) -> String {
    let mut words: Vec<&str> = text.split_whitespace().collect();
    while let Some(first) = words.first() {
        match alnum(first).as_str() {
            "me" | "the" | "a" | "an" | "my" | "to" | "myself" | "for" | "i" | "it" | "that"
            | "this" | "please" => words.remove(0),
            _ => break,
        };
    }
    words.join(" ")
}

/// Clean a bare role argument ("`pink`", "\"Blue Role\"") into a query,
/// dropping the scaffolding words wherever they landed: "add me to the
/// purple role please" → "purple".
pub fn clean_name(raw: &str) -> String {
    let trimmed = raw.trim().trim_matches(['"', '\'', '`']);
    let words: Vec<&str> = trimmed
        .split_whitespace()
        .filter(|w| {
            !matches!(
                alnum(w).as_str(),
                "role" | "roles" | "please" | "thanks" | "pls"
            )
        })
        .collect();
    words.join(" ")
}

/// Everything after the first matched verb phrase, cleaned. Longest phrases
/// are tried first so "take off" wins over a bare "take".
fn after_verb(text: &str, verbs: &[&str]) -> Option<String> {
    let mut ordered: Vec<&str> = verbs.to_vec();
    ordered.sort_by_key(|v| std::cmp::Reverse(v.len()));
    for verb in ordered {
        if let Some(pos) = text.find(verb) {
            let rest = &text[pos + verb.len()..];
            let cleaned = clean_name(&strip_leading_filler(rest.trim()));
            if !cleaned.is_empty() && cleaned.chars().count() <= 60 {
                return Some(cleaned);
            }
        }
    }
    None
}

const ADD_VERBS: &[&str] = &[
    "give me",
    "add me",
    "gimme",
    "let me have",
    "i want",
    "i'd like",
    "i would like",
    "can i get",
    "can i have",
    "may i have",
    "make me",
    "set my role to",
    "get me",
    "assign me",
    "grant me",
    "pick",
    "choose",
    "select",
    "grab",
    "add",
];

const REMOVE_VERBS: &[&str] = &[
    "take off",
    "take away",
    "get rid of",
    "remove",
    "drop",
    "strip",
    "lose",
];

/// Recognise a self-service role request in chat. Conservative: it must
/// mention "role"/"roles" (or the legacy `!role` command) AND carry an
/// intent verb, so ordinary conversation about someone's role is untouched.
pub fn parse_chat_request(text: &str) -> Option<Ask> {
    let trimmed = text.trim();
    let lower = trimmed.to_lowercase();

    for prefix in ["!selfrole", "!roles", "!role"] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let rest = rest.trim().trim_start_matches(':').trim();
            if rest.is_empty() || matches!(rest, "list" | "help" | "?") {
                return Some(Ask::List);
            }
            if let Some(name) = after_verb(rest, REMOVE_VERBS) {
                return Some(Ask::Remove(name));
            }
            let name = clean_name(rest);
            if name.is_empty() {
                return Some(Ask::List);
            }
            return Some(Ask::Add(name));
        }
    }

    if !lower
        .split(|c: char| !c.is_alphanumeric())
        .any(|w| w == "role" || w == "roles")
    {
        return None;
    }
    // "what roles can I pick?" / "which roles are there?" → the shelf.
    // The plural is required: "what's your role here?" is a real question
    // about the bot, not a role request.
    if lower.contains("what roles")
        || lower.contains("which roles")
        || lower.contains("what role can")
        || lower.contains("which role can")
        || lower.contains("roles can i")
        || lower.contains("role options")
    {
        return Some(Ask::List);
    }
    if let Some(name) = after_verb(&lower, REMOVE_VERBS) {
        return Some(Ask::Remove(name));
    }
    if let Some(name) = after_verb(&lower, ADD_VERBS) {
        return Some(Ask::Add(name));
    }
    None
}

/// The reply when a requested non-colour role does not exist: an explicit
/// owner ping, never a silent creation.
pub fn owner_request_text(requester: &str, role: &str, owner: Option<u64>) -> String {
    let creation = match owner {
        Some(id) => format!("<@{id}>"),
        None => "(no owner configured)".to_string(),
    };
    format!(
        "🪪 There's no `{role}` role in this server yet, and new non-colour roles \
         need the operator's ok — I won't create one on my own.\n\
         {creation} — {requester} is asking for a `{role}` role. \
         Colour roles are the exception: I can make those on request with \
         `/namecolor <colour>`."
    )
}

/// The shelf text (pure so the command and the chat lane share it).
pub fn shelf_text(available: &[SelfRole], held: &[String], colour_help: bool) -> String {
    if available.is_empty() {
        return "🪪 No self-assignable roles in this server yet — an operator can \
                open some up with the `[roles]` table, or we can just talk about it."
            .to_string();
    }
    let names: Vec<String> = available
        .iter()
        .take(30)
        .map(|r| format!("`{}`", r.name))
        .collect();
    let extra = if available.len() > 30 {
        format!(" …and {} more", available.len() - 30)
    } else {
        String::new()
    };
    let held_line = if held.is_empty() {
        "You're not holding any of them right now.".to_string()
    } else {
        format!(
            "You're holding: {}",
            held.iter()
                .map(|h| format!("`{h}`"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    format!(
        "🪪 **Roles you can give yourself** ({}){}\n{}\n{}\n\
         Use `/role <name>` — only **your own** roles, always. \
         `/role remove <name>` takes one back off.{}",
        available.len(),
        extra,
        names.join(" · "),
        held_line,
        if colour_help {
            "\nName colour lives at `/namecolor <colour>` (colour roles need no approval)."
        } else {
            ""
        }
    )
}

fn held_names(member: &serenity::Member, rows: &[SelfRole]) -> Vec<String> {
    let ids: Vec<u64> = member.roles.iter().map(|r| r.get()).collect();
    let mut names: Vec<String> = rows
        .iter()
        .filter(|r| ids.contains(&r.id) && !r.is_colour() && !r.managed)
        .map(|r| r.name.clone())
        .collect();
    names.sort();
    names.dedup();
    names
}

fn display_name(member: &serenity::Member) -> String {
    member
        .nick
        .clone()
        .filter(|n| !n.trim().is_empty())
        .unwrap_or_else(|| member.user.name.clone())
}

/// Apply one self-service role action for `member` — the caller is always the
/// target, which is why there is no target argument.
pub async fn run(
    http: &serenity::Http,
    guild_id: serenity::GuildId,
    bot_id: serenity::UserId,
    member: &serenity::Member,
    ask: Ask,
    cfg: &crate::config::BotConfig,
) -> anyhow::Result<String> {
    anyhow::ensure!(
        cfg.roles.enabled,
        "self-service roles are switched off (`[roles] enabled = false`)"
    );
    anyhow::ensure!(
        cfg.roles.allows_guild(guild_id.get()),
        "self-service roles are scoped to a different server (`[roles] guild_id`)"
    );

    let roles: HashMap<serenity::RoleId, serenity::Role> = guild_id.roles(http).await?;
    let rows: Vec<SelfRole> = roles.values().map(SelfRole::from_role).collect();
    let available = assignable(&rows, guild_id.get(), &cfg.roles);
    let held = held_names(member, &rows);

    let (query, removing) = match &ask {
        Ask::List => return Ok(shelf_text(&available, &held, true)),
        Ask::Add(q) => (q.clone(), false),
        Ask::Remove(q) => (q.clone(), true),
    };

    let Some(target) = resolve(&rows, &query) else {
        return Ok(owner_request_text(
            &display_name(member),
            &query,
            cfg.owner_discord_id,
        ));
    };

    // Colour roles: never created here, never assigned here — the colour
    // lane owns them (and needs no approval to create).
    if target.is_colour() {
        return Ok(format!(
            "🎨 `{}` is a name-colour role — that lane is `/namecolor <colour>` \
             (or just ask me to change my name colour). Colour roles need no approval.",
            target.name
        ));
    }
    let assignable_here = is_self_assignable(target, guild_id.get(), &cfg.roles);
    if !assignable_here {
        let reason = if target.carries_blocked_permissions() {
            "it carries moderation/admin permissions — trust tiers aren't self-granted"
        } else if cfg.roles.deny.iter().any(|d| target.named(d)) {
            "it's on the operator's deny list"
        } else if target.managed {
            "it's managed by an integration, not by me"
        } else {
            "the operator has limited the self-service shelf with `[roles] allow`"
        };
        return Ok(format!(
            "🪪 I won't hand out `{}` from chat: {reason}. An operator can assign it.",
            target.name
        ));
    }

    if removing {
        if !held.iter().any(|h| h.eq_ignore_ascii_case(&target.name)) {
            return Ok(format!("🪪 You don't have `{}` right now.", target.name));
        }
        member
            .remove_role(http, serenity::RoleId::new(target.id))
            .await?;
        let mut after = held.clone();
        after.retain(|h| !h.eq_ignore_ascii_case(&target.name));
        persist(cfg, member, &after).await;
        tracing::info!(
            target: "bot",
            member = member.user.id.get(),
            role = %target.name,
            "self-service role removed"
        );
        return Ok(format!(
            "🪪 Took `{}` off you. Only your own roles, as always.",
            target.name
        ));
    }

    // The bot's authority: Manage Roles + the role must sit below its top.
    let bot_member = guild_id.member(http, bot_id).await?;
    let (bot_perms, bot_top) = crate::namecolor::member_permissions(&bot_member, &roles, guild_id);
    anyhow::ensure!(
        bot_perms.contains(serenity::Permissions::ADMINISTRATOR)
            || bot_perms.contains(serenity::Permissions::MANAGE_ROLES),
        "I don't have **Manage Roles** in this server, so I can't assign roles — an admin can grant it"
    );
    anyhow::ensure!(
        target.position < bot_top || bot_top == 0,
        "`{}` sits above my highest role, so Discord won't let me assign it — an admin needs to move it below mine",
        target.name
    );

    if held.iter().any(|h| h.eq_ignore_ascii_case(&target.name)) {
        return Ok(format!("🪪 You already have `{}`.", target.name));
    }
    member
        .add_role(http, serenity::RoleId::new(target.id))
        .await?;
    let mut after = held.clone();
    after.push(target.name.clone());
    after.sort();
    persist(cfg, member, &after).await;
    tracing::info!(
        target: "bot",
        member = member.user.id.get(),
        role = %target.name,
        "self-service role assigned"
    );
    Ok(format!(
        "🪪 Added `{}` to you. Your own roles only — nobody can pick for you, \
         and I can't pick for anyone else.",
        target.name
    ))
}

/// Best-effort registry write: the member's self-selected roles, so a later
/// `people.lookup` answers from the database instead of guessing.
async fn persist(cfg: &crate::config::BotConfig, member: &serenity::Member, names: &[String]) {
    if let Some(pg) = cfg.postgres.as_ref() {
        let value = names.join(", ");
        if let Err(e) =
            crate::people::remember_roles(pg, &member.user.id.get().to_string(), &value).await
        {
            tracing::debug!(error = %e, "self-service roles not persisted to jerith_people");
        }
    }
}

/// Chat entry point. Never returns an error: the reply text carries the
/// honest refusal (permission, hierarchy, unknown role).
pub async fn handle_chat_request(
    http: &serenity::Http,
    guild_id: serenity::GuildId,
    author_id: serenity::UserId,
    ask: Ask,
    cfg: &crate::config::BotConfig,
) -> String {
    if !cfg.roles.allows_guild(guild_id.get()) {
        return "🪪 self-service roles aren't enabled in this server.".to_string();
    }
    let member = match guild_id.member(http, author_id).await {
        Ok(m) => m,
        Err(e) => return format!("🪪 I couldn't read your member record here ({e})."),
    };
    let bot_id = match http.get_current_user().await {
        Ok(u) => u.id,
        Err(e) => return format!("🪪 I couldn't confirm my own identity ({e})."),
    };
    match run(http, guild_id, bot_id, &member, ask, cfg).await {
        Ok(text) => text,
        Err(e) => format!("🪪 {e}"),
    }
}

/// `/role [name] [remove]` (also `!role`, `!roles`, `!selfrole`).
#[poise::command(
    slash_command,
    prefix_command,
    guild_only,
    aliases("roles", "selfrole")
)]
pub async fn role(
    ctx: Ctx<'_>,
    #[description = "Role name (blank lists what you can pick for yourself)"] name: Option<String>,
    #[description = "Remove this role from yourself instead of adding it"] remove: Option<bool>,
) -> Result<(), Error> {
    let Some(guild_id) = ctx.guild_id() else {
        ctx.say("🪪 roles are per-server — run this inside the server.")
            .await?;
        return Ok(());
    };
    let http = ctx.serenity_context().http.clone();
    let author = guild_id.member(&http, ctx.author().id).await?;
    let cfg = ctx.data().cfg.clone();
    let ask = match name {
        None => Ask::List,
        Some(raw) => {
            let lower = raw.trim().to_lowercase();
            if remove.unwrap_or(false) {
                Ask::Remove(clean_name(&raw))
            } else if let Some(rest) = lower.strip_prefix("remove ") {
                Ask::Remove(clean_name(rest))
            } else if let Some(rest) = lower.strip_prefix("drop ") {
                Ask::Remove(clean_name(rest))
            } else {
                Ask::Add(clean_name(&raw))
            }
        }
    };
    match run(
        &http,
        guild_id,
        ctx.framework().bot_id(),
        &author,
        ask,
        &cfg,
    )
    .await
    {
        Ok(text) => ctx.say(text).await?,
        Err(e) => ctx.say(format!("🪪 {e}")).await?,
    };
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: u64, name: &str, position: u16, perms: u64) -> SelfRole {
        SelfRole {
            id,
            name: name.to_string(),
            position,
            permissions: perms,
            managed: false,
        }
    }

    fn shelf() -> Vec<SelfRole> {
        vec![
            row(1, "No Role Selected", 10, 0),
            row(2, "Notifier", 20, 0),
            row(3, "Announcements", 30, 0),
            row(4, "Purple Color", 40, 0),
            row(
                5,
                "Moderator",
                50,
                serenity::Permissions::KICK_MEMBERS.bits(),
            ),
            row(
                6,
                "Head Hanchos",
                60,
                serenity::Permissions::ADMINISTRATOR.bits(),
            ),
            // Bare colour words belong to the colour lane, never this shelf.
            row(7, "Blue", 70, 0),
        ]
    }

    #[test]
    fn only_safe_non_colour_roles_are_self_assignable() {
        let cfg = RolesConfig::default();
        let rows = shelf();
        let names: Vec<String> = assignable(&rows, 999, &cfg)
            .into_iter()
            .map(|r| r.name)
            .collect();
        assert_eq!(names, vec!["Announcements", "Notifier", "No Role Selected"]);
        // Colour roles never land on this shelf (with or without the
        // "… Color" suffix — bare palette words are picker-owned too).
        assert!(!names.contains(&"Purple Color".to_string()));
        assert!(!names.contains(&"Blue".to_string()));
        // A role with kick/admin permissions is staff-only.
        assert!(!names.contains(&"Moderator".to_string()));
        assert!(!names.contains(&"Head Hanchos".to_string()));
    }

    #[test]
    fn everyone_and_managed_roles_are_never_self_assignable() {
        let cfg = RolesConfig::default();
        let rows = vec![row(999, "@everyone", 0, 0)];
        assert!(assignable(&rows, 999, &cfg).is_empty());
        let mut managed = row(1, "Integration", 5, 0);
        managed.managed = true;
        assert!(assignable(&[managed], 999, &cfg).is_empty());
    }

    #[test]
    fn allow_and_deny_lists_narrow_the_shelf() {
        let mut cfg = RolesConfig {
            allow: vec!["notifier".into()],
            ..Default::default()
        };
        let rows = shelf();
        let names: Vec<String> = assignable(&rows, 999, &cfg)
            .into_iter()
            .map(|r| r.name)
            .collect();
        assert_eq!(names, vec!["Notifier"]);
        cfg.allow.clear();
        cfg.deny = vec!["Notifier".into()];
        let names: Vec<String> = assignable(&rows, 999, &cfg)
            .into_iter()
            .map(|r| r.name)
            .collect();
        assert!(!names.contains(&"Notifier".to_string()));
        assert!(names.contains(&"Announcements".to_string()));
    }

    #[test]
    fn name_resolution_prefers_exact_then_shortest() {
        let rows = vec![
            row(1, "Royal Purple", 10, 0),
            row(2, "Purple", 20, 0),
            row(3, "Purpleish", 30, 0),
        ];
        assert_eq!(resolve(&rows, "purple").map(|r| r.id), Some(2));
        assert_eq!(resolve(&rows, "PURPLE").map(|r| r.id), Some(2));
        assert_eq!(resolve(&rows, "royal purple").map(|r| r.id), Some(1));
        assert_eq!(resolve(&rows, "purpl").map(|r| r.id), Some(2));
        assert!(resolve(&rows, "green").is_none());
    }

    #[test]
    fn chat_requests_are_recognised_without_hijacking_conversation() {
        assert_eq!(
            parse_chat_request("give me the Notifier role"),
            Some(Ask::Add("notifier".into()))
        );
        assert_eq!(
            parse_chat_request("@jerith can I get the blue role?"),
            Some(Ask::Add("blue".into()))
        );
        assert_eq!(
            parse_chat_request("add me to the purple role please"),
            Some(Ask::Add("purple".into()))
        );
        assert_eq!(
            parse_chat_request("remove my Notifier role"),
            Some(Ask::Remove("notifier".into()))
        );
        assert_eq!(
            parse_chat_request("take off the blue role"),
            Some(Ask::Remove("blue".into()))
        );
        assert_eq!(
            parse_chat_request("what roles can I pick?"),
            Some(Ask::List)
        );
        assert_eq!(parse_chat_request("!role"), Some(Ask::List));
        assert_eq!(
            parse_chat_request("!role notifier"),
            Some(Ask::Add("notifier".into()))
        );
        assert_eq!(
            parse_chat_request("!role remove notifier"),
            Some(Ask::Remove("notifier".into()))
        );
        // Ordinary conversation is left to the model.
        assert_eq!(
            parse_chat_request("what's your role in this project?"),
            None
        );
        assert_eq!(parse_chat_request("nice role you played there"), None);
        assert_eq!(parse_chat_request("I like your role"), None);
    }

    #[test]
    fn missing_roles_become_an_owner_ping_not_a_creation() {
        let text = owner_request_text("Noodly", "Historian", Some(42));
        assert!(text.contains("<@42>"), "owner must be tagged: {text}");
        assert!(text.contains("Noodly"));
        assert!(text.contains("Historian"));
        assert!(text.contains("/namecolor"), "colour exception stated");
        // No owner configured → still no silent creation, honest text.
        let text = owner_request_text("Noodly", "Historian", None);
        assert!(text.contains("no owner configured"));
        assert!(text.contains("Historian"));
    }

    #[test]
    fn shelf_text_lists_roles_and_the_current_holdings() {
        let available = vec![row(1, "Notifier", 10, 0), row(2, "Blue", 20, 0)];
        let text = shelf_text(&available, &["Blue".to_string()], true);
        assert!(text.contains("`Notifier`") && text.contains("`Blue`"));
        assert!(text.contains("You're holding: `Blue`"));
        assert!(text.contains("/namecolor"));
        let empty = shelf_text(&[], &[], true);
        assert!(empty.contains("No self-assignable roles"));
    }

    #[test]
    fn arguments_are_cleaned_before_resolution() {
        assert_eq!(clean_name("  \"Blue Role\"  "), "Blue");
        assert_eq!(clean_name("notifier"), "notifier");
        assert_eq!(clean_name("roles"), "");
    }
}
