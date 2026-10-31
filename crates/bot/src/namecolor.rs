//! Name-colour roles, converted from the legacy OpenClaw picker
//! (`~/.openclaw/workspace/scripts/role_color_picker.py`) into what the
//! Jerith bot can do from Discord itself.
//!
//! The legacy design: a separate Python bot held a 15-swatch picker embed,
//! mapped hex → role id in a JSON file, created missing colour roles, stripped
//! the previous colour role, and wrote the choice back to the member's people
//! file. Two things made that stack feel absent in chat: it was another bot
//! (no shared conversation), and Jerith itself had no role tools, so a
//! natural-language "make my name purple" was answered with a refusal.
//!
//! This module keeps the working parts and drops the extra process:
//!
//! - the same 15 presets and natural-language colour names (`red`,
//!   `burnt orange`, `#cc5500`, …);
//! - live role resolution against the guild (by colour *and* by name), with
//!   the same "strip the old colour role" behaviour;
//! - missing colours can be created into the palette (`create_missing`), with
//!   an emoji-tagged name and a position at the bottom of the colour block;
//! - strictly self-service: a member may change **their own** colour and
//!   nothing else. Nobody — owner included — can repaint someone else from
//!   chat (no "target member" option, no permission bypass);
//! - the result is persisted to `jerith_people.user_profiles` (`name_color`),
//!   so `people.lookup` can answer "what colour does X use?" afterwards;
//! - a deterministic chat fast-path (`parse_chat_request`) so a mention/DM
//!   asking for a colour runs the same code instead of being routed to a
//!   model that has no role tools.
//!
//! Pure helpers are unit-tested; the live half is one function (`apply`).

use std::collections::HashMap;

use poise::serenity_prelude as serenity;
use serde::Deserialize;

use crate::commands::Data;

type Error = anyhow::Error;
type Ctx<'a> = poise::Context<'a, Data, Error>;

// ---------------------------------------------------------------------------
// Palette — carried over from the legacy color_utils.py
// ---------------------------------------------------------------------------

/// The 15 preset swatches (display name, 0xRRGGBB). Display names must stay
/// single words so role-name matching is unambiguous.
pub const PRESETS: &[(&str, u32)] = &[
    ("Red", 0xE74C3C),
    ("Orange", 0xE67E22),
    ("Yellow", 0xF1C40F),
    ("Green", 0x2ECC71),
    ("Blue", 0x3498DB),
    ("Purple", 0x9B59B6),
    ("Brown", 0x8B4513),
    ("Gray", 0x95A5A6),
    ("Black", 0x1A1A1A),
    ("White", 0xF0F0F0),
    ("Mint", 0x1ABC9C),
    ("Pink", 0xE91E90),
    ("Coral", 0xFF6B6B),
    ("Teal", 0x008080),
    ("Indigo", 0x4B0082),
];

/// Extra natural-language names (the legacy COLOR_NAME_MAP), including the
/// words the live guild already uses for its colour roles ("royal purple",
/// "hacker amber", "neon magenta", "electric cyan", …).
pub const NAMED: &[(&str, u32)] = &[
    ("burnt orange", 0xCC5500),
    ("dark red", 0x8B0000),
    ("dark green", 0x006400),
    ("dark blue", 0x00008B),
    ("navy", 0x000080),
    ("royal blue", 0x4169E1),
    ("royal purple", 0x7851A9),
    ("synthwave purple", 0x9D00FF),
    ("lavender", 0xB57EDC),
    ("violet", 0x71368A),
    ("magenta", 0xFF00FF),
    ("neon magenta", 0xFF00FF),
    ("cyan", 0x00FFFF),
    ("electric cyan", 0x00FFFF),
    ("electric blue", 0x00D4FF),
    ("aqua", 0x00FFFF),
    ("lime", 0x00FF00),
    ("toxic green", 0x39FF14),
    ("hacker amber", 0xFFB000),
    ("amber", 0xFFB000),
    ("gold", 0xFFD700),
    ("silver", 0xC0C0C0),
    ("rose", 0xFF007F),
    ("pink", 0xFF8A9D),
    ("maroon", 0x800000),
    ("olive", 0x808000),
    ("turquoise", 0x40E0D0),
    ("salmon", 0xFA8072),
    ("peach", 0xFFDAB9),
    ("cream", 0xFFFDD0),
    ("beige", 0xF5F5DC),
    ("tan", 0xD2B48C),
    ("chocolate", 0x7B3F00),
    ("coffee", 0x6F4E37),
    ("grey", 0x95A5A6),
    ("ruby red", 0xE0115F),
];

/// `[namecolor]` in bot.toml. On by default: it only touches cosmetic roles.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct NameColorConfig {
    /// Run the lane at all.
    pub enabled: bool,
    /// Create a palette role when the guild does not have that colour yet.
    /// Off → the lane only ever assigns roles that already exist.
    pub create_missing: bool,
    /// Word a colour role's name ends with ("Red Color"). Also used when
    /// naming a created role.
    pub color_word: String,
    /// Scope to one guild (None = every guild the bot is in).
    pub guild_id: Option<u64>,
}

impl Default for NameColorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            create_missing: true,
            color_word: "Color".to_string(),
            guild_id: None,
        }
    }
}

impl NameColorConfig {
    pub fn allows_guild(&self, guild_id: u64) -> bool {
        self.enabled && self.guild_id.map(|g| g == guild_id).unwrap_or(true)
    }
}

// ---------------------------------------------------------------------------
// Pure colour logic
// ---------------------------------------------------------------------------

/// Parse `#RGB`, `#RRGGBB`, `RGB` or `RRGGBB` (case-insensitive) into
/// 0xRRGGBB. Anything else is `None` — never a silent default colour.
pub fn parse_hex(input: &str) -> Option<u32> {
    let raw = input.trim().trim_start_matches('#');
    let raw = raw
        .strip_prefix("0x")
        .or(raw.strip_prefix("0X"))
        .unwrap_or(raw);
    if !raw.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    match raw.len() {
        3 => {
            let expanded: String = raw.chars().flat_map(|c| [c, c]).collect();
            u32::from_str_radix(&expanded, 16).ok()
        }
        6 => u32::from_str_radix(raw, 16).ok(),
        _ => None,
    }
}

/// Resolve a user's colour phrase to (display label, 0xRRGGBB).
pub fn resolve(query: &str) -> Option<(String, u32)> {
    let q = query.trim();
    if q.is_empty() {
        return None;
    }
    if let Some(rgb) = parse_hex(q) {
        return Some((label_for(rgb), rgb));
    }
    let lower = q.to_lowercase();
    for (name, rgb) in PRESETS.iter().chain(NAMED.iter()) {
        if name.to_lowercase() == lower {
            return Some((name.to_string(), *rgb));
        }
    }
    // "purple name" / "color: purple" style answers.
    let words: Vec<&str> = lower.split_whitespace().collect();
    for (name, rgb) in PRESETS.iter().chain(NAMED.iter()) {
        let nl = name.to_lowercase();
        let parts: Vec<&str> = nl.split_whitespace().collect();
        if lower.contains(&nl) || (parts.len() == 1 && words.contains(&parts[0])) {
            return Some((name.to_string(), *rgb));
        }
    }
    None
}

/// Palette label for a colour ("Purple", "Burnt Orange"); custom hexes get
/// an uppercase hex label ("#FF6600").
pub fn label_for(rgb: u32) -> String {
    for (name, preset) in PRESETS.iter().chain(NAMED.iter()) {
        if *preset == rgb {
            return name.to_string();
        }
    }
    format!("#{:06X}", rgb & 0xFFFFFF)
}

/// Hue/lightness emoji, ported from the legacy color_to_emoji. Used for
/// created role names and replies.
pub fn emoji_for(rgb: u32) -> &'static str {
    let r = ((rgb >> 16) & 0xFF) as f64 / 255.0;
    let g = ((rgb >> 8) & 0xFF) as f64 / 255.0;
    let b = (rgb & 0xFF) as f64 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    // Near-grayscale: brightness decides black/mid/white.
    if delta < 0.15 {
        let lightness = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        return if lightness < 0.2 {
            "⚫"
        } else if lightness > 0.8 {
            "⚪"
        } else {
            "🩶"
        };
    }
    let hue = if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    let hue = (hue + 360.0) % 360.0;
    match hue {
        h if h < 15.0 || h >= 345.0 => "🔴",
        h if h < 45.0 => "🟠",
        h if h < 70.0 => "🟡",
        h if h < 150.0 => "🟢",
        h if h < 260.0 => "🔵",
        h if h < 300.0 => "🟣",
        h if h < 330.0 => "🌸",
        _ => "🟤",
    }
}

/// "🟣 Purple Color" (the legacy naming, which `is_color_role_name` also
/// recognizes), or "⚪ custom FF6600 Color" for off-palette hexes.
pub fn role_name_for(rgb: u32, word: &str) -> String {
    let label = label_for(rgb);
    format!("{} {} {}", emoji_for(rgb), label, word)
}

/// Colour words used to recognise a picker role by bare name ("pink",
/// "royal purple", "electric cyan"). Deliberately colour-only: staff/topic
/// roles that merely happen to have a colour must never be stripped.
fn is_color_word(name: &str) -> bool {
    let lower = name.trim().to_lowercase();
    if lower.is_empty() {
        return false;
    }
    PRESETS
        .iter()
        .chain(NAMED.iter())
        .any(|(n, _)| n.to_lowercase() == lower)
        .then_some(())
        .map(|_| true)
        .unwrap_or_else(|| lower.ends_with(" amber") || lower.starts_with("custom "))
}

/// Is this role name one the colour picker owns (and may therefore strip or
/// reuse)? True for the legacy "… Color" family, bare palette words, and
/// `custom <hex>`; false for "No Role Selected", "Unverified", "bots",
/// "Head Hanchos", etc.
pub fn is_color_role_name(name: &str) -> bool {
    let lower = name.trim().to_lowercase();
    if lower.is_empty() || lower == "@everyone" {
        return false;
    }
    if lower.ends_with(" color") || lower.ends_with(" colour") {
        return true;
    }
    if let Some(rest) = lower.strip_prefix("custom ") {
        return parse_hex(rest).is_some();
    }
    // Bare palette names, including multi-word ones the guild uses
    // ("royal purple", "hacker amber", "synthwave purple").
    if is_color_word(&lower) {
        return true;
    }
    let words: Vec<&str> = lower.split_whitespace().collect();
    words.len() >= 2 && is_color_word(words[words.len() - 1])
}

/// One role row (serenity's Role boiled down to what the planning logic
/// needs — keeps the pure half testable without a Discord connection).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleRow {
    pub id: u64,
    pub name: String,
    pub rgb: u32,
    pub position: u16,
    pub managed: bool,
}

impl RoleRow {
    pub fn is_color_role(&self) -> bool {
        self.rgb != 0 && !self.managed && is_color_role_name(&self.name)
    }
}

/// Pick the row that best represents `rgb` + `label`: an exact hex match on a
/// picker role first (name closeness breaks ties), then a name match.
pub fn pick_role<'a>(rows: &'a [RoleRow], label: &str, rgb: u32) -> Option<&'a RoleRow> {
    let label_lower = label.to_lowercase();
    let mut exact: Vec<&RoleRow> = rows
        .iter()
        .filter(|r| r.is_color_role() && r.rgb == rgb)
        .collect();
    exact.sort_by_key(|r| {
        let n = r.name.to_lowercase();
        // Prefer the row whose name carries the requested word.
        (!n.contains(&label_lower)) as u8
    });
    if let Some(row) = exact.first() {
        return Some(row);
    }
    rows.iter()
        .filter(|r| r.is_color_role())
        .find(|r| r.name.to_lowercase().contains(&label_lower))
}

/// The plan for one change: the role to add and the picker roles to remove.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub add: u64,
    pub remove: Vec<u64>,
}

pub fn plan_change(rows: &[RoleRow], target: u64) -> Plan {
    Plan {
        add: target,
        remove: rows
            .iter()
            .filter(|r| r.is_color_role() && r.id != target)
            .map(|r| r.id)
            .collect(),
    }
}

/// Best position for a newly created colour role: the lowest existing picker
/// role's position (the legacy mapper put custom roles at the bottom of the
/// colour block). `None` when the guild has no picker roles yet.
pub fn new_role_position(rows: &[RoleRow]) -> Option<u16> {
    rows.iter()
        .filter(|r| r.is_color_role())
        .map(|r| r.position)
        .min()
}

/// Palette help text (also the reply when `/namecolor` has no colour).
pub fn palette_text(current: Option<&str>) -> String {
    let swatches: Vec<String> = PRESETS
        .iter()
        .map(|(name, rgb)| format!("{} {}", emoji_for(*rgb), name.to_lowercase()))
        .collect();
    format!(
        "🎨 **Name colours** — {} colors\n{}\n\
         Use `/namecolor <colour>` — this only ever changes **your own** name.\n\
         Custom hexes work too: `/namecolor #ff6600`.\n\
         Current: {}",
        PRESETS.len(),
        swatches.join(" · "),
        current.unwrap_or("none set")
    )
}

// ---------------------------------------------------------------------------
// Chat fast-path — the piece that stops the "I can't assign roles" answer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatAsk {
    /// Show the palette (and the asker's current colour).
    List,
    /// Apply this colour phrase.
    Set(String),
}

/// Recognise a name-colour request in a chat message. Conservative on
/// purpose: it needs the explicit phrase ("name color", "colour role", …) or
/// the legacy `!color` command, plus an intent verb — a passing mention of
/// colours in normal conversation is left to the model.
pub fn parse_chat_request(text: &str) -> Option<ChatAsk> {
    let trimmed = text.trim();
    let lower = trimmed.to_lowercase();
    for prefix in [
        "!namecolor",
        "!namecolour",
        "!color",
        "!colour",
        "!identity",
    ] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let rest = rest.trim();
            if rest.is_empty() {
                return Some(ChatAsk::List);
            }
            return Some(ChatAsk::Set(rest.to_string()));
        }
    }
    // Explicit picker phrases may answer with the palette even when no
    // colour was named. A looser "name" mention ("make my name purple",
    // "I want a blue name") only acts when a colour is actually present.
    let explicit = [
        "name colour",
        "name color",
        "colour role",
        "color role",
        "my colour",
        "my color",
    ]
    .iter()
    .any(|p| lower.contains(p));
    if !explicit && !lower.contains("name") {
        return None;
    }
    let verb = [
        "make", "change", "set", "switch", "give", "want", "wanna", "get", "pick", "choose",
        "apply", "update", "use", "be ",
    ]
    .iter()
    .any(|v| lower.contains(v));
    if !verb {
        return None;
    }
    // Prefer an explicit hex, then the longest colour-name match.
    if let Some(pos) = lower.find('#') {
        let fragment: String = lower[pos..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '#')
            .collect();
        if let Some(rgb) = parse_hex(&fragment) {
            return Some(ChatAsk::Set(format!("#{:06X}", rgb)));
        }
    }
    let mut best: Option<&str> = None;
    for (name, _) in PRESETS.iter().chain(NAMED.iter()) {
        let n = name.to_lowercase();
        if lower.contains(&n) && best.map(|b| b.len() < n.len()).unwrap_or(true) {
            best = Some(name);
        }
    }
    match best {
        Some(name) => Some(ChatAsk::Set(name.to_string())),
        // "make my name <something I don't know>" stays the model's turn;
        // only an explicit picker phrase falls back to the palette.
        None if explicit => Some(ChatAsk::List),
        None => None,
    }
}

// ---------------------------------------------------------------------------
// Live half
// ---------------------------------------------------------------------------

fn rows_of(roles: &HashMap<serenity::RoleId, serenity::Role>) -> Vec<RoleRow> {
    roles
        .values()
        .map(|r| RoleRow {
            id: r.id.get(),
            name: r.name.clone(),
            rgb: r.colour.0 & 0xFFFFFF,
            position: r.position,
            managed: r.managed,
        })
        .collect()
}

/// Sum a member's guild permissions (role bits + @everyone), plus their top
/// role position.
pub(crate) fn member_permissions(
    member: &serenity::Member,
    roles: &HashMap<serenity::RoleId, serenity::Role>,
    guild_id: serenity::GuildId,
) -> (serenity::Permissions, u16) {
    let everyone = roles
        .values()
        .find(|r| r.id.get() == guild_id.get())
        .map(|r| r.permissions)
        .unwrap_or_default();
    let mut perms = everyone;
    let mut top = 0;
    for rid in &member.roles {
        if let Some(role) = roles.get(rid) {
            perms |= role.permissions;
            top = top.max(role.position);
        }
    }
    (perms, top)
}

pub(crate) fn held_colors(member: &serenity::Member, rows: &[RoleRow]) -> Vec<RoleRow> {
    let ids: Vec<u64> = member.roles.iter().map(|r| r.get()).collect();
    rows.iter()
        .filter(|r| ids.contains(&r.id) && r.is_color_role())
        .cloned()
        .collect()
}

/// Apply one colour to `member` — always the caller. There is no target
/// parameter: repainting another member is not a capability this bot has
/// (owner included), so the signature cannot offer one.
pub async fn apply(
    http: &serenity::Http,
    guild_id: serenity::GuildId,
    bot_id: serenity::UserId,
    member: &serenity::Member,
    query: &str,
    cfg: &crate::config::BotConfig,
) -> anyhow::Result<String> {
    anyhow::ensure!(
        cfg.namecolor.enabled,
        "name colours are switched off ([namecolor] enabled = false)"
    );
    anyhow::ensure!(
        cfg.namecolor.allows_guild(guild_id.get()),
        "name colours are scoped to a different server ([namecolor] guild_id)"
    );
    let Some((label, rgb)) = resolve(query) else {
        anyhow::bail!(
            "I don't know the colour `{}` — try one of the presets or a hex like #ff6600.\n{}",
            query.trim(),
            palette_text(None)
        );
    };
    let roles = guild_id.roles(http).await?;
    let rows = rows_of(&roles);
    // The bot's own authority: Manage Roles + a role above the target role.
    let bot_member = guild_id.member(http, bot_id).await?;
    let (bot_perms, bot_top) = member_permissions(&bot_member, &roles, guild_id);
    anyhow::ensure!(
        bot_perms.contains(serenity::Permissions::ADMINISTRATOR)
            || bot_perms.contains(serenity::Permissions::MANAGE_ROLES),
        "I don't have **Manage Roles** in this server, so I can't set colours — an admin can grant it"
    );

    let mut rows = rows;
    let target_role = match pick_role(&rows, &label, rgb) {
        Some(row) => row.clone(),
        None => {
            anyhow::ensure!(
                cfg.namecolor.create_missing,
                "no `{}` colour role exists in this server and creating new ones is off (`[namecolor] create_missing = false`)",
                label
            );
            let name = role_name_for(rgb, &cfg.namecolor.color_word);
            let created = guild_id
                .create_role(
                    http,
                    serenity::EditRole::new()
                        .name(name)
                        .colour(serenity::Colour(rgb))
                        .permissions(serenity::Permissions::empty())
                        .mentionable(false),
                )
                .await?;
            let position = new_role_position(&rows);
            rows.push(RoleRow {
                id: created.id.get(),
                name: created.name.clone(),
                rgb: created.colour.0 & 0xFFFFFF,
                position: created.position,
                managed: created.managed,
            });
            if let Some(pos) = position {
                // Best-effort ordering: Discord returns an error when the
                // position outranks the bot, but the role still exists.
                if let Err(e) = guild_id.edit_role_position(http, created.id, pos).await {
                    tracing::debug!(error = %e, role = %created.name, "colour role created but not repositioned");
                }
            }
            tracing::info!(target: "bot", role = %created.name, hex = format!("#{:06X}", rgb), "created colour role from chat");
            rows.iter()
                .find(|r| r.id == created.id.get())
                .cloned()
                .expect("created role is in the row set")
        }
    };
    anyhow::ensure!(
        target_role.position < bot_top || bot_top == 0,
        "`{}` sits above my highest role, so Discord won't let me assign it — an admin needs to move it below mine",
        target_role.name
    );

    let plan = plan_change(&rows, target_role.id);
    let before = held_colors(member, &rows);
    let already = before.iter().any(|r| r.id == target_role.id);
    if !already {
        member
            .add_role(http, serenity::RoleId::new(target_role.id))
            .await?;
    }
    let mut removed = 0usize;
    for rid in &plan.remove {
        if before.iter().any(|r| r.id == *rid) {
            member
                .remove_role(http, serenity::RoleId::new(*rid))
                .await?;
            removed += 1;
        }
    }
    tracing::info!(
        target: "bot",
        member = member.user.id.get(),
        role = %target_role.name,
        hex = format!("#{:06X}", rgb),
        removed,
        "name colour applied"
    );
    // Persist the preference so people.lookup can answer it later (the
    // legacy people-file update, now the shared registry).
    if let Some(pg) = cfg.postgres.as_ref() {
        let value = format!("{} {} (#{:06X})", emoji_for(rgb), label, rgb);
        if let Err(e) =
            crate::people::remember_name_color(pg, &member.user.id.get().to_string(), &value).await
        {
            tracing::debug!(error = %e, "name colour not persisted to jerith_people");
        }
    }
    Ok(format!(
        "🎨 Your name colour is now {} **{}** `#{:06X}`\nrole: `{}`{} · https://www.colorhexa.com/{:06x}",
        emoji_for(rgb),
        label,
        rgb,
        target_role.name,
        if already && removed == 0 {
            " (already set)"
        } else if removed > 0 {
            " · previous colour role cleared"
        } else {
            ""
        },
        rgb
    ))
}

/// The chat entry point shared by the mention/DM fast-path. Never returns an
/// error: the reply itself carries the honest refusal when something is
/// missing (permission, hierarchy, unknown colour).
pub async fn handle_chat_request(
    http: &serenity::Http,
    guild_id: serenity::GuildId,
    author_id: serenity::UserId,
    ask: ChatAsk,
    cfg: &crate::config::BotConfig,
) -> String {
    if !cfg.namecolor.allows_guild(guild_id.get()) {
        return "🎨 name colours aren't enabled in this server.".to_string();
    }
    let member = match guild_id.member(http, author_id).await {
        Ok(m) => m,
        Err(e) => return format!("🎨 I couldn't read your member record here ({e})."),
    };
    match ask {
        ChatAsk::List => {
            let roles = guild_id.roles(http).await.unwrap_or_default();
            let current = held_colors(&member, &rows_of(&roles))
                .first()
                .map(|r| r.name.clone());
            palette_text(current.as_deref())
        }
        ChatAsk::Set(query) => {
            let bot_id = match http.get_current_user().await {
                Ok(u) => u.id,
                Err(e) => return format!("🎨 I couldn't confirm my own identity ({e})."),
            };
            match apply(http, guild_id, bot_id, &member, &query, cfg).await {
                Ok(text) => text,
                Err(e) => format!("🎨 {e}"),
            }
        }
    }
}

/// `/namecolor [colour]` — your own name colour only (no member argument).
// Also `!namecolor`, `!color`, `!colour`.
#[poise::command(
    slash_command,
    prefix_command,
    guild_only,
    aliases("color", "colour", "namecolour")
)]
pub async fn namecolor(
    ctx: Ctx<'_>,
    #[description = "Colour name or #hex (blank lists the palette)"] color: Option<String>,
) -> Result<(), Error> {
    let Some(guild_id) = ctx.guild_id() else {
        ctx.say("🎨 name colours are per-server — run this inside the server.")
            .await?;
        return Ok(());
    };
    let http = ctx.serenity_context().http.clone();
    let author = guild_id.member(&http, ctx.author().id).await?;
    let cfg = ctx.data().cfg.clone();
    match color {
        None => {
            let roles = guild_id.roles(&http).await.unwrap_or_default();
            let current = held_colors(&author, &rows_of(&roles))
                .first()
                .map(|r| r.name.clone());
            ctx.say(palette_text(current.as_deref())).await?;
        }
        Some(query) => {
            match apply(
                &http,
                guild_id,
                ctx.framework().bot_id(),
                &author,
                &query,
                &cfg,
            )
            .await
            {
                Ok(text) => ctx.say(text).await?,
                Err(e) => ctx.say(format!("🎨 {e}")).await?,
            };
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_parsing_accepts_the_legacy_forms() {
        assert_eq!(parse_hex("#ff7700"), Some(0xFF7700));
        assert_eq!(parse_hex("ff7700"), Some(0xFF7700));
        assert_eq!(parse_hex("#F70"), Some(0xFF7700));
        assert_eq!(parse_hex("0xff7700"), Some(0xFF7700));
        for bad in ["", "#", "#ff77", "#ff77001", "blurple", "#xyz123"] {
            assert_eq!(parse_hex(bad), None, "{bad:?} must not parse");
        }
    }

    #[test]
    fn palette_and_names_resolve_like_the_old_picker() {
        assert_eq!(PRESETS.len(), 15);
        assert_eq!(resolve("purple"), Some(("Purple".into(), 0x9B59B6)));
        assert_eq!(resolve("PURPLE"), Some(("Purple".into(), 0x9B59B6)));
        assert_eq!(
            resolve("burnt orange"),
            Some(("burnt orange".into(), 0xCC5500))
        );
        assert_eq!(resolve("#cc5500"), Some(("burnt orange".into(), 0xCC5500)));
        assert_eq!(
            resolve("royal purple"),
            Some(("royal purple".into(), 0x7851A9))
        );
        assert_eq!(resolve("blurple"), None);
    }

    #[test]
    fn colour_role_naming_matches_the_legacy_family() {
        assert_eq!(role_name_for(0xE74C3C, "Color"), "🔴 Red Color");
        assert_eq!(role_name_for(0xFF6600, "Color"), "🟠 #FF6600 Color");
    }

    #[test]
    fn only_picker_roles_count_as_colour_roles() {
        for name in [
            "🔴 Red Color",
            "🟡 🟡 Yellow Color",
            "pink",
            "royal purple",
            "burnt orange",
            "hacker amber",
            "electric cyan",
            "custom ff6600",
        ] {
            assert!(is_color_role_name(name), "{name} is a picker role");
        }
        for name in [
            "No Role Selected",
            "Unverified",
            "Jerith Verified",
            "bots",
            "Head Hanchos",
            "AI Entities",
            "@everyone",
        ] {
            assert!(!is_color_role_name(name), "{name} must never be stripped");
        }
    }

    #[test]
    fn role_selection_and_plan_keep_one_colour() {
        let rows = vec![
            RoleRow {
                id: 1,
                name: "pink".into(),
                rgb: 0xFF8A9D,
                position: 66,
                managed: false,
            },
            RoleRow {
                id: 2,
                name: "red".into(),
                rgb: 0xFF0000,
                position: 65,
                managed: false,
            },
            RoleRow {
                id: 3,
                name: "No Role Selected".into(),
                rgb: 0x979C9F,
                position: 10,
                managed: false,
            },
            RoleRow {
                id: 4,
                name: "🟣 Purple Color".into(),
                rgb: 0x9B59B6,
                position: 1,
                managed: false,
            },
        ];
        let picked = pick_role(&rows, "Purple", 0x9B59B6).expect("purple role");
        assert_eq!(picked.id, 4);
        let plan = plan_change(&rows, picked.id);
        assert_eq!(plan.add, 4);
        assert!(plan.remove.contains(&1) && plan.remove.contains(&2));
        assert!(!plan.remove.contains(&3), "staff marker roles stay");
        assert!(!plan.remove.contains(&4), "the target is never removed");
        assert_eq!(new_role_position(&rows), Some(1));
    }

    #[test]
    fn chat_requests_are_recognised_but_conversation_is_not_hijacked() {
        assert_eq!(
            parse_chat_request("make my name purple"),
            Some(ChatAsk::Set("Purple".into()))
        );
        assert_eq!(
            parse_chat_request("@jerith can you change my name color to burnt orange?"),
            Some(ChatAsk::Set("burnt orange".into()))
        );
        assert_eq!(
            parse_chat_request("set my name color to #ff6600"),
            Some(ChatAsk::Set("#FF6600".into()))
        );
        assert_eq!(parse_chat_request("!color"), Some(ChatAsk::List));
        assert_eq!(
            parse_chat_request("!namecolor royal purple"),
            Some(ChatAsk::Set("royal purple".into()))
        );
        assert_eq!(
            parse_chat_request("I want a blue name"),
            Some(ChatAsk::Set("Blue".into()))
        );
        // A colour mentioned without a colour-change request → model's turn.
        assert_eq!(parse_chat_request("what colour is the sky?"), None);
        assert_eq!(parse_chat_request("pink is a nice colour"), None);
        assert_eq!(parse_chat_request("my name is purple"), None);
        assert_eq!(parse_chat_request("what's my name color?"), None);
    }

    #[test]
    fn palette_text_names_every_preset_and_the_current_role() {
        let text = palette_text(None);
        for (name, _) in PRESETS {
            assert!(
                text.to_lowercase().contains(&name.to_lowercase()),
                "missing {name}"
            );
        }
        assert!(text.contains("none set"));
        assert!(palette_text(Some("Purple Color")).contains("Purple Color"));
    }
}
