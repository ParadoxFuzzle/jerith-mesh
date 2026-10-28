//! Welcome-DM lane (`[welcome]` in bot.toml) — the OpenClaw
//! guild-watcher port (workspace/silly_welcome_bot.py →
//! ~/.openclaw/plugin-skills/guild-admin/guild_watcher.py), with byte-
//! identical welcome embeds and the migrated GIF pool:
//!
//! - one of three welcome embeds + a local red-panda GIF, DM'd to every
//!   human joining a `[welcome] guild_id` server
//! - embed/gif pick varies per member (no rand dependency; the picks are
//!   deterministic per join id, which is what a retried event wants)
//! - DMs-closed / missing GIF dir / send failures log debug and keep
//!   going — a welcome miss is never a bot crash (the watcher's own
//!   contract, or the join flood would have taken it down daily)
//! - the bot skips welcomings other bots (the watcher's `member.bot`
//!   guard)
//!
//! Dormant unless `[welcome] enabled = true` (config.rs), like every
//! other OpenClaw skill port ([archiver], [monitor]). Departs from the
//! watcher by replacing its heartbeat-file/cron-watchdog machinery with
//! this bot's systemd supervision (Restart=always) — the watcher was a
//! whole service unit approximating what jerith-bot already is.

use serde::Deserialize;
use std::path::PathBuf;

use poise::serenity_prelude as serenityall;

type Error = anyhow::Error;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// `[welcome]` in bot.toml. Off by default; `guild_id` scopes the DMs to
/// one server (the watcher served exactly one), `gif_dir` overrides the
/// migrated pool location.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct WelcomeConfig {
    /// Run the member-add lane.
    #[serde(default)]
    pub enabled: bool,
    /// The guild whose joins get a welcome DM. None sends to ANY guild
    /// the bot shares with a new member.
    #[serde(default)]
    pub guild_id: Option<u64>,
    /// Local GIF pool dir override. Default: the site-migrated workspace
    /// copy of the watcher's red-panda pool (the migration script copied
    /// workspace scripts/docs/config but NOT gif binaries — staged
    /// separately on this box, 2026-10-27).
    #[serde(default)]
    pub gif_dir: Option<String>,
}

impl WelcomeConfig {
    /// The effective GIF pool dir: `[welcome] gif_dir` beats the default
    /// `~/.jerith-mesh/workspace/gifworkshop/redpandas`.
    pub fn gif_dir(&self) -> PathBuf {
        match self.gif_dir.as_deref() {
            Some(d) => PathBuf::from(d),
            None => crate::workspace::workspace_dir()
                .join("gifworkshop")
                .join("redpandas"),
        }
    }
}

// ---------------------------------------------------------------------------
// Welcome embeds — carried over from the watcher (same titles, colors,
// copy; every variant still points at the roles channel guide).
// ---------------------------------------------------------------------------

/// One welcome embed's static data (the watcher's {title, description,
/// color} records; colors keep the colorhexa-cited palette values).
pub struct WelcomeEmbed {
    pub title: &'static str,
    pub description: &'static str,
    pub colour: u32,
}

pub const WELCOME_EMBEDS: &[WelcomeEmbed] = &[
    WelcomeEmbed {
        title: "🎉 Welcome to Creator Junction!",
        description: concat!(
            "Hey there! Welcome to **Creator Junction** — the community for creators of all kinds! 🐼\n\n",
            "To get started, head over to <#1502479727430996090> and **@mention Jerith** to:\n",
            "• 🎨 Get your **creator role** (artist, writer, developer, gamer, and more!)\n",
            "• 🌈 Pick a **name color** (pink, red, green, blue, purple, orange, violet, yellow, electric cyan)\n",
            "• ✅ Get verified and unlock all channels\n\n",
            "Just say something like:\n",
            "`@Jerith Hey! I'm a digital artist and I want a blue name`\n\n",
            "Jerith will get you sorted! 🐾"
        ),
        colour: 0xFF6B6B,
    },
    WelcomeEmbed {
        title: "🐼 Welcome to the Jungle!",
        description: concat!(
            "Hey! You just joined **Creator Junction** — let's get you set up! 🎪\n\n",
            "Head to <#1502479727430996090> and **@mention Jerith** to claim your:\n",
            "• 🎭 **Creator role** — show the world what you make!\n",
            "• 🎨 **Name color** — stand out in chat!\n",
            "• 🔓 **Full access** — unlock all the channels!\n\n",
            "Example message:\n",
            "`@Jerith Hi! I'm a game developer, can I get the gamedev role and purple?`\n\n",
            "See you in there! 🚀"
        ),
        colour: 0x9B59B6,
    },
    WelcomeEmbed {
        title: "✨ Welcome, Creator!",
        description: concat!(
            "Welcome to **Creator Junction**! 🐼✨ We're stoked to have you!\n\n",
            "To unlock everything, pop into <#1502479727430996090> and **@mention Jerith**.\n",
            "You can ask for:\n",
            "• 🖌️ **Creator roles** — artist, musician, writer, coder, streamer, and way more\n",
            "• 🌈 **Name colors** — pink, red, green, blue, purple, orange, violet, yellow, or electric cyan\n",
            "• ✅ **Verification** — get the green check and full access\n\n",
            "Try something like:\n",
            "`@Jerith Hey! I make music and I'd love the musician role with an orange name`\n\n",
            "Can't wait to see what you create! 🎵🎨🎮"
        ),
        colour: 0xF39C12,
    },
];

// ---------------------------------------------------------------------------
// Pure picks (testable without Discord)
// ---------------------------------------------------------------------------

/// Round-robin embed pick: a member id indexes the pool directly, so a
/// retried member-add event is stable and variety comes free.
pub fn pick_embed(member_id: u64) -> &'static WelcomeEmbed {
    &WELCOME_EMBEDS[(member_id % WELCOME_EMBEDS.len() as u64) as usize]
}

/// FNV-1a over the member id (same construction as transports::user_key)
/// salted by the pool length, indexing a SORTED file list — the pick is
/// deterministic per (id, pool snapshot) yet varies across joins. Empty
/// or unreadable pool → None (the embed still sends).
fn gif_index(member_id: u64, pool_len: usize) -> usize {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in member_id.to_le_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x1_0000_001b);
    }
    (h % pool_len as u64) as usize
}

/// One *gif* pool entry candidate: keep only files whose extension (not
/// the stem) reads `gif` case-insensitively.
fn gif_path(p: &std::path::Path) -> bool {
    p.is_file()
        && p.extension()
            .is_some_and(|x| x.to_str().is_some_and(|s| s.eq_ignore_ascii_case("gif")))
}

/// The pick: list the pool, sort once (dir order isn't stable across
/// snapshots), index by FNV. Unreadable dir or no gifs → None.
pub fn pick_gif(dir: &std::path::Path, member_id: u64) -> Option<PathBuf> {
    let mut gifs: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| gif_path(p))
        .collect();
    // Sorting must be case-SENSITIVE (byte order) so the FNV pick stays
    // deterministic per (id, pool); contents themselves are untouched.
    gifs.sort();
    if gifs.is_empty() {
        return None;
    }
    Some(gifs[gif_index(member_id, gifs.len())].clone())
}

/// Is this join welcome material at all? (the bot never welcomes
/// another bot, and the lane is config-gated).
pub fn should_welcome(cfg: &WelcomeConfig, guild_id: u64, author_bot: bool) -> bool {
    if !cfg.enabled || author_bot {
        return false;
    }
    match cfg.guild_id {
        Some(id) => id == guild_id,
        None => true,
    }
}

// ---------------------------------------------------------------------------
// Live lane
// ---------------------------------------------------------------------------

/// Called from chat_fallback's event handler on `GuildMemberAdd`. Does
/// the cheap checks, then spawns the DM side so the dispatch loop never
/// waits on a Discord round-trip for it.
pub fn on_member_add(
    http: std::sync::Arc<serenityall::Http>,
    cfg: WelcomeConfig,
    member: serenityall::Member,
) {
    let guild = member.guild_id.get();
    let bot = member.user.bot;
    let member_id = member.user.id.get();
    let who = member
        .user
        .global_name
        .clone()
        .unwrap_or_else(|| member.user.name.clone());
    if !should_welcome(&cfg, guild, bot) {
        return;
    }
    tokio::spawn(async move {
        match welcome_member(&http, &cfg, member_id).await {
            Ok(()) => tracing::info!(
                target: "bot",
                member = member_id,
                name = %who,
                guild,
                "welcome DM sent"
            ),
            Err(e) => tracing::debug!(
                target: "bot",
                member = member_id,
                error = %e,
                "welcome miss (DMs closed is normal)"
            ),
        }
    });
}

/// Build + send one welcome embed (+ optional GIF attachment). Errors
/// are typed up to the caller's log line — nothing panics here.
async fn welcome_member(
    http: &serenityall::Http,
    cfg: &WelcomeConfig,
    member_id: u64,
) -> Result<(), Error> {
    let data = pick_embed(member_id);
    let gif = pick_gif(&cfg.gif_dir(), member_id);
    let mut embed = serenityall::CreateEmbed::new()
        .title(data.title)
        .description(data.description)
        .colour(data.colour);
    if gif.is_some() {
        embed = embed.image("attachment://welcome.gif");
    }
    let mut msg = serenityall::CreateMessage::new().embed(embed);
    if let Some(path) = gif {
        // serenity's CreateAttachment::file takes tokio's File — open
        // async; a 1-7 MB gif read is still fast enough here.
        let file = tokio::fs::File::open(&path)
            .await
            .map_err(|e| anyhow::anyhow!("welcome gif unreadable {}: {e}", path.display()))?;
        let filename = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "welcome.gif".into());
        let attach = serenityall::CreateAttachment::file(&file, &filename)
            .await
            .map_err(|e| anyhow::anyhow!("welcome gif inlined {}: {}", path.display(), e))?;
        msg = msg.add_file(attach);
    }
    let channel = serenityall::UserId::new(member_id)
        .create_dm_channel(http)
        .await
        .map_err(|e| anyhow::anyhow!("dm channel open refused: {e}"))?;
    channel
        .id
        .send_message(http, msg)
        .await
        .map_err(|e| anyhow::anyhow!("welcome DM send refused: {e}"))?;
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_pick_cycles_and_is_stable() {
        let a = pick_embed(11);
        assert_eq!(a.title, WELCOME_EMBEDS[11 % 3].title);
        assert_eq!(a.title, pick_embed(11).title, "per-member pick is stable");
        // Variety: all three titles reachable across a spread of ids.
        let picks: std::collections::HashSet<&'static str> =
            (0..30u64).map(|i| pick_embed(i).title).collect();
        assert_eq!(picks.len(), 3, "all three embeds reachable: {picks:?}");
        // Copy preserved: every variant still points the roles guide.
        for e in WELCOME_EMBEDS {
            assert!(e.description.contains("1502479727430996090"));
        }
    }

    #[test]
    fn gif_pick_skips_empty_and_non_gif_dirs() {
        let dir = std::env::temp_dir().join(format!("jerith-welcome-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Empty pool → None.
        assert!(pick_gif(&dir, 1).is_none());
        // A non-gif file is not part of the pool.
        std::fs::write(dir.join("readme.txt"), "x").unwrap();
        assert!(pick_gif(&dir, 1).is_none());
        // Two gifs → deterministic per member, case-insensitive pool,
        // both reachable across joins.
        std::fs::write(dir.join("a.gif"), "a").unwrap();
        std::fs::write(dir.join("b.GIF"), "b").unwrap();
        let a = pick_gif(&dir, 42).unwrap();
        assert_eq!(
            pick_gif(&dir, 42),
            Some(a.clone()),
            "per-member pick is stable"
        );
        let both: std::collections::HashSet<String> = (0..50u64)
            .map(|i| {
                pick_gif(&dir, i)
                    .unwrap()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert!(
            both.contains("a.gif") && both.contains("b.GIF"),
            "case-insensitive pool: {both:?}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unreadable_pool_falls_back_to_embed_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir =
            std::env::temp_dir().join(format!("jerith-welcome-test-{}-2", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("x.gif"), "x").unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o000)).unwrap();
        assert!(pick_gif(&dir, 1).is_none(), "unlistable dir → None");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn should_welcome_honors_gate_and_bot_filter() {
        let mut cfg = WelcomeConfig::default();
        assert!(!should_welcome(&cfg, 1, false), "default off");
        cfg.enabled = true;
        assert!(should_welcome(&cfg, 1, false), "no guild id → any guild");
        assert!(!should_welcome(&cfg, 1, true), "never another bot");
        cfg.guild_id = Some(5);
        assert!(!should_welcome(&cfg, 1, false), "scoped guild filters");
        assert!(should_welcome(&cfg, 5, false));
    }

    #[test]
    fn default_pool_is_the_migrated_workspace_copy() {
        let cfg = WelcomeConfig::default();
        let dir = cfg.gif_dir();
        assert!(
            dir.ends_with(std::path::Path::new("gifworkshop").join("redpandas")),
            "{dir:?}"
        );
    }
}
