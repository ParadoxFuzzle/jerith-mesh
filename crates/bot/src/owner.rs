//! Owner allowlist + command classification — port of OpenClaw's trust
//! model (`commands.ownerAllowFrom` + the security-audit default-deny
//! stance), tuned for the mesh bot:
//!
//! 1. **Default-deny classification.** Every slash/prefix command is
//!    explicitly classified as user-level in [`USER_COMMANDS`]. Anything
//!    not listed there (a command added later, without a classification)
//!    is refused *for everyone* until it is classified — the same rule
//!    OpenClaw's security audit applies when commands ship without an
//!    owner allowlist ("commands will be rejected for everyone").
//! 2. **System-level actions are owner-DM only.** A command that escapes
//!    the user-level list can only run when the invocation is a DM and
//!    the author is the bot owner ([`borrow_owner_from_openclaw`] /
//!    `[owner] discord_id` in bot.toml / `$JERITH_BOT_OWNER_ID`).
//!    Guild messages never unlock it, not even the owner's — matches the
//!    user's rule that the bot never executes system-level changes
//!    outside an owner DM.
//! 3. **The conversational paths stay open.** Chat (via `/ask` or the
//!    @mention/DM fallback, with the web skills as its only tools) is
//!    read-only user interaction, not a system change — everyone may use
//!    it, as before.
//!
//! Enforcement lives in main.rs's `command_check`, so no command body can
//! bypass it and no future command can ride in unclassified.

use std::path::Path;

/// Where the OpenClaw stack stored its owner allowlist (read-only borrow;
/// the mesh bot never writes to ~/.openclaw).
pub const OPENCLAW_CONFIG_PATH: &str = ".openclaw/openclaw.json";

/// Classified user-level commands — the enumeration must stay in lockstep
/// with the `commands::…()` list registered in main.rs. Everything here is
/// either read-only (`status`, `peers`, `leaderboard`, `myjercraft`,
/// `help`) or confined to the caller's own data (own chat, own memory
/// notes, own JER wallet via `mine`; chain mining is a per-user wallet
/// feature, not node/host control).
pub const USER_COMMANDS: [&str; 12] = [
    "ask",
    "reset",
    "remember",
    "recall",
    "briefing",
    "lore",
    "status",
    "peers",
    "mine",
    "leaderboard",
    "myjercraft",
    "help",
];

/// Default-deny classification: `true` only for the explicitly listed
/// user-level commands. Accepts the bare name or poise's qualified form
/// (group-qualified names trace back to their root, which is what the
/// list stores).
pub fn command_is_user_level(qualified_or_base: &str) -> bool {
    let name = qualified_or_base
        .rsplit(|c| c == ':' || c == ' ')
        .next()
        .unwrap_or(qualified_or_base);
    USER_COMMANDS.contains(&name)
}

/// Outcome of the framework gate.
pub enum Gate {
    /// Command may run.
    Allow,
    /// Command refused; carries the user-facing text to send.
    Deny(String),
}

/// OpenClaw-rule decision for one command invocation. `dm` = invoked in a
/// direct message, `author_id` = invoking Discord user, `owner_id` =
/// resolved bot owner (None → no classified owner exists → default-deny).
pub fn gate_command(
    dm: bool,
    author_id: u64,
    owner_id: Option<u64>,
    command: &str,
) -> Gate {
    if command_is_user_level(command) {
        return Gate::Allow;
    }
    match owner_id {
        Some(id) if dm && author_id == id => Gate::Allow,
        _ => Gate::Deny(format!(
            "🛡️ `{command}` is a system-level action — owner-DM only \
             (OpenClaw trust model ported to jerith-mesh). Direct-message me \
             from the bot owner's account to use it."
        )),
    }
}

/// Borrow the owner's Discord id from the OpenClaw config
/// (`commands.ownerAllowFrom: ["discord:<id>", …]` / bare ids accepted).
/// Read-only, same borrow pattern as the bot token migration.
pub fn borrow_owner_from_openclaw(path: &Path) -> Option<u64> {
    let text = std::fs::read_to_string(path).ok()?;
    let cfg: serde_json::Value = serde_json::from_str(&text).ok()?;
    let list = cfg.pointer("/commands/ownerAllowFrom")?.as_array()?;
    for entry in list {
        let raw = entry.as_str()?;
        let id = raw.strip_prefix("discord:").filter(|s| !s.is_empty()).unwrap_or(raw);
        if let Ok(parsed) = id.parse::<u64>() {
            return Some(parsed);
        }
    }
    None
}

/// Default borrow source for the current install (env `HOME`).
pub fn openclaw_config_file() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    Some(Path::new(&home).join(OPENCLAW_CONFIG_PATH))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_counts_the_registered_set() {
        // Every main.rs-registered command is in the list (10 commands).
        for (i, cmd) in USER_COMMANDS.iter().enumerate() {
            assert!(command_is_user_level(cmd), "{cmd} must classify");
            let _ = i;
        }
        // Qualified forms resolve to their root.
        assert!(command_is_user_level("leaderboard"));
        assert!(command_is_user_level("myjercraft"));
        assert!(!command_is_user_level("shutdown"));
        assert!(!command_is_user_level("runexec"));
        assert!(!command_is_user_level("guild-admin:demote"));
        assert!(!command_is_user_level(""));
    }

    #[test]
    fn user_level_commands_ignore_the_dm_owner_rule() {
        for cmd in USER_COMMANDS {
            let g = gate_command(false, 1, Some(2), cmd);
            assert!(matches!(g, Gate::Allow), "{cmd} must be open to all");
        }
    }

    #[test]
    fn privileged_requires_owner_dm() {
        // Owner in DM → allowed.
        assert!(matches!(
            gate_command(true, 28547119782585964, Some(28547119782585964), "shutdown"),
            Gate::Allow
        ));
        // Owner in a guild channel → refused (DM-only, stricter than upstream
        // which accepted owner messages anywhere).
        assert!(matches!(
            gate_command(false, 28547119782585964, Some(28547119782585964), "shutdown"),
            Gate::Deny(_)
        ));
        // Someone else in DM → refused.
        assert!(matches!(
            gate_command(true, 111, Some(28547119782585964), "shutdown"),
            Gate::Deny(_)
        ));
        // Owner unconfigured → refused for everyone (default-deny).
        assert!(matches!(
            gate_command(true, 28547119782585964, None, "shutdown"),
            Gate::Deny(_)
        ));
    }

    #[test]
    fn denies_carry_user_facing_text_and_the_command_name() {
        let Gate::Deny(text) = gate_command(false, 1, Some(2), "runexec") else {
            panic!("expected denial");
        };
        assert!(text.contains("runexec"));
        assert!(text.contains("owner-DM only"));
        assert!(text.contains("🛡️"));
    }

    #[test]
    fn owner_borrow_parses_openclaw_allowlist() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("owner-test-{}.json", std::process::id()));
        std::fs::write(
            &path,
            r#"{"commands": {"ownerAllowFrom": ["discord:28547119782585964", "user-not-a-number"]}}"#,
        )
        .unwrap();
        assert_eq!(borrow_owner_from_openclaw(&path), Some(28547119782585964));
        // Bare id in the list also accepted.
        std::fs::write(
            &path,
            r#"{"commands": {"ownerAllowFrom": ["28547119782585964"]}}"#,
        )
        .unwrap();
        assert_eq!(borrow_owner_from_openclaw(&path), Some(28547119782585964));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn owner_borrow_survives_missing_file_shape() {
        assert_eq!(borrow_owner_from_openclaw(Path::new("/nonexistent")), None);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("owner-test2-{}.json", std::process::id()));
        std::fs::write(&path, r#"{"commands": {"ownerAllowFrom": []}}"#).unwrap();
        assert_eq!(borrow_owner_from_openclaw(&path), None);
        let _ = std::fs::remove_file(&path);
    }
}
