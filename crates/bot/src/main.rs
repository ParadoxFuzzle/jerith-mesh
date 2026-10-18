//! `jerith-bot` — Discord front door for a jerith-mesh node.
//!
//! Slash commands call the node's gateway (`/v1/chat/completions`,
//! `/v1/skills/memory/*`, `/status`) and the local JerithChain node
//! (`/mine`). The bot adds no new wire protocol: the gateway already
//! does mesh-peer fallback for chat, so the bot inherits it.
//!
//! Run:
//!   JERITH_DISCORD_TOKEN=... jerith-bot
//! or put `discord_token = "..."` in `~/.jerith-mesh/bot.toml`.
//! Gateway and chain URLs default to localhost (8900 / 8322) and can be
//! overridden with `JERITH_GATEWAY_URL` / `JERITH_CHAIN_URL`.

mod api;
mod archiver;
mod chat_fallback;
mod commands;
mod config;
mod converse;
mod external;
mod gifs;
mod hooks;
mod mailbox;
mod monitor;
mod owner;
mod people;
mod plugins;
mod sandbox;
mod squeeze;
mod state;
mod workspace;

/// Shared error type of the framework closures (`command_check`,
/// `event_handler`, command bodies) — anyhow all the way down.
type Error = anyhow::Error;

/// Owner gate as the framework's `command_check` (poise gives no inferred
/// types here, so it's a named fn). See owner.rs: user-level commands run
/// for everyone; anything else — including unclassified future commands —
/// runs only in a DM from the bot owner (default-deny, OpenClaw trust
/// model ported).
fn owner_gate_check(
    ctx: poise::Context<'_, Data, Error>,
) -> poise::BoxFuture<'_, Result<bool, Error>> {
    Box::pin(async move {
        let cmd = ctx.command().qualified_name.clone();
        let dm = ctx.guild_id().is_none();
        let author = ctx.author().id.get();
        let owner = ctx.data().cfg.owner_discord_id;
        match owner::gate_command(dm, author, owner, &cmd) {
            owner::Gate::Allow => Ok(true),
            owner::Gate::Deny(text) => {
                tracing::warn!(
                    command = %cmd,
                    author,
                    dm,
                    "command refused by owner gate (owner-DM only)"
                );
                let _ = ctx.say(text).await;
                Ok(false)
            }
        }
    })
}

use poise::serenity_prelude as serenityall;

use commands::Data;
use config::BotConfig;
use state::new_state;

/// Read the JerithChain api token the same way the core miner does
/// (env `JERITH_TOKEN` first, then `keys/api.token`). Duplicated here
/// because the core helper is private, and the bot must not create or
/// depend on a full node config.
fn core_token() -> anyhow::Result<String> {
    if let Ok(t) = std::env::var("JERITH_TOKEN") {
        let t = t.trim().to_string();
        if !t.is_empty() {
            return Ok(t);
        }
    }
    let path = jerith_mesh_core::economics::default_token_file();
    match std::fs::read_to_string(&path) {
        Ok(s) => {
            let s = s.trim().to_string();
            if s.is_empty() {
                anyhow::bail!(
                    "no JerithChain api token — {} is empty; set JERITH_TOKEN",
                    path.display()
                );
            }
            Ok(s)
        }
        Err(e) => anyhow::bail!(
            "no JerithChain api token — cannot read {} ({e}); set JERITH_TOKEN",
            path.display()
        ),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,poise=warn")),
        )
        .init();

    let mut cfg = BotConfig::load()?;
    // Webhook event hooks (FEATURES §8): installed once; every emit site
    // goes through hooks::global() and is fire-and-forget.
    hooks::init(cfg.hooks.clone());
    if !cfg.hooks.urls.is_empty() {
        tracing::info!(urls = cfg.hooks.urls.len(), "webhook event hooks enabled");
    }
    // Persona: composed from the OpenClaw-style workspace canon docs
    // (~/.jerith-mesh/workspace/IDENTITY|SOUL|USER|USER_MODEL|AGENTS|MEMORY.md,
    // staged on first boot from the migrated OpenClaw workspace) when
    // neither `JERITH_PERSONA` nor bot.toml's `system_prompt` is set. The
    // flat `~/.jerith-mesh/persona.md` remains the fallback when the
    // workspace holds no canon docs.
    if cfg.system_prompt.is_none() {
        let (composed, staged) = workspace::load();
        if !staged.is_empty() {
            tracing::info!(
                staged = staged.join(", "),
                "workspace canon docs staged from the OpenClaw workspace"
            );
        }
        match composed {
            Some(p) => {
                let chars = p.chars().count();
                cfg.system_prompt = Some(p);
                tracing::info!(
                    files = staged.len(),
                    chars,
                    "persona composed from workspace canon docs"
                );
            }
            None => {
                let path = config::persona_file_path();
                if let Ok(raw) = std::fs::read_to_string(&path) {
                    cfg.system_prompt = Some(raw.trim().to_string());
                    let chars = cfg
                        .system_prompt
                        .as_ref()
                        .map(|s| s.chars().count())
                        .unwrap_or(0);
                    tracing::info!(file = %path.display(), chars, "persona system prompt loaded (flat fallback)");
                } else {
                    tracing::info!(
                        dir = %crate::workspace::workspace_dir().display(),
                        "no workspace canon and no persona file — bot chats without a persona"
                    );
                }
            }
        }
    }
    // Known users: the jerith_people PostgreSQL database, exported live by
    // the site cron (people + profile facts + trust levels + identity
    // links). Appended to whatever persona the bot ended up with — this is
    // the "reading its database" step the OpenClaw gateway did implicitly.
    {
        let path = config::known_users_file_path();
        if let Ok(raw) = std::fs::read_to_string(&path) {
            let section = format!(
                "\n\n## Known users (live export of the jerith_people database)\n\n{}",
                raw.trim()
            );
            cfg.system_prompt = Some(match cfg.system_prompt.take() {
                Some(existing) => format!("{existing}{section}"),
                None => section,
            });
            let chars = cfg
                .system_prompt
                .as_ref()
                .map(|s| s.chars().count())
                .unwrap_or(0);
            tracing::info!(file = %path.display(), chars, "known-users database view appended to persona");
        }
    }
    // Cap once, on the combined prompt (persona + known users together).
    if let Some(p) = cfg.system_prompt.as_mut() {
        *p = config::cap_system_prompt(p, config::PERSONA_MAX_CHARS);
    }
    let state = new_state();
    // Restore the persisted mine tally + chat history (best-effort: a
    // missing or corrupt file means a fresh tally, never a failed startup).
    let path = state::tally_path();
    match state.load_from_disk(&path) {
        Ok(n) if n > 0 => {
            tracing::info!(path = %path.display(), users = n, "restored mine tally")
        }
        Ok(_) => {}
        Err(e) => tracing::debug!(error = %e, "no persisted tally loaded"),
    }
    let token = cfg.discord_token.clone();
    tracing::info!(
        gateway = %cfg.gateway_url,
        chain = %cfg.chain_url,
        "starting jerith-bot"
    );

    let intents = serenityall::GatewayIntents::non_privileged();
    let framework = poise::Framework::builder()
        .options(poise::FrameworkOptions {
            commands: vec![
                commands::ask(),
                commands::reset(),
                commands::remember(),
                commands::recall(),
                commands::briefing(),
                commands::lore(),
                commands::status(),
                commands::peers(),
                // User-level in owner.rs: reads only attachments the
                // invoker supplies, through the same Docker sandbox
                // every chat attachment already rides.
                commands::read(),
                // Deliberately unclassified in owner.rs USER_COMMANDS:
                // inbox contents are owner's eyes only (default-deny gate).
                commands::mail(),
                commands::mine(),
                commands::leaderboard(),
                commands::myjercraft(),
                commands::gif(),
                // Guild-admin lever — deliberately unclassified in
                // owner.rs's USER_COMMANDS: the default-deny gate makes
                // it owner-DM only (system-level action).
                commands::timeout(),
                commands::help(),
            ],
            event_handler: |ctx, event| Box::pin(chat_fallback::handle(ctx, event)),
            // Owner gate (see owner_gate_check + owner.rs).
            command_check: Some(owner_gate_check),
            on_error: |err| {
                Box::pin(async move {
                    if let Err(e) = poise::builtins::on_error(err).await {
                        tracing::error!(error = %e, "poise on_error failed");
                    }
                })
            },
            ..Default::default()
        })
        .setup(move |ctx, _ready, framework| {
            Box::pin(async move {
                poise::builtins::register_globally(ctx, &framework.options().commands).await?;
                tracing::info!("slash commands registered globally");
                // Discord thread archiver (OpenClaw skill port): an
                // operator-configured background ingest — no commands, no
                // system-level change; subscribed in bot.toml [archiver].
                // Spawns after the gateway is registered so memory embeds
                // have a reachable target from the first sweep.
                if cfg.archiver.enabled {
                    let archiving = cfg.clone();
                    tokio::spawn(archiver::run_loop(archiving));
                }
                // Monitor scaffold (OpenClaw ops-monitor port): periodic
                // checks with a single owner DM when something fails.
                // Reports only — no restarts/no repairs (owner.rs holds
                // the system-level boundary).
                {
                    // serenity's Arc-based http handle, cloned for the loop.
                    if cfg.monitor.enabled {
                        let http_handle = ctx.http.clone();
                        let monitor_cfg = cfg.monitor.clone();
                        tokio::spawn(async move {
                            monitor::run_loop(http_handle, cfg.owner_discord_id, monitor_cfg).await;
                        });
                    }
                }
                Ok(Data { cfg, state })
            })
        })
        .build();

    let mut client = serenityall::ClientBuilder::new(token, intents)
        .framework(framework)
        .await?;
    client.start().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_token_fails_cleanly_without_env() {
        // Either JERITH_TOKEN / keys/api.token exist on this box (Ok), or the
        // bot fails with a clear message (Err). Never panic.
        let _ = core_token();
    }
}
