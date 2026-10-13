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
mod chat_fallback;
mod commands;
mod config;
mod state;

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

    let cfg = BotConfig::load()?;
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
                commands::status(),
                commands::peers(),
                commands::mine(),
                commands::leaderboard(),
                commands::myjercraft(),
                commands::help(),
            ],
            event_handler: |ctx, event| {
                Box::pin(chat_fallback::handle(ctx, event))
            },
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
                Ok(Data { cfg, state })
            })
        })
        .build();

    let mut client =
        serenityall::ClientBuilder::new(token, intents).framework(framework).await?;
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
