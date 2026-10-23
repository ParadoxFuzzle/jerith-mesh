//! `jerith-pet` — the mesh pet's runnable half, over the plugin SDK
//! (`crates/plugin-template`, docs/PLUGIN-SDK.md). The state machine
//! lives in `lib.rs` (pure, clock-injected, tested); this binary is
//! the wire glue:
//!
//! - reads its OWN `plugin.toml` for `[plugin] endpoint` + `secret`
//!   and binds that exact address — manifest and service cannot drift
//!   (the same one-source-of-truth rule as the example plugin);
//! - keeps ONE state file (`$JERITH_PET_STATE`, else
//!   `<home>/.jerith-mesh/pet/state.json`) loaded at boot, decayed
//!   against the wall clock lazily by each handler, saved atomically
//!   (tmp+rename) on every serving; a corrupt file refuses to boot —
//!   a life is not "restartable by accident";
//! - registers the five model-visible skills the manifest declares
//!   (`pet.look / talk / feed / play / rename`), with honest typed
//!   refusals the chat model reads and relays.
//!
//! Authoring note: this is the Phase 3d story's second citizen — copy
//! this layout, swap the handlers, sync `plugin.toml`, ship.

use anyhow::{Context as _, Result};
use jerith_pet_plugin::{PetState, load_or_create, state_path};
use jerith_plugin_template::{SkillServer, endpoint_addr, skill};
use serde_json::Value;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex as AsyncMutex;

/// now in ms — the only clock read on the wire; everything stateful in
/// the lib takes it as a parameter so tests drive time by hand.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// `[plugin] endpoint` + `[plugin] secret` (None = unsigned dev mode)
/// straight out of the manifest — one file rules both halves.
fn read_manifest(path: &str) -> Result<(std::net::SocketAddr, Option<String>)> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("reading manifest {path}"))?;
    // Document parse (toml::Table), NOT `str::parse::<toml::Value>` —
    // a Value FromStr only accepts a single value, so a whole manifest
    // string comes back «unexpected content, expected nothing».
    let file: toml::Table =
        toml::from_str(&raw).with_context(|| format!("{path}: not valid TOML"))?;
    let plugin = file
        .get("plugin")
        .and_then(toml::Value::as_table)
        .context("manifest has no [plugin] table")?;
    let endpoint = plugin
        .get("endpoint")
        .and_then(toml::Value::as_str)
        .context("[plugin] endpoint missing")?;
    let addr = endpoint_addr(endpoint)?;
    let secret = plugin
        .get("secret")
        .and_then(toml::Value::as_str)
        .map(str::to_string);
    Ok((addr, secret))
}

/// The five handlers share one tiny pattern: lock the pet, apply the
/// verb with the wall clock, save on success, hand the refusal
/// through on failure. A save failure is NOT swallowed — an unsaved
/// pet would quietly lie about what it ate.
async fn with_pet<F>(
    pet: &AsyncMutex<PetState>,
    path: &std::path::Path,
    act: F,
) -> Result<Value, String>
where
    F: FnOnce(&mut PetState, u64) -> Result<Value, String>,
{
    let now = now_ms();
    let mut st = pet.lock().await;
    let out = act(&mut st, now)?;
    st.save(path)
        .map_err(|e| format!("state save failed: {e}"))?;
    Ok(out)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let manifest = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "plugin.toml".to_string());
    let (addr, secret) = read_manifest(&manifest)?;
    tracing::info!(manifest, %addr, secret = secret.is_some(), "pet plugin starting");

    let path = state_path();
    let now = now_ms();
    let (pet, hatched) = load_or_create(&path, now)
        .with_context(|| format!("loading pet state {}", path.display()))?;
    let waking = if hatched {
        "hatched a new pet"
    } else {
        "the pet remembers"
    };
    tracing::info!(
        state = %path.display(),
        name = %pet.name,
        hatched = hatched,
        max_bond = pet.max_bond(),
        living = waking
    );
    let pet = Arc::new(AsyncMutex::new(pet));
    let path = Arc::new(path);

    SkillServer::new("pet", secret)
        .handler(
            "pet.look",
            skill({
                let pet = Arc::clone(&pet);
                let path = Arc::clone(&path);
                move |_args: Value| {
                    let pet = Arc::clone(&pet);
                    let path = Arc::clone(&path);
                    async move { with_pet(&pet, &path, |st, now| Ok(st.look(now))).await }
                }
            }),
        )
        .handler(
            "pet.talk",
            skill({
                let pet = Arc::clone(&pet);
                let path = Arc::clone(&path);
                move |args: Value| {
                    let pet = Arc::clone(&pet);
                    let path = Arc::clone(&path);
                    async move {
                        let note = args["note"].as_str().unwrap_or_default().to_string();
                        with_pet(&pet, &path, |st, now| st.talk(now, &note)).await
                    }
                }
            }),
        )
        .handler(
            "pet.feed",
            skill({
                let pet = Arc::clone(&pet);
                let path = Arc::clone(&path);
                move |_args: Value| {
                    let pet = Arc::clone(&pet);
                    let path = Arc::clone(&path);
                    async move { with_pet(&pet, &path, |st, now| st.feed(now)).await }
                }
            }),
        )
        .handler(
            "pet.play",
            skill({
                let pet = Arc::clone(&pet);
                let path = Arc::clone(&path);
                move |_args: Value| {
                    let pet = Arc::clone(&pet);
                    let path = Arc::clone(&path);
                    async move { with_pet(&pet, &path, |st, now| st.play(now)).await }
                }
            }),
        )
        .handler(
            "pet.rename",
            skill({
                let pet = Arc::clone(&pet);
                let path = Arc::clone(&path);
                move |args: Value| {
                    let pet = Arc::clone(&pet);
                    let path = Arc::clone(&path);
                    async move {
                        let name = args["name"].as_str().unwrap_or_default().to_string();
                        with_pet(&pet, &path, |st, now| st.rename(now, &name)).await
                    }
                }
            }),
        )
        .serve(addr)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SHIPPED manifest must stay servable by THIS binary: the
    /// endpoint has to bind (literal loopback IP per the drop contract),
    /// on a port that is not the example's, secret signed, and the
    /// declared skill rows must be exactly the five registered above.
    #[test]
    fn shipped_manifest_fits_this_binary() {
        let raw = include_str!("../plugin.toml");
        let file: toml::Table =
            toml::from_str(raw).expect("manifest is valid TOML (document parse)");
        let plugin = file
            .get("plugin")
            .expect("[plugin] table")
            .as_table()
            .expect("[plugin] is a table");
        assert_eq!(plugin.get("id").and_then(toml::Value::as_str), Some("pet"));
        let endpoint = plugin
            .get("endpoint")
            .and_then(toml::Value::as_str)
            .expect("endpoint line");
        let addr = endpoint_addr(endpoint).expect("endpoint binds");
        assert!(addr.ip().is_loopback(), "loopback by contract");
        assert_ne!(
            addr.port(),
            7788,
            "the example's port belongs to the example"
        );
        assert!(
            plugin.get("secret").and_then(toml::Value::as_str).is_some(),
            "the pet ships signed by default"
        );
        let mut names: Vec<&str> = file
            .get("skill")
            .and_then(toml::Value::as_array)
            .expect("[[skill]] rows")
            .iter()
            .filter_map(|s| s.get("name").and_then(toml::Value::as_str))
            .collect();
        names.sort_unstable();
        assert_eq!(
            names,
            vec!["feed", "look", "play", "rename", "talk"],
            "manifest skill rows = the five handlers"
        );
    }

    /// The serialization the state file relies on round-trips a fresh
    /// pet — the bin's `load_or_create` then only ever has to decay.
    #[test]
    fn pet_state_serializes_round_trip() {
        let st = PetState::fresh(123);
        let bytes = serde_json::to_vec(&st).unwrap();
        let back: PetState = serde_json::from_slice(&bytes).unwrap();
        // Byte-for-byte identity through the round trip (field-level
        // privacy means the bin only ever owns a serialized contract).
        assert_eq!(
            serde_json::to_value(&back).unwrap(),
            serde_json::to_value(&st).unwrap()
        );
        assert_eq!(back.name, st.name);
    }
}
