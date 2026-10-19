//! GG transport — TYPED DORMANT, by research (2026-10-18). Searched the
//! platform's public documentation and current web references: **GG
//! (GG Bot / gg platform "GG Bot" in the game-chat sense) publishes no
//! public bot API** — no developer portal, no OAuth app registry, no
//! official docs of any inbound or outbound surface. A third-party
//! reverse-engineered client violates the platform trust model and
//! would break without notice.
//!
//! The honest non-implementation is this typed stub: config parses,
//! validate() always refuses with the above explanation (fail-loud at
//! LOAD — the operator sees why nothing runs), run() is unreachable in
//! practice (spawn_all never reaches it after the load refusal). If the
//! platform ships a public API, fill the stub in this one file — the
//! adapter contract (Incoming + DeliveryFn + handle_incoming) is the
//! same as every sibling here. `verify_incoming` documents the contract
//! so a future author knows exactly what to implement, and stays
//! compile-checked.
//!
//! The unit test locks the refusal in: a silent regression to a fake
//! runner would be caught.

use super::{Incoming, DeliveryFn};
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GgConfig {
    /// Free-form operator memo (never used to connect).
    #[serde(default)]
    pub note: Option<String>,
}impl GgConfig {
    /// Always fails: no public API exists to connect to (see module doc).
    /// (Referenced by the unit test below and any future integration
    /// point; the allow documents the dormant-by-research status.)
    #[allow(dead_code)]
    pub fn validate(&self) -> Result<()> {
        anyhow::bail!(
            "gg: no public bot API exists (checked the platform docs + web, 2026-10); \
             reverse-engineered clients are out of scope for this build. If GG ships \
             one, implement it in transports/gg.rs against the shared adapter contract."
        )
    }
}

/// The contract a future implementation must satisfy (compile-checked):
/// route an [`Incoming`] through `handle_incoming` with a platform
/// deliverer; never re-implement pipeline steps transport-side.
#[allow(dead_code, unreachable_code)]
fn _future_contract(_i: Incoming, _d: DeliveryFn) {
    // Example of the someday shape (never called):
    // let key = user_key("gg", &i.user);
    // super::handle_incoming(&cfg, &state, i, None, d).await;
}

pub async fn run(_cfg: std::sync::Arc<crate::config::BotConfig>, _state: crate::state::BotState, _gcfg: GgConfig, _owner: Option<u64>) -> Result<()> {
    // Unreachable in practice: validate() refuses at load. Typed anyway.
    unreachable!("gg has no public API — validate() refuses before this runs")
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The refusal is the FEATURE: locked so it never alternates.
    #[test]
    fn validate_refuses_loud_with_reason() {
        let err = GgConfig::default().validate().unwrap_err().to_string();
        assert!(err.contains("no public bot API"), "{err}");
        assert!(err.contains("transports/gg.rs"), "{err}");
    }
}
