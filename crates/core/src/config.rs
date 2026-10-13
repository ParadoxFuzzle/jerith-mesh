//! Node configuration, identity and the setup-flow state machine.
//!
//! Config lives at `~/.jerith-mesh/config.toml` by default (overridable via
//! `JERITH_HOME`). The node identity is an Ed25519-style random keypair
//! placeholder for Phase 0 — key's serialized as hex; it is regenerated only
//! if missing so a node keeps the same `node_id` across restarts.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

mod hex_opt_bytes {
    pub fn serialize<S: serde::Serializer>(v: &Option<Vec<u8>>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            None => s.serialize_none(),
            Some(bytes) => s.serialize_str(&hex::encode(bytes)),
        }
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Vec<u8>>, D::Error> {
        let s: Option<String> = serde ::Deserialize::deserialize(d)?;
        match s {
            None => Ok(None),
            Some(hexstr) => hex::decode(hexstr)
                .map(Some)
                .map_err(serde::de::Error::custom),
        }
    }
}

use crate::models::{ModelId, ModelSource};

pub const MIN_CONTRIBUTION_PERCENT: u64 = 25;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryBackend {
    Sqlite,
    PostgresPgvector,
}

impl MemoryBackend {
    pub fn label(self) -> &'static str {
        match self {
            MemoryBackend::Sqlite => "SQLite (local file)",
            MemoryBackend::PostgresPgvector => "PostgreSQL + pgvector",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NodeRole {
    Contributor,
    Requester,
    Hybrid,
}

/// Which chain the node settles earnings in.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CryptoChain {
    /// Default: JerithChain (JER), integrated natively.
    Jerithchain,
    /// User-managed: the node only exports metering data; the user's own
    /// miner/wallet handles BTC/ETH/DOGE/XMR settlement.
    Bitcoin,
    Ethereum,
    Dogecoin,
    Monero,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoConfig {
    pub chain: CryptoChain,
    /// Wallet/endpoint details ( JerithChain node RPC for JER; user-supplied
    /// endpoint URIs for the other chains).
    #[serde(default)]
    pub wallet_endpoint: Option<String>,
    #[serde(default)]
    pub wallet_address: Option<String>,
    /// Chance (per-mille, 0–1000) to attempt a JER mine on each served
    /// inference. 120 ≈ ~12 % like the JerithChain Discord agent.
    #[serde(default = "default_mine_chance_permille")]
    pub mine_chance_permille: u32,
}

fn default_mine_chance_permille() -> u32 {
    120
}

impl Default for CryptoConfig {
    fn default() -> Self {
        Self {
            chain: CryptoChain::Jerithchain,
            wallet_endpoint: None,
            wallet_address: None,
            mine_chance_permille: default_mine_chance_permille(),
        }
    }
}

/// Local specialist models — always private, never part of the 25 % quota.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialistConfig {
    #[serde(default = "default_true")]
    pub vision_enabled: bool,
    #[serde(default = "default_true")]
    pub embeddings_enabled: bool,
    /// Matryoshka truncation for nomic-embed-text-v2-moe.
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: u32,
    /// Local Ollama base URL (serves the nomic embedding model).
    #[serde(default = "default_ollama_url")]
    pub ollama_url: String,
}

fn default_ollama_url() -> String {
    "http://127.0.0.1:11434".to_string()
}

fn default_true() -> bool {
    true
}
fn default_embedding_dim() -> u32 {
    256
}

impl Default for SpecialistConfig {
    fn default() -> Self {
        Self {
            vision_enabled: true,
            embeddings_enabled: true,
            embedding_dim: default_embedding_dim(),
            ollama_url: default_ollama_url(),
        }
    }
}

/// Full node configuration (what `setup` writes and `run` reads).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub node_id: String,
    /// Hex-encoded secret key for signing pings and results (Phase 0
    /// placeholder — random 32 bytes).
    #[serde(with = "hex_opt_bytes", default)]
    pub secret_key_hex: Option<Vec<u8>>,
    pub model: Option<ModelId>,
    pub role: NodeRole,
    pub contribution_percent: u64,
    pub memory_backend: MemoryBackend,
    pub bootstrap_url: String,
    pub crypto: CryptoConfig,
    pub specialists: SpecialistConfig,
    /// Port for the local OpenAI-compatible gateway + web UI.
    pub gateway_port: u16,
    /// UDP port for the QUIC NodeRPC listener (defaults to gateway_port+1).
    pub rpc_port: Option<u16>,
    /// Colibri / Ollama base URL to forward chat completions to.
    pub runtime_url: String,
    /// Phase 4: probability (percent) of redundantly replaying a remote
    /// expert result on a second node before trusting it. Spec range:
    /// 5–15 %; 0 disables replay (commitments always checked).
    #[serde(default = "default_verify_percent")]
    pub verify_percent: u32,
}

fn default_verify_percent() -> u32 {
    10
}



impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            node_id: String::new(),
            secret_key_hex: None,
            model: None,
            role: NodeRole::Hybrid,
            contribution_percent: MIN_CONTRIBUTION_PERCENT,
            memory_backend: MemoryBackend::Sqlite,
            bootstrap_url: "https://p2p.jerithai.com".to_string(),
            crypto: CryptoConfig::default(),
            specialists: SpecialistConfig::default(),
            gateway_port: 8900,
            rpc_port: None,
            runtime_url: "http://127.0.0.1:8000".to_string(),
            verify_percent: default_verify_percent(),
        }
    }
}

impl NodeConfig {
    /// Effective NodeRPC UDP port. Defaults to `GATEWAY+100` — deliberately
    /// far from `gateway_port+1`, which collides with the bootstrap node's
    /// TCP 8901 whenever gateway = 8900 (seen live during E2E).
    pub fn effective_rpc_port(&self) -> u16 {
        self.rpc_port.unwrap_or(self.gateway_port + 100)
    }
}

impl NodeConfig {
    /// Path helpers — everything lives under the node's home dir.
    pub fn home() -> PathBuf {
        if let Ok(dir) = std::env::var("JERITH_HOME") {
            return PathBuf::from(dir);
        }
        if let Ok(home) = std::env::var("HOME") {
            return Path::new(&home).join(".jerith-mesh");
        }
        PathBuf::from(".jerith-mesh")
    }

    pub fn config_path() -> PathBuf {
        Self::home().join("config.toml")
    }
    pub fn data_dir() -> PathBuf {
        Self::home().join("data")
    }
    pub fn memory_db_path() -> PathBuf {
        Self::data_dir().join("memory.sqlite")
    }

    /// Load config, tolerating a missing file (caller decides what to do).
    pub fn load() -> anyhow::Result<Option<Self>> {
        let path = Self::config_path();
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let cfg: Self =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        Ok(Some(cfg))
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let text = toml::to_string_pretty(self).context("serializing config")?;
        std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    /// Generate (or restore) the Ed25519 identity — idempotent.
    /// The `node_id` is the pubkey fingerprint; it is (re)derived from the
    /// secret so a node always self-consistently signs as its own node_id.
    pub fn ensure_identity(&mut self) -> anyhow::Result<()> {
        let identity = match &self.secret_key_hex {
            Some(secret) => crate::signing::SigningIdentity::from_secret_bytes(secret)?,
            None => {
                let id = crate::signing::SigningIdentity::generate();
                self.secret_key_hex = Some(id.secret_bytes().to_vec());
                id
            }
        };
        self.node_id = identity.node_id();
        Ok(())
    }

    /// Load the signing identity for wire auth (errors if unset).
    pub fn identity(&self) -> anyhow::Result<crate::signing::SigningIdentity> {
        let secret = self
            .secret_key_hex
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no identity — run setup first"))?;
        crate::signing::SigningIdentity::from_secret_bytes(secret)
    }

    /// Set the model and (per spec) this implies disconnecting from any peers
    /// of the previous model. The mesh layer performs the actual disconnect.
    pub fn set_model(&mut self, model: ModelId) {
        self.model = Some(model);
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.node_id.is_empty() {
            anyhow::bail!("node_id is empty — run setup first");
        }
        if self.contribution_percent < MIN_CONTRIBUTION_PERCENT {
            anyhow::bail!(
                "contribution_percent {} is below the minimum {}",
                self.contribution_percent,
                MIN_CONTRIBUTION_PERCENT
            );
        }
        Ok(())
    }
}

/// Finalize a setup flow: given the collected answers produce a validated,
/// persisted config with a fresh identity.
pub struct SetupChoices {
    pub model: ModelId,
    pub memory_backend: MemoryBackend,
    pub contribution_percent: u64,
    pub role: NodeRole,
    pub crypto: CryptoConfig,
    pub specialists: SpecialistConfig,
    pub bootstrap_url: Option<String>,
    pub gateway_port: u16,
    pub runtime_url: Option<String>,
}

impl SetupChoices {
    pub fn into_config(self, existing: Option<NodeConfig>) -> anyhow::Result<NodeConfig> {
        let mut cfg = existing.unwrap_or_default();
        cfg.ensure_identity()?;
        cfg.set_model(self.model);
        cfg.memory_backend = self.memory_backend;
        cfg.contribution_percent = self.contribution_percent;
        cfg.role = self.role;
        cfg.crypto = self.crypto;
        cfg.specialists = self.specialists;
        if let Some(url) = self.bootstrap_url {
            cfg.bootstrap_url = url;
        }
        cfg.gateway_port = self.gateway_port;
        if let Some(url) = self.runtime_url {
            cfg.runtime_url = url;
        }
        cfg.validate()?;
        Ok(cfg)
    }
}

/// Default ModelId for easy setup defaults.
pub fn default_model(model_id: impl Into<String>) -> ModelId {
    ModelId {
        id: model_id.into(),
        quant: "int4".to_string(),
        source: ModelSource::Curated,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ModelSource;

    fn sample_model() -> ModelId {
        ModelId {
            id: "glm-5.3".into(),
            quant: "int4".into(),
            source: ModelSource::Curated,
        }
    }

    fn sample_choices() -> SetupChoices {
        SetupChoices {
            model: sample_model(),
            memory_backend: MemoryBackend::Sqlite,
            contribution_percent: 40,
            role: NodeRole::Hybrid,
            crypto: CryptoConfig::default(),
            specialists: SpecialistConfig::default(),
            bootstrap_url: Some("https://p2p.example.org".into()),
            gateway_port: 8901,
            runtime_url: Some("http://127.0.0.1:8000".into()),
        }
    }

    #[test]
    fn default_config_passes_min_contribution() {
        assert_eq!(
            NodeConfig::default().contribution_percent,
            MIN_CONTRIBUTION_PERCENT
        );
    }

    #[test]
    fn identity_is_stable_across_ensure_calls() {
        let mut cfg = NodeConfig::default();
        cfg.ensure_identity().unwrap();
        let id1 = cfg.node_id.clone();
        let key1 = cfg.secret_key_hex.clone();
        cfg.ensure_identity().unwrap();
        assert_eq!(id1, cfg.node_id);
        assert_eq!(key1, cfg.secret_key_hex);
        assert_eq!(key1.unwrap().len(), 32);
    }

    #[test]
    fn node_id_is_pubkey_fingerprint() {
        let mut cfg = NodeConfig::default();
        cfg.ensure_identity().unwrap();
        let identity = cfg.identity().unwrap();
        assert_eq!(cfg.node_id, identity.node_id());
        assert_eq!(cfg.node_id.len(), 16);
    }

    #[test]
    fn identity_restores_from_saved_secret() {
        let mut cfg = NodeConfig::default();
        cfg.ensure_identity().unwrap();
        let text = toml::to_string_pretty(&cfg).unwrap();
        let back: NodeConfig = toml::from_str(&text).unwrap();
        assert_eq!(back.identity().unwrap().pubkey_hex(), cfg.identity().unwrap().pubkey_hex());
    }

    #[test]
    fn setup_choices_produce_valid_config() {
        let cfg = sample_choices().into_config(None).unwrap();
        assert_eq!(cfg.model.as_ref().unwrap().id, "glm-5.3");
        assert_eq!(cfg.contribution_percent, 40);
        assert_eq!(cfg.bootstrap_url, "https://p2p.example.org");
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_rejects_below_minimum_contribution() {
        let choices = SetupChoices {
            contribution_percent: 10,
            ..sample_choices()
        };
        assert!(choices.into_config(None).is_err());
        let mut cfg = NodeConfig::default();
        cfg.contribution_percent = 10;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_round_trips_through_toml() {
        let mut cfg = sample_choices().into_config(None).unwrap();
        cfg.gateway_port = 9123;
        let text = toml::to_string_pretty(&cfg).unwrap();
        let back: NodeConfig = toml::from_str(&text).unwrap();
        assert_eq!(back.gateway_port, 9123);
        assert_eq!(back.node_id, cfg.node_id);
        assert!(back.secret_key_hex.is_some());
    }

    #[test]
    fn set_model_replaces_previous() {
        let mut cfg = sample_choices().into_config(None).unwrap();
        cfg.set_model(ModelId {
            id: "qwen3.8-flash-next".into(),
            quant: "int4".into(),
            source: ModelSource::Curated,
        });
        assert_eq!(cfg.model.unwrap().id, "qwen3.8-flash-next");
    }

    #[test]
    fn rpc_port_defaults_clear_of_bootstrap_tcp() {
        let mut cfg = NodeConfig::default();
        // Default gateway 8900 → RPC lands on 9000, never on the public
        // bootstrap's TCP 8901.
        assert_eq!(cfg.effective_rpc_port(), 9000);
        cfg.rpc_port = Some(9201);
        assert_eq!(cfg.effective_rpc_port(), 9201);
    }

    #[test]
    fn default_specialists_enabled() {
        let s = SpecialistConfig::default();
        assert!(s.vision_enabled);
        assert!(s.embeddings_enabled);
        assert_eq!(s.embedding_dim, 256);
    }

    #[test]
    fn crypto_defaults_to_jerithchain() {
        let c = CryptoConfig::default();
        assert_eq!(c.chain, CryptoChain::Jerithchain);
        assert_eq!(c.mine_chance_permille, 120);
    }
}
