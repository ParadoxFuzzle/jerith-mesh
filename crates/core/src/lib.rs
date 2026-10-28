//! jerith-mesh core: hardware probing, config, discovery, mesh runtime,
//! gateway and web surfaces.

pub mod backup;
pub mod bootstrap;
pub mod bootstrap_server;
pub mod config;
pub mod contrib;
pub mod coordinator;
pub mod discovery;
pub mod economics;
pub mod expert_stats;
pub mod expert_store;
pub mod freerider;
pub mod gateway;
pub mod gguf_import;
pub mod guard;
pub mod http;
pub mod hw;
pub mod isolate;
pub mod lore;
pub mod memory;
pub mod metrics;
pub mod model_router;
pub mod models;
pub mod ping;
pub mod pipeline;
pub mod pluginbundle;
pub mod quic;
pub mod ratelimit;
pub mod replication;
pub mod signing;
pub mod stake;
pub mod suggest;
pub mod toolstack;
pub mod verify;
pub mod web;
pub mod websearch;

pub use bootstrap::{BootstrapClient, PeerAd, PeerRegistry, PeerState};
pub use config::{MemoryBackend, NodeConfig, NodeRole, SetupChoices};
pub use models::{
    ModelFit, ModelId, ModelSource, ModelSpec, curated, detect_capable_models, format_bytes,
    recommended_model,
};
pub use ping::{MeshLoop, MeshState, Meters};
