//! jerith-mesh core: hardware probing, config, discovery, mesh runtime,
//! gateway and web surfaces.

pub mod bootstrap;
pub mod bootstrap_server;
pub mod config;
pub mod economics;
pub mod gateway;
pub mod guard;
pub mod hw;
pub mod memory;
pub mod models;
pub mod ping;
pub mod quic;
pub mod signing;
pub mod verify;
pub mod web;

pub use bootstrap::{BootstrapClient, PeerAd, PeerRegistry, PeerState};
pub use config::{MemoryBackend, NodeConfig, NodeRole, SetupChoices};
pub use models::{
    curated, detect_capable_models, format_bytes, recommended_model, ModelFit, ModelId,
    ModelSpec, ModelSource,
};
pub use ping::{MeshLoop, MeshState, Meters};
