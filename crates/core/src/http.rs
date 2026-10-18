//! Central, TLS-pinned HTTP client construction for the whole mesh.
//!
//! Why this exists: a dependency evaluation (the serenity
//! `native_tls_backend` stopgap, branch `stopgap-native-tls`) measured
//! that one stray feature-unification — any dep enabling
//! `reqwest/native-tls` — flips reqwest's *default* backend from rustls
//! to OpenSSL, and unpinned `Client::new()` sites switch silently with
//! it. The mesh's posture is rustls everywhere (see the workspace
//! manifest: `reqwest = { default-features = false, features =
//! ["rustls-tls", ...] }`); this module makes that posture local and
//! enforced instead of implicit on 33 call sites.
//!
//! Rule of the codebase: **no bare `reqwest::Client::new()` / unpinned
//! `Client::builder()` anywhere.** Build mesh clients through
//! [`mesh_http`] (or pin `.use_rustls_tls()` explicitly when a builder
//! chain needs custom tuning). The bootstrap-server's own server-side
//! code is unaffected — this is about the mesh's *outgoing* clients.

/// The mesh's outgoing HTTP client: rustls TLS, pinned against feature
/// unification. Construction cost matches `Client::new()` (a small
/// connection pool per call), so reuse the returned client where the
/// surrounding code already did.
///
/// Panics like `Client::new()` does if TLS client setup fails — the
/// previous behavior of every site this replaces.
pub fn mesh_http() -> reqwest::Client {
    reqwest::Client::builder()
        .use_rustls_tls()
        .build()
        .expect("mesh HTTP client with the pinned rustls TLS backend")
}

/// The shared PROBE client: process-wide, lazily built once, rustls
/// TLS plus a 3 s timeout (the short-budget contract of the /status
/// side probes — chain, bootstrap health). A fresh client per probe
/// paid a TLS handshake every /status poll; this one pools the
/// connections those hot-path probes hammer every few seconds.
///
/// Panics like `mesh_http` does if TLS client setup fails — the same
/// previous behavior of every site this replaces.
pub fn probe_http() -> &'static reqwest::Client {
    static PROBE_CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    PROBE_CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .use_rustls_tls()
            .timeout(std::time::Duration::from_secs(3))
            .build()
            .expect("probe HTTP client with the pinned rustls TLS backend")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_client_is_pooled() {
        let a: *const reqwest::Client = probe_http();
        let b: *const reqwest::Client = probe_http();
        assert!(std::ptr::eq(a, b), "probe client must be ONE pooled instance");
        // Offline request build proves the client is usable without any
        // network (the 3 s timeout is enforced per request, at send).
        let get = probe_http().get("https://probe.invalid/").build();
        assert!(get.is_ok(), "probe client builds requests");
    }

    #[test]
    fn pinned_client_builds() {
        // Construction is the pinned part (the backend choice compiles
        // in); building an offline request proves wiring works without
        // touching any network.
        let client = mesh_http();
        let get = client.get("https://example.invalid").build();
        assert!(get.is_ok(), "client builds requests fine");
    }
}
