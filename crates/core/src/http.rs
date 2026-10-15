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

#[cfg(test)]
mod tests {
    use super::*;

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
