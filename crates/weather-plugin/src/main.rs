//! `jerith-weather` — the weather skill's runnable half, over the
//! plugin SDK (`crates/plugin-template`, docs/PLUGIN-SDK.md). The
//! upstream/parsing half lives in `lib.rs` (pure, request-free, tested);
//! this binary is the wire glue:
//!
//! - reads its OWN `plugin.toml` for `[plugin] endpoint` + `secret`
//!   and binds that exact address — manifest and service cannot drift
//!   (the same one-source-of-truth rule as the pet);
//! - registers the two model-visible skills the manifest declares
//!   (`weather.now` = current conditions; `weather.forecast` = 1–7
//!   day series), with honest typed refusals the chat model reads and
//!   relays (unknown place names included — geocoding fails SHOULD
//!   reach the user spelled out, never as a dead pause);
//! - upstream is open-meteo.com's free geocoding + forecast APIs (no
//!   api key, no install-side rate meter; the manifest's
//!   `timeout_secs` bounds each chat's end anyway).
//!
//! Authoring note: third Phase-3d consumer after pet/house — copy this
//! layout, swap the handlers, sync `plugin.toml`, ship.

use anyhow::{Context as _, Result};
use jerith_plugin_template::{SkillServer, endpoint_addr, skill};
use jerith_weather_plugin::{
    Day, clamp_days, forecast_result, forecast_url, geocode_url, now_result, parse_forecast,
    parse_geocode,
};
use serde_json::Value;
use std::sync::OnceLock;

/// Upstream HTTP client: ONE pool per process, rustls pin untouched,
/// bounded total fetch (a page never exceeds ~4 KiB of JSON; 8 s cap
/// covers cold DNS + TLS on the desktop box).
fn http() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(8))
            .build()
            .expect("reqwest client builds")
    })
}

/// One upstream GET → (status, body). Non-200 fails typed with the
/// status so the model can say "upstream is briefly down".
async fn get_body(url: &str) -> Result<String, String> {
    let resp = http()
        .get(url)
        .header("User-Agent", "jerith-mesh weather plugin")
        .send()
        .await
        .map_err(|e| format!("open-meteo unreachable: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!(
            "open-meteo answered {} — brief upstream outage, try again",
            resp.status()
        ));
    }
    // Bounded read: the cap matches external::MAX_RESPONSE_BYTES's
    // spirit — a page is a few KiB, a hijacked upstream can't stream.
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("open-meteo body: {e}"))?;
    if bytes.len() > 65536 {
        return Err("open-meteo body exceeds 64 KB — refusing".to_string());
    }
    String::from_utf8(bytes.to_vec()).map_err(|e| format!("open-meteo body not utf-8: {e}"))
}

/// `[plugin] endpoint` + `[plugin] secret` (None = unsigned dev mode)
/// straight out of the manifest — one file rules both halves.
fn read_manifest(path: &str) -> Result<(std::net::SocketAddr, Option<String>)> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("reading manifest {path}"))?;
    // Document parse (toml::Table), NOT `str::parse::<toml::Value>` —
    // a Value FromStr only accepts a single value, so a whole manifest
    // string comes back «unexpected content, expected nothing» (the
    // pet's 2026-ergonomics lesson, reused verbatim).
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

/// Format the speed title for a real result Value — one pass over the
/// parsed body keeps handler dispatch for the json answer only.
fn args_place(args: &Value) -> Result<String, String> {
    let place = args["place"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "place is required (a city or place name)".to_string())?;
    Ok(place.to_string())
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
    tracing::info!(manifest, %addr, secret = secret.is_some(), "weather plugin starting");

    SkillServer::new("weather", secret)
        .handler(
            "weather.now",
            skill(move |args: Value| async move {
                let place = args_place(&args)?;
                let geo = get_body(&geocode_url(&place))
                    .await
                    .map_err(|e| format!("{place}: {e}"))?;
                let p = parse_geocode(&geo)?;
                let body = get_body(&forecast_url(p.latitude, p.longitude, 1)).await?;
                let fc = parse_forecast(&body)?;
                let tz = fc.timezone.clone().unwrap_or_else(|| "local".into());
                Ok(now_result(&p, &tz, &fc.current))
            }),
        )
        .handler(
            "weather.forecast",
            skill(move |args: Value| async move {
                let place = args_place(&args)?;
                let days = args["days"]
                    .as_u64()
                    .map(|v| clamp_days(v as u32))
                    .unwrap_or(3);
                let geo = get_body(&geocode_url(&place))
                    .await
                    .map_err(|e| format!("{place}: {e}"))?;
                let p = parse_geocode(&geo)?;
                let body = get_body(&forecast_url(p.latitude, p.longitude, days)).await?;
                let fc = parse_forecast(&body)?;
                let tz = fc.timezone.clone().unwrap_or_else(|| "local".into());
                let daily: Vec<Day> = fc
                    .daily
                    .time
                    .iter()
                    .enumerate()
                    .map(|(i, t)| Day {
                        date: t.clone(),
                        weather_code: fc.daily.weather_code.get(i).copied().flatten(),
                        temperature_2m_max: fc.daily.temperature_2m_max.get(i).copied().flatten(),
                        temperature_2m_min: fc.daily.temperature_2m_min.get(i).copied().flatten(),
                        precipitation_probability_max: fc
                            .daily
                            .precipitation_probability_max
                            .get(i)
                            .copied()
                            .flatten(),
                    })
                    .collect();
                Ok(forecast_result(&p, &tz, &daily))
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
    /// on the NEXT free port in the plugin lane (quarantine took 7791,
    /// ledger 7792 → weather is 7793), not the example's, secret
    /// signed, and the declared skill rows must be exactly the two
    /// registered above.
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
        assert_eq!(
            plugin.get("id").and_then(toml::Value::as_str),
            Some("weather")
        );
        let endpoint = plugin
            .get("endpoint")
            .and_then(toml::Value::as_str)
            .expect("endpoint line");
        let addr = endpoint_addr(endpoint).expect("endpoint binds");
        assert!(addr.ip().is_loopback(), "loopback by contract");
        assert_ne!(addr.port(), 7788, "the example's port belongs to the example");
        assert_eq!(addr.port(), 7793, "the weather lane's port");
        assert!(
            plugin.get("secret").and_then(toml::Value::as_str).is_some(),
            "the weather ships signed by default"
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
            vec!["forecast", "now"],
            "manifest skill rows = the two handlers"
        );
    }

    /// args_place contract: trimmed non-empty place, missing/blank args
    /// refuse typed (the model reads the string verbatim).
    #[test]
    fn args_place_refuses_empty_typed() {
        let err = args_place(&serde_json::json!({})).unwrap_err();
        assert!(err.contains("place is required"), "{err}");
        assert_eq!(args_place(&serde_json::json!({"place": " Prague "})).unwrap(), "Prague");
    }
}
