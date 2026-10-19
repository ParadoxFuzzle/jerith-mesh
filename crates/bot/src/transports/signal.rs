//! Signal transport via the standard signal-cli REST daemon
//! (`signal-cli --http-api` style: JSON over plain HTTP, loopback by
//! default — the daemon owns all the crypto plumbing and account
//! state, this adapter owns none of it). Inbound: long-poll
//! `/v1/receive?number={account}&timeout=25` (returns batched
//! envelopes); outbound: `/v2/send` with per-message recipients.
//!
//! Self-echo discipline relies on the daemon: envelopes authored by our
//! own account appear without a dataMessage we route — but belt and
//! braces: `sourceNumber == account` is skipped explicitly. Group
//! transport: `respond_groups` lists base64 groupIds to serve; DMs are
//! always served (mention mode makes no sense in a 1:1).
//!
//! The daemon is the failure surface: if not running, the transport
//! fails fast on the first poll → supervisor respawn loop with its
//! exponential backoff — the same boot-time handshake as every poller.

use super::{chunk_utf8, handle_incoming, http_json, DeliveryFn, Incoming};
use crate::config::BotConfig;
use crate::state::BotState;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConfig {
    /// signal-cli REST daemon base url (loopback default of the stack).
    pub daemon_url: String,
    /// The OWN account (E.164, e.g. "+15551234567") — the daemon's linked
    /// number this bot speaks as.
    pub account: String,
    /// Base64 groupIds to serve; empty = DMs only.
    #[serde(default)]
    pub respond_groups: Vec<String>,
}
impl SignalConfig {
    pub fn validate(&mut self) -> Result<()> {
        let d = self.daemon_url.trim_end_matches('/').to_string();
        anyhow::ensure!(d.starts_with("http://127.0.0.1") || d.starts_with("http://localhost") || d.starts_with("https://"), "signal: daemon_url must be loopback or https");
        self.daemon_url = d;
        anyhow::ensure!(self.account.trim().starts_with('+'), "signal: account must be E.164 (+1555…)");
        Ok(())
    }
}

/// Pull ONE batch of receive envelopes (a long-poll: empty array on
/// nothing-new). Returns ([] batch) or a typed daemon failure.
async fn poll_once(scfg: &SignalConfig) -> Result<Vec<Value>> {
    let url = format!(
        "{}/v1/receive?number={}&timeout=25",
        scfg.daemon_url,
        urlencode(&scfg.account)
    );
    let (code, v) = http_json(reqwest::Method::GET, &url, None, None, Duration::from_secs(40)).await?;
    anyhow::ensure!(
        code >= 200 && code < 300,
        "signal daemon refused receive ({code}): {v}"
    );
    Ok(v.as_array().cloned().unwrap_or_default())
}

fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Extract a routable message from one receive-batch envelope
/// (unit-tested shape — the daemon nestings are the whole parse).
/// Returns Some((group_id_or_empty, source, name, text)).
pub fn parse_envelope(item: &Value, account: &str) -> Option<(String, String, String, String)> {
    let env = item.get("envelope")?;
    // Own echoes: skipped before anything else.
    if env["sourceNumber"].as_str() == Some(account) {
        return None;
    }
    let dm = env.get("dataMessage")?;
    if let Some(sync) = env.get("sentMessage") {
        // Messages SENT from own account from another device: never route.
        let _ = sync;
        return None;
    }
    let Some(text) = dm["message"].as_str().filter(|t| !t.trim().is_empty()).map(str::to_string) else {
        return None;
    };
    let source = env["sourceNumber"]
        .as_str()
        .or_else(|| env["source"].as_str())
        .unwrap_or_default()
        .to_string();
    if source.is_empty() {
        return None;
    }
    let group = dm
        .pointer("/groupInfo/groupId")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let name = env["sourceName"].as_str().unwrap_or(&source).to_string();
    Some((group, source, name, text))
}

pub async fn run(cfg: Arc<BotConfig>, state: BotState, mut scfg: SignalConfig, owner: Option<u64>) -> Result<()> {
    scfg.validate()?;
    tracing::info!(transport = "signal", account = %scfg.account, groups = scfg.respond_groups.len(), "starting");
    let deliverer: DeliveryFn = {
        let scfg2 = scfg.clone();
        Arc::new(move |channel: &str, text: String| {
            // channel IS the conversation id ("group:<groupid>" or a number).
            let scfg2 = scfg2.clone();
            let (recip, group) = channel
                .strip_prefix("group:")
                .map(|g| (String::new(), Some(g.to_string())))
                .unwrap_or((channel.to_string(), None));
            Box::pin(async move {
                for chunk in chunk_utf8(&text, 4000) {
                    let url = format!("{}/v2/send", scfg2.daemon_url);
                    let mut body = json!({ "account": scfg2.account, "message": chunk });
                    match group.as_ref() {
                        Some(g) => body["groupIds"] = json!([g]),
                        None => body["recipients"] = json!([recip]),
                    }
                    let (code, v) = http_json(reqwest::Method::POST, &url, None, Some(body), Duration::from_secs(30)).await?;
                    anyhow::ensure!(code >= 200 && code < 300, "signal send failed ({code}): {v}");
                }
                Ok(())
            })
        })
    };

    loop {
        let batch = poll_once(&scfg).await?;
        for item in &batch {
            let Some((group, source, name, text)) = parse_envelope(item, &scfg.account) else {
                continue;
            };
            // Group filter (DMs always route).
            if !group.is_empty() {
                if !scfg.respond_groups.iter().any(|amt| amt == &group) {
                    continue;
                }
            }
            let channel = if group.is_empty() {
                source.clone()
            } else {
                format!("group:{group}")
            };
            let inc = Incoming {
                platform: "signal",
                channel,
                user: source.clone(),
                display: name.clone(),
                text,
            };
            handle_incoming(&cfg, &state, inc, owner, Arc::clone(&deliverer)).await;
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_parse_routes_human_dms_and_groups_skips_own_and_sync() {
        let me = "+1555000111";
        // Human DM.
        let dm = json!({
            "envelope": {
                "sourceNumber": "+1555999888", "sourceName": "Noodly",
                "dataMessage": { "message": "hello mesh" }
            }
        });
        assert_eq!(
            parse_envelope(&dm, me),
            Some(("".into(), "+1555999888".into(), "Noodly".into(), "hello mesh".into()))
        );
        // Group message.
        let grp = json!({
            "envelope": {
                "sourceNumber": "+1555999888",
                "dataMessage": { "message": "greet", "groupInfo": { "groupId": "AbCdEf==" } }
            }
        });
        let (group, _, _, text) = parse_envelope(&grp, me).unwrap();
        assert_eq!((group.as_str(), text.as_str()), ("AbCdEf==", "greet"));
        // Own echo skipped.
        let own = json!({
            "envelope": {
                "sourceNumber": "+1555000111",
                "dataMessage": { "message": "self" }
            }
        });
        assert_eq!(parse_envelope(&own, me), None);
        // Typing receipt (no dataMessage) skipped.
        let receipt = json!({ "envelope": { "sourceNumber": "+1555999888", "typingMessage": {} } });
        assert_eq!(parse_envelope(&receipt, me), None);
        // Empty text skipped.
        let empty = json!({ "envelope": { "sourceNumber": "+1555999888", "dataMessage": { "message": "" } } });
        assert_eq!(parse_envelope(&empty, me), None);
    }

    #[test]
    fn config_daemon_loopback_https_rule() {
        let mut ok = SignalConfig {
            daemon_url: "http://127.0.0.1:8080".into(),
            account: "+1555000111".into(),
            respond_groups: vec![],
        };
        ok.validate().unwrap();
        assert_eq!(ok.daemon_url, "http://127.0.0.1:8080");
        let mut bad = SignalConfig { daemon_url: "http://0.0.0.0:8080".into(), ..ok.clone() };
        assert!(bad.validate().is_err(), "non-loopback plaintext daemon refused");
        let mut bad2 = SignalConfig { account: "me".into(), ..ok };
        assert!(bad2.validate().is_err());
    }
}
