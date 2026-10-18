//! Tenor GIF capability for the chat surfaces (Discord).
//!
//! Two halves, one module:
//!
//! - **Seeing GIFs:** a typed detection over the message's primitives —
//!   `image/gif` attachments, `.gif` filenames, Tenor/GIPHY share URLs in
//!   the text (tenor.com/view, media.tenor.com, giphy.com/gifs,
//!   media.giphy.com) and Tenor/GIPHY embeds (Discord unrolls a Tenor
//!   link into an embed whose media points at media.tenor.com). The
//!   event handler uses this to REACT (an emoji ack, picked
//!   deterministically per message) and — for a message that is
//!   genuinely ONLY a gif — to acknowledge it without feeding a bare URL
//!   to the chat pipeline as if it were a question.
//! - **Sending GIFs:** a Tenor search that resolves a query to real
//!   shareable gif URLs. The API key read follows the mesh's skip-loud
//!   env convention (`TENOR_API_KEY`): set → Google-hosted Tenor v2
//!   (`tenor.googleapis.com/v2/search`, gc-gif results, content filter
//!   high); unset → an honest typed error the command renders as a
//!   config hint, never a silent shrug.

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Detection (pure functions over primitives — testable without Discord)
// ---------------------------------------------------------------------------

/// Is this URL/share text a GIF source we recognize (Tenor, GIPHY, or a
/// raw `.gif` link)? Pure substring logic, deliberately conservative
/// (false negatives beat false reactions).
pub fn is_gif_url(s: &str) -> bool {
    let lower = s.trim().to_ascii_lowercase();
    let gif_ext = lower.split('?').next().is_some_and(|p| p.ends_with(".gif"));
    gif_ext
        // SHARE paths only — tenor.com/search and giphy.com/channel are
        // pages, not gifs (the live test caught the first draft).
        || lower.contains("tenor.com/view/")
        || lower.contains("tenor.com/watch/")
        || lower.contains("media.tenor.com/")
        || lower.contains("giphy.com/gifs/")
        || lower.contains("media.giphy.com/")
}

/// One attachment's gif-ness: content type and/or filename.
fn attachment_is_gif(content_type: Option<&str>, filename: &str) -> bool {
    content_type.is_some_and(|t| t.starts_with("image/gif"))
        || filename.to_ascii_lowercase().ends_with(".gif")
}

/// Does this message read as a GIF delivery? True for a `image/gif`
/// attachment, a `.gif` filename, a Tenor/GIPHY URL in the text, or a
/// Tenor/GIPHY embed (Discord unrolls share links; the embed points at
/// media.tenor.com). `embed_parts` should carry the fields the handler
/// can access cheaply (embed url, media/video url, description).
pub fn message_is_gif(
    content: &str,
    attachments: &[(Option<&str>, &str)],
    embed_parts: &[&str],
) -> bool {
    attachments
        .iter()
        .any(|(t, name)| attachment_is_gif(*t, name))
        || embed_parts.iter().any(|p| is_gif_url(p))
        || content.split_whitespace().any(is_gif_url)
}

/// The deterministic ack-pick: variety across messages, stability on
/// retries (the SAME message always earns the SAME reaction — a re-run
/// of the handler never double-reacts with different emoji).
pub const GIF_ACK_REACTIONS: &[&str] = &["👀", "🎬", "🔥", "😄", "🫡"];

pub fn pick_reaction(message_id: u64) -> &'static str {
    GIF_ACK_REACTIONS[(message_id % GIF_ACK_REACTIONS.len() as u64) as usize]
}

/// A message whose content is ONLY gif links (every remaining token is a
/// gif URL) — such a message should not become a chat prompt.
pub fn content_is_gif_only(content: &str, gif_urls: &[&str]) -> bool {
    let mut rest = content.trim().to_string();
    for url in gif_urls {
        rest = rest.replacen(url, "", 1);
    }
    rest.trim().is_empty()
}

// ---------------------------------------------------------------------------
// Tenor search (env-gated, skip-loud)
// ---------------------------------------------------------------------------

/// One Tenor result: the share URL (Discord unrolls it) + the direct gif
/// URL (for surfaces that do not unroll).
#[derive(Debug, Clone, PartialEq)]
pub struct TenorGif {
    pub description: String,
    /// The canonical share page (tenor.com/view/...), markdown-linkable.
    pub share_url: String,
    /// The direct gif bytes URL (media.tenor.com/...gif), when present.
    pub gif_url: String,
}

/// The api key: `TENOR_API_KEY` (raw env), mirroring how the miner reads
/// `JERITH_TOKEN`. `None` = the honest skip (the caller renders the hint).
pub fn tenor_key() -> Option<String> {
    let t = std::env::var("TENOR_API_KEY").ok()?;
    let t = t.trim().to_string();
    (!t.is_empty()).then_some(t)
}

/// The search URL (Google-hosted Tenor v2): up to 10 gc-gif results,
/// high content filter, our `client_key` for Tenor's partner logging.
pub fn tenor_search_url(api_key: &str, query: &str) -> String {
    format!(
        "https://tenor.googleapis.com/v2/search?q={}&key={}&limit=10&media_filter=gif,tinygif&contentfilter=high&client_key=jerith-mesh-bot",
        urlencode(query),
        urlencode(api_key)
    )
}

/// Percent/URL-encode (same shape as economics::urlencode — symbols and
/// spaces must survive the query).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

#[derive(Deserialize)]
struct TenorResponse {
    results: Vec<TenorResult>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
struct TenorResult {
    content_description: Option<String>,
    itemurl: String,
    #[serde(default)]
    media_formats: std::collections::HashMap<String, TenorMedia>,
}

#[derive(Deserialize)]
struct TenorMedia {
    url: String,
}

impl TenorGif {
    /// How the bot posts it: the share link markdown-named by the
    /// description (Discord unrolls the bare URL anyway; the markdown
    /// keeps group channels tidy).
    pub fn render(&self) -> String {
        let name = if self.description.is_empty() {
            "a GIF"
        } else {
            self.description.as_str()
        };
        format!("[{name}]({})", self.share_url)
    }
}

/// Parse a Tenor v2 search body (pure — every live surface funnels
/// through here, errors included).
pub fn parse_tenor_search(body: &str) -> anyhow::Result<Vec<TenorGif>> {
    let parsed: TenorResponse =
        serde_json::from_str(body).map_err(|e| anyhow::anyhow!("tenor body not a search: {e}"))?;
    Ok(parsed
        .results
        .into_iter()
        .map(|r| TenorGif {
            description: r.content_description.unwrap_or_default(),
            share_url: r.itemurl,
            gif_url: r
                .media_formats
                .get("gif")
                .map(|m| m.url.clone())
                .unwrap_or_default(),
        })
        .collect())
}

/// The live search. Errors carry the config hint when the key is unset —
/// the skip-loud contract (never a silent shrug, never a fake result).
pub async fn tenor_search(query: &str) -> anyhow::Result<Vec<TenorGif>> {
    let Some(key) = tenor_key() else {
        anyhow::bail!(
            "tenor search needs an api key — set TENOR_API_KEY (Google Cloud, Tenor API enabled); \
             reactions and gif detection work without it"
        );
    };
    let url = tenor_search_url(&key, query);
    let resp = jerith_mesh_core::http::mesh_http()
        .get(&url)
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("tenor unreachable: {e}"))?
        .error_for_status()
        .map_err(|e| anyhow::anyhow!("tenor refused the search ({e}) — is the key valid?"))?;
    let body = resp
        .text()
        .await
        .map_err(|e| anyhow::anyhow!("tenor body: {e}"))?;
    parse_tenor_search(&body)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gif_urls_are_recognized_conservatively() {
        for ok in [
            "https://tenor.com/view/cat-fall-gif-12345",
            "https://media.tenor.com/abcd1234/whatever.gif",
            "https://giphy.com/gifs/party-hard-abc",
            "https://media.giphy.com/media/l0ExdMUDPjXZNn8Z2/giphy.gif",
            "https://example.com/stuff/loop.gif",
            "https://example.com/stuff/loop.gif?w=200",
        ] {
            assert!(is_gif_url(ok), "{ok} is a gif");
        }
        for no in [
            "https://tenor.com/search/hello", // a search page, not a gif
            "https://example.com/cat.gif.png",
            "https://example.com/photos",
            "https://giphy.com/channel/jerith",
            "plain text with nothing",
        ] {
            assert!(!is_gif_url(no), "{no} is not a gif");
        }
    }

    #[test]
    fn message_gif_detection_covers_every_delivery_shape() {
        // Direct attachment (content type and filename variants).
        assert!(message_is_gif("", &[(Some("image/gif"), "x")], &[]));
        assert!(message_is_gif(
            "",
            &[(Some("image/gif"), "weird-name")],
            &[]
        ));
        assert!(message_is_gif("", &[(None, "clip.GIF")], &[]));
        // Tenor link in the text.
        assert!(message_is_gif(
            "look: https://tenor.com/view/cat-gif-1",
            &[],
            &[]
        ));
        // Embed (Discord unrolls the link server-side).
        assert!(message_is_gif("", &[], &["https://media.tenor.com/z/gif"]));
        // Negatives.
        assert!(!message_is_gif("what is a gif", &[], &[]));
        assert!(!message_is_gif("", &[(Some("image/png"), "shot.png")], &[]));
        assert!(!message_is_gif("", &[], &["https://example.com/cat.png"]));
    }

    #[test]
    fn ack_reaction_is_deterministic_per_message() {
        let a = pick_reaction(11);
        assert_eq!(pick_reaction(11), a, "retry earns the SAME emoji");
        assert_eq!(a, GIF_ACK_REACTIONS[(11 % 5) as usize]);
        // And the pool actually varies across a handful of ids.
        let picks: std::collections::HashSet<&str> = (0..40).map(pick_reaction).collect();
        assert!(picks.len() >= 3, "the ack should have variety: {picks:?}");
    }

    #[test]
    fn gif_only_content_is_not_a_prompt() {
        let url = "https://tenor.com/view/cat-gif-1";
        assert!(content_is_gif_only(url, &[url]));
        assert!(!content_is_gif_only(
            "look at this https://tenor.com/view/cat-gif-1",
            &[url]
        ));
    }

    #[test]
    fn tenor_search_url_escapes_and_pins_the_shape() {
        let url = tenor_search_url("K3Y-Q-9_~", "cat & dog jump!");
        assert!(
            url.contains("q=cat%20%26%20dog%20jump%21")
                && url.contains("key=K3Y-Q-9_~")
                && url.contains("media_filter=gif,tinygif")
                && url.contains("contentfilter=high")
                && url.contains("client_key=jerith-mesh-bot")
                && url.contains("limit=10"),
            "{url}"
        );
    }

    #[test]
    fn tenor_body_parses_into_shareable_gifs() {
        // A representative Tenor v2 search body (trimmed, shape-faithful).
        let body = r#"{
            "results": [
                {
                    "content_description": "A cat falling in slow motion",
                    "itemurl": "https://tenor.com/view/cat-fall-gif-12345",
                    "media_formats": {
                        "gif": {"url": "https://media.tenor.com/aaa/cat.gif"},
                        "tinygif": {"url": "https://media.tenor.com/aaa/cat-tiny.gif"}
                    }
                },
                {
                    "itemurl": "https://tenor.com/view/dog-woof-gif-9",
                    "media_formats": {"gif": {"url": "https://media.tenor.com/bbb/dog.gif"}}
                }
            ]
        }"#;
        let gifs = parse_tenor_search(body).expect("parses");
        assert_eq!(gifs.len(), 2);
        assert_eq!(gifs[0].description, "A cat falling in slow motion");
        assert_eq!(
            gifs[0].share_url,
            "https://tenor.com/view/cat-fall-gif-12345"
        );
        assert_eq!(gifs[0].gif_url, "https://media.tenor.com/aaa/cat.gif");
        // Missing description renders as "a GIF", never as an empty label.
        assert_eq!(gifs[1].description, "");
        let rendered = gifs[1].render();
        assert!(rendered.starts_with("[a GIF]("), "{rendered}");
        // A garbage body is an honest error, not a panic.
        assert!(parse_tenor_search("{\"nope\":true}").is_err());
    }

    #[test]
    fn missing_key_is_a_typed_skip() {
        // Read-only env-mirror check (no env mutation — remove_var is
        // unsafe in this edition and global state anyway): the gate is
        // exactly the trimmed-env contract.
        match std::env::var("TENOR_API_KEY") {
            Ok(v) if !v.trim().is_empty() => assert!(tenor_key().is_some()),
            _ => assert!(tenor_key().is_none()),
        }
    }
}
