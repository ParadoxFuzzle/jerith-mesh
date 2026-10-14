//! `/mail` support: read the mailhook archive (raw `.eml` files the tunnel
//! webhook wrote per message) and format a compact recent-messages listing.
//!
//! Data source is the local archive (not IMAP) so listing works even while
//! the mail server is restarting; arrival timestamp = filename stamp written
//! by the mailhook (`YYYYMMDD-HHMMSS-<rand>.eml`). This is privileged data —
//! the command is deliberately NOT in owner.rs `USER_COMMANDS`, so the
//! default-deny gate makes it owner-DM only (OpenClaw trust model).

use std::path::{Path, PathBuf};

/// One archived message (headers + body preview only — full raw stays on
/// disk at `path`).
#[derive(Debug, Clone, PartialEq)]
pub struct MailSummary {
    /// Which local inbox it arrived in ("business" | "ai" | …).
    pub box_name: String,
    /// Archive file (absolute) the full message lives in.
    pub path: PathBuf,
    /// Arrival stamp from the filename, `YYYYMMDD-HHMMSS`.
    pub arrived: String,
    /// Envelope `From:` header (display value only, truncated to 64 chars).
    pub from: String,
    /// `Subject:` header (truncated to 80 chars).
    pub subject: String,
    /// First ~140 chars of the plain-text body, single line.
    pub preview: String,
    /// Raw `.eml` size in bytes.
    pub size_bytes: u64,
}

/// Default archive location: `$HOME/.jerith-mail/inbox-archive`.
pub fn default_base_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    Path::new(&home).join(".jerith-mail").join("inbox-archive")
}

/// List archived messages, newest first, optionally restricted to one box.
/// `limit` is applied AFTER the merge across boxes.
pub fn list_messages_in(base: &Path, box_filter: Option<&str>, limit: usize) -> Vec<MailSummary> {
    let mut out = Vec::new();
    let boxes: Vec<String> = match box_filter {
        Some(b) => vec![b.to_string()],
        None => {
            let mut dirs: Vec<String> = std::fs::read_dir(base)
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .filter(|e| e.path().is_dir())
                        .filter_map(|e| e.file_name().into_string().ok())
                        .collect()
                })
                .unwrap_or_default();
            dirs.sort();
            dirs
        }
    };
    for box_name in boxes {
        let dir = base.join(&box_name);
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("eml")))
                    .collect()
            })
            .unwrap_or_default();
        // Filename stamps sort chronologically; reverse = newest first.
        files.sort();
        files.reverse();
        for path in files {
            if out.len() >= limit {
                break;
            }
            match summarize(&path) {
                Ok(mut m) => {
                    m.box_name = box_name.clone();
                    out.push(m);
                }
                Err(e) => {
                    tracing::warn!(target: "mail", path = %path.display(), error = %e, "unreadable archived mail");
                }
            }
        }
    }
    out
}

/// Parse the minimal header set + text preview from a raw `.eml`.
pub fn summarize(path: &Path) -> anyhow::Result<MailSummary> {
    let raw = std::fs::read(path)?;
    let stamp = path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.split('-').take(2).collect::<Vec<_>>().join("-"))
        .unwrap_or_default();
    let (headers, body) = split_headers(&raw);
    let from = header_value(&headers, "from").unwrap_or_else(|| "?".into());
    let subject = header_value(&headers, "subject").unwrap_or_else(|| "(no subject)".into());
    let preview = body
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    Ok(MailSummary {
        box_name: String::new(),
        path: path.to_path_buf(),
        arrived: stamp,
        from: clip(&from, 64),
        subject: clip(&subject, 80),
        preview: clip(&preview, 140),
        size_bytes: raw.len() as u64,
    })
}

/// RFC822-ish splitter: headers until the first blank line, body after.
fn split_headers(raw: &[u8]) -> (String, String) {
    let text = String::from_utf8_lossy(raw);
    match text.find("\r\n\r\n") {
        Some(i) => (
            text[..i].to_string(),
            normalise_newlines(&text[i + 4..]).to_string(),
        ),
        None => match text.find("\n\n") {
            Some(i) => (text[..i].to_string(), text[i + 2..].to_string()),
            None => (text.to_string(), String::new()),
        },
    }
}

/// Case-insensitive header lookup with folded-continuation support. The
/// matched value accumulates into ONE buffer (folds append to it); it is
/// emitted when the next real header starts or the block ends.
fn header_value(headers: &str, name: &str) -> Option<String> {
    let prefix = format!("{name}:");
    let mut acc: Option<String> = None; // matched value under construction
    for line in headers.lines() {
        if line.starts_with(' ') || line.starts_with('\t') {
            if let Some(v) = &mut acc {
                if !v.is_empty() {
                    v.push(' ');
                }
                v.push_str(line.trim());
            }
            continue;
        }
        if let Some(v) = acc.take() {
            return Some(clip(&v, 200)); // block ended for this header
        }
        if line.len() >= prefix.len() && line[..prefix.len()].eq_ignore_ascii_case(&prefix) {
            acc = Some(line[prefix.len()..].trim().to_string());
        }
    }
    acc.map(|v| clip(&v, 200))
}

fn normalise_newlines(s: &str) -> String {
    s.replace("\r\n", "\n")
}

/// Trim + hard-clip with an ellipsis so Discord replies never blow up.
fn clip(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!(
            "{}…",
            s.chars().take(max.saturating_sub(1)).collect::<String>()
        )
    }
}

/// Discord-ready listing; empty input renders a human answer instead of "".
pub fn format_listing(messages: &[MailSummary], box_filter: Option<&str>) -> String {
    if messages.is_empty() {
        return match box_filter {
            Some(b) => format!("📬 No archived mail for `{b}@jerithai.com` yet."),
            None => {
                "📬 No archived mail yet — inboxes are empty (or the archive directory is missing)."
                    .to_string()
            }
        };
    }
    let binding = messages[0].box_name.clone();
    let mut out = match box_filter {
        Some(b) => format!(
            "📬 `{b}@jerithai.com` — {} recent message(s):\n",
            messages.len()
        ),
        None if messages.iter().all(|m| m.box_name == binding) => {
            format!(
                "📬 `{binding}@jerithai.com` — {} recent message(s):\n",
                messages.len()
            )
        }
        None => format!(
            "📬 jerithai.com archive — {} recent message(s):\n",
            messages.len()
        ),
    };
    for m in messages {
        out.push_str(&format!(
            "`{arrived}` **{boxname}** — from {from}\n  ✉ {subject}\n  > {preview}\n",
            arrived = m.arrived,
            boxname = if box_filter.is_some() {
                ""
            } else {
                m.box_name.as_str()
            },
            from = m.from,
            subject = m.subject,
            preview = m.preview,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn fixture() -> PathBuf {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = env::temp_dir().join(format!(
            "jerith-mail-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let biz = dir.join("business");
        std::fs::create_dir_all(&biz).unwrap();
        std::fs::write(
            bix(
                // newest
                &biz,
                "20261014-120001-00001.eml",
            ),
            "From: Alice <alice@example.org>\r\nSubject: first email\r\n\r\nHello from Alice.\r\n",
        )
        .unwrap();
        std::fs::write(
            bix(&biz, "20261014-120002-00002.eml"),
            "From: Bob <bob@example.org>\nTo: business@jerithai.com\nSubject: second email\n\nMulti\nline body here.\n",
        )
        .unwrap();
        dir
    }

    fn bix(dir: &Path, name: &str) -> PathBuf {
        dir.join(name)
    }

    #[test]
    fn lists_newest_first_with_parse() {
        let base = fixture();
        let msgs = list_messages_in(&base, Some("business"), 10);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].subject, "second email");
        assert_eq!(msgs[0].arrived, "20261014-120002");
        assert_eq!(msgs[0].from, "Bob <bob@example.org>");
        assert_eq!(msgs[1].subject, "first email");
        let listing = format_listing(&msgs, Some("business"));
        assert!(listing.contains("`business@jerithai.com`"));
        assert!(listing.contains("second email"));
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn limit_and_missing_box_yield_sane_results() {
        let base = fixture();
        assert_eq!(list_messages_in(&base, Some("ai"), 10).len(), 0);
        let one = list_messages_in(&base, Some("business"), 1);
        assert_eq!(one.len(), 1);
        let empty_msg = format_listing(&[], Some("ai"));
        assert!(empty_msg.contains("No archived mail for `ai@"));
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn folded_headers_and_clip_behave() {
        assert_eq!(
            header_value("Subject: long\n continued here\nFrom: x@y", "Subject"),
            Some("long continued here".into())
        );
        assert_eq!(clip("abcdefghijklmnop", 10), "abcdefghi…");
    }
}
