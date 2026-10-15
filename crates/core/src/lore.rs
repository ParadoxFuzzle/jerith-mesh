//! Lore index — void-lore-index skill port.
//!
//! Indexes a local markdown lore vault (characters, places, factions,
//! rules, artifacts — any directory of `.md` files, subdirs included) and
//! answers canon questions **with sourced excerpts**: every result carries
//! the file, the section heading, and a real excerpt from the vault. No
//! embeddings, no network — the OpenClaw skill's "excerpt + id" contract
//! kept, as a pure-Rust in-memory index rebuilt on demand (or at boot).
//!
//! Vault default: the migrated OpenClaw vault
//! (`~/.openclaw/workspace/voidsaga-vault/` — read-only borrow; the mesh
//! never writes to it). Override with the `[lore]` config table
//! (`vault_dir`, see config.rs) or `$JERITH_LORE_VAULT`.
//!
//! Node-local like memory: served via `POST /v1/skills/lore/search`
//! (gateway) and `jerith lore search` (CLI), never mesh-peered.

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::Serialize;

/// One addressable lore unit: a whole small doc or one `#`/`##` section
/// of a bigger one — the granularity the OpenClaw API served.
#[derive(Debug, Clone, Serialize)]
pub struct LoreEntry {
    /// Stable id: vault-relative path + optional `#Section` anchor.
    pub id: String,
    /// Filename stem (or front-matter `title`).
    pub title: String,
    /// Section heading text when this is a section-level entry.
    pub heading: Option<String>,
    /// Warehouse tags from YAML-ish front matter (`tags: a, b`).
    pub tags: Vec<String>,
    /// Excerpt-length body (source of truth for answers).
    pub body: String,
    /// Full body kept for search; excerpt used at render time.
    #[serde(skip)]
    pub full_text: String,
}

/// One search result: entry + the matched excerpt.
#[derive(Debug, Clone, Serialize)]
pub struct LoreHit {
    pub id: String,
    pub title: String,
    pub heading: Option<String>,
    pub tags: Vec<String>,
    /// Real excerpt from the vault around the best-matching sentence.
    pub excerpt: String,
    /// Number of query-term matches that ranked this hit here.
    pub score: u32,
}

/// The index: entries kept whole; search is substring/term-based.
#[derive(Debug, Default)]
pub struct LoreIndex {
    entries: Vec<LoreEntry>,
    /// When the index was (re)built, epoch ms.
    built_ms: u64,
}

/// Default vault location (the migrated OpenClaw vault), when present.
pub fn default_vault_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("JERITH_LORE_VAULT") {
        let p = PathBuf::from(d.trim());
        if p.is_dir() {
            return Some(p);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let p = Path::new(&home).join(".openclaw/workspace/voidsaga-vault");
    p.is_dir().then_some(p)
}

/// Parse `tags: a, b, c` out of YAML-ish front matter (the OpenClaw skill's
/// minimal subset: only the tags key is consumed).
fn parse_front_tags(front: &str) -> Vec<String> {
    front
        .lines()
        .find_map(|l| l.trim().strip_prefix("tags:"))
        .map(|rest| {
            rest.split(',')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Split a markdown doc into (heading, body) sections: `#{1,2}` headings
/// are the boundaries, the pre-heading intro belongs to the doc itself.
fn split_sections(text: &str) -> Vec<(Option<String>, String)> {
    let mut sections: Vec<(Option<String>, String)> = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut current = String::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed
            .strip_prefix("## ")
            .or_else(|| trimmed.strip_prefix("# "))
        {
            if !current.trim().is_empty() || current_heading.is_some() {
                sections.push((current_heading.take(), std::mem::take(&mut current)));
            }
            current_heading = Some(rest.trim().to_string());
        }
        current.push_str(line);
        current.push('\n');
    }
    sections.push((current_heading, current));
    sections
}

/// Build the index from a vault directory.
pub fn build(vault: &Path) -> Result<LoreIndex> {
    let mut entries: Vec<LoreEntry> = Vec::new();
    let mut files: Vec<PathBuf> = walk_md(vault, 0);
    files.sort();
    let file_count = files.len();
    for path in files {
        let rel = path
            .strip_prefix(vault)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                tracing::debug!(path = %path.display(), error = %e, "lore: unreadable vault file skipped");
                continue;
            }
        };
        // Front matter split (--- fenced YAML-ish head), tags consumed.
        let (front, body) = match text.strip_prefix("---\n") {
            Some(rest) => match rest.split_once("\n---") {
                Some((front, rest)) => (front, rest.trim_start_matches('\n')),
                None => ("", text.as_str()),
            },
            None => ("", text.as_str()),
        };
        let tags = parse_front_tags(front);
        let title = front
            .lines()
            .find_map(|l| l.trim().strip_prefix("title:"))
            .map(|t| t.trim().to_string())
            .or_else(|| {
                // First `# ` heading, else the file stem.
                body.lines()
                    .find_map(|l| l.trim_start().strip_prefix("# "))
                    .map(|t| t.trim().to_string())
            })
            .unwrap_or_else(|| {
                path.file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| rel.clone())
            });
        for (heading, section_text) in split_sections(body) {
            let trimmed = section_text.trim();
            if trimmed.chars().count() < 48 {
                continue; // not enough content to answer from
            }
            entries.push(LoreEntry {
                id: match &heading {
                    Some(h) => format!("{rel}#{h}"),
                    None => rel.clone(),
                },
                title: title.clone(),
                heading: heading.clone(),
                tags: tags.clone(),
                body: excerpt_of(trimmed, 400),
                full_text: trimmed.to_string(),
            });
        }
    }
    tracing::info!(
        entries = entries.len(),
        files = file_count,
        "lore index built"
    );
    Ok(LoreIndex {
        entries,
        built_ms: crate::bootstrap::now_ms(),
    })
}

/// Recurse `.md` files (bounded depth; skips hidden + underscores-as-first
/// data dirs are kept — vault files use `_arc.md` legitimately).
fn walk_md(dir: &Path, depth: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if depth > 6 {
        return out;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(walk_md(&p, depth + 1));
        } else if p
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.eq_ignore_ascii_case("md"))
            .unwrap_or(false)
            && !p
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.starts_with('.'))
                .unwrap_or(false)
        {
            out.push(p);
        }
    }
    out
}

/// Excerpt of `n` chars at sentence-ish boundaries.
fn excerpt_of(text: &str, n: usize) -> String {
    let mut out: String = text.chars().take(n).collect();
    if out.chars().count() < n {
        return out.trim().to_string();
    }
    if let Some(pos) = out.rfind(['.', '!', '?']) {
        out.truncate(pos + 1);
    }
    out.trim().to_string()
}

impl LoreIndex {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn built_ms(&self) -> u64 {
        self.built_ms
    }

    /// Search: score entries by case-insensitive term matches, return the
    /// top `k` with a real excerpt around the first matched term.
    pub fn search(&self, query: &str, k: usize) -> Vec<LoreHit> {
        let terms: Vec<String> = query
            .split(|c: char| c.is_whitespace() || c == ',' || c == '?')
            .map(|t| t.trim().to_ascii_lowercase())
            .filter(|t| t.chars().count() >= 3)
            // Stopwords: scoring on them drowns the real lore terms (a
            // query of "the of" would match everything otherwise).
            .filter(|t| {
                !matches!(
                    t.as_str(),
                    "the"
                        | "and"
                        | "for"
                        | "was"
                        | "were"
                        | "you"
                        | "are"
                        | "with"
                        | "that"
                        | "this"
                        | "from"
                        | "into"
                        | "when"
                        | "what"
                )
            })
            .collect();
        if terms.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(u32, &LoreEntry)> = self
            .entries
            .iter()
            .map(|e| {
                let hay = format!(
                    "{} {} {} {}",
                    e.title,
                    e.heading.as_deref().unwrap_or(""),
                    e.tags.join(" "),
                    e.full_text
                )
                .to_ascii_lowercase();
                let score: u32 = terms
                    .iter()
                    .map(|t| {
                        // Title/heading/taq matches weigh more than body hits.
                        let in_title = format!("{} {}", e.title, e.heading.as_deref().unwrap_or(""))
                            .to_ascii_lowercase()
                            .matches(t)
                            .count() as u32;
                        let in_body = hay.matches(t).count().min(4) as u32;
                        in_title * 3 + in_body
                    })
                    .sum();
                (score, e)
            })
            .filter(|(score, _)| *score > 0)
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.id.cmp(&b.1.id)));
        scored
            .into_iter()
            .take(k.max(1))
            .map(|(score, e)| LoreHit {
                id: e.id.clone(),
                title: e.title.clone(),
                heading: e.heading.clone(),
                tags: e.tags.clone(),
                excerpt: excerpt_around(&e.full_text, &terms, 400),
                score,
            })
            .collect()
    }
}

/// Real excerpt (the skill's "answer is an excerpt from a real vault file"
/// contract): the sentence window around the first matched term — the
/// sentence containing the match, kept whole, then char-capped.
fn excerpt_around(text: &str, terms: &[String], cap: usize) -> String {
    let lower = text.to_ascii_lowercase();
    let Some(first) = terms.iter().find_map(|t| lower.find(t)) else {
        return excerpt_of(text, cap);
    };
    // Sentence boundaries: start after the previous terminator, end after
    // the next one following the match.
    let before = &text[..first];
    let start = before
        .rfind(['.', '!', '?'])
        .map(|i| i + 1)
        .unwrap_or(0);
    let after = &text[first..];
    let end = after
        .find(['.', '!', '?'])
        .map(|i| first + i + 1)
        .unwrap_or(text.len());
    let sentence = text[start..end].trim();
    // Longer than the cap: fall through to the char-capped excerpt.
    if sentence.chars().count() > cap {
        let from = text[start..].trim_start();
        return excerpt_of(from, cap);
    }
    sentence.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn vault_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lore-vault-{}", std::process::id()));
        let sub = dir.join("seasons");
        let _ = std::fs::create_dir_all(&sub);
        std::fs::write(
            dir.join("the-void-whispercoil.md"),
            "---\ntitle: The Whispercoil\ntags: void, mechanics\n---\n\n# The Whispercoil\n\nThe Whispercoil spins invisible threads around Colony X. It hums at night when the void draws near, and the hum keeps the borders of the safe zones intact.\n\n## Maintenance\n\nMaintenance of the Whispercoil is forbidden without an ember-bearing escort. The last unescorted technician was lost for three days and returned without a shadow.\n",
        )
        .unwrap();
        std::fs::write(
            sub.join("colonies.md"),
            "# Colony X\n\nColony X is the last human enclave beneath the Void. Its streets remember the Ember festival. The tacos there are famous across the arc.\n\n# Colony Ash\n\nColony Ash burned in year nine of the Void. Nothing grows there now.\n",
        )
        .unwrap();
        dir
    }

    #[test]
    fn build_splits_files_and_sections() {
        let idx = build(&vault_dir()).unwrap();
        // 3 sections from whispercoil (intro + Maintenance) + 2 colonies.
        assert_eq!(
            idx.len(),
            4,
            "got {:?}",
            idx.entries.iter().map(|e| e.id.clone()).collect::<Vec<_>>()
        );
        assert!(
            idx.entries
                .iter()
                .any(|e| e.heading.as_deref() == Some("Maintenance"))
        );
        // Tags travelled from front matter into every section of the file.
        let wh = idx
            .entries
            .iter()
            .find(|e| e.id.contains("whispercoil"))
            .unwrap();
        assert_eq!(wh.tags, vec!["void".to_string(), "mechanics".to_string()]);
        assert_eq!(wh.title, "The Whispercoil");
    }

    #[test]
    fn search_answers_with_sourced_excerpts() {
        let idx = build(&vault_dir()).unwrap();
        let hits = idx.search("whispercoil safe zones", 3);
        assert!(!hits.is_empty());
        let top = &hits[0];
        assert!(top.id.contains("whispercoil"));
        assert!(
            top.excerpt.contains("Whispercoil"),
            "excerpt: {}",
            top.excerpt
        );
        assert!(top.score > 0);
        // Unrelated query returns nothing scored.
        assert!(idx.search("waffle iron recipes", 3).is_empty());
        // Too-short terms return nothing rather than everything.
        assert!(idx.search("the of", 3).is_empty());
    }

    #[test]
    fn empty_vault_is_empty_not_error() {
        let dir = std::env::temp_dir().join(format!("lore-empty-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let idx = build(&dir).unwrap();
        assert!(idx.is_empty());
        assert!(idx.search("anything", 3).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn excerpt_reads_from_sentence_start() {
        let text = "First sentence here. The Whispercoil hums at night and keeps shadows attached. Last sentence.";
        let out = excerpt_around(text, &["whispercoil".to_string()], 400);
        assert!(out.starts_with("The Whispercoil"), "got: {out}");
        assert!(out.ends_with("shadows attached.") || out.ends_with("shadows attached"));
    }
}
