//! Bot workspace — the OpenClaw workspace pattern, ported.
//!
//! OpenClaw's agent read its personality/canon from a `workspace/`
//! directory of editable markdown docs (SOUL.md, IDENTITY.md, USER.md,
//! USER_MODEL.md, AGENTS.md…). This bot ships the same layout under
//! `~/.jerith-mesh/workspace/`:
//!
//! - **`bootstrap()`** stages canon docs it doesn't yet have from the
//!   migrated OpenClaw workspace (`~/.openclaw/workspace/` — a read-only
//!   borrow, like the token/key borrows) and writes a README. Stages
//!   missing files only: user edits are never overwritten, and `~/.openclaw`
//!   is never written to.
//! - **`compose_persona()`** builds the bot's system prompt from the
//!   workspace canon in OpenClaw's own doc order (IDENTITY → SOUL →
//!   soul/* → USER → USER_MODEL → AGENTS → MEMORY), replacing the older
//!   flat `persona.md` read (which stays the fallback when no canon
//!   exists).
//! - Docs are capped individually so one bloated doc degrades gracefully
//!   instead of blowing the runtime context (main.rs applies the final
//!   hard cap over the combined prompt + known-users).

use std::path::{Path, PathBuf};

/// The canon docs the mesh bot consumes, in compose order. A doc missing
/// from the workspace is simply skipped.
///
/// The `soul/*.md` entries are the persona sub-docs that the OpenClaw
/// gateway composed implicitly and the 2026-10 migration left behind
/// (IDENTITY.md/SOUL.md still name `soul/model-identity.md` in their
/// text). They sit right after SOUL.md — they extend it. The other soul
/// sub-docs from the old stack are deliberately NOT canon: response-style/
/// trust/security duplicate SOUL.md's own sections, execute-tools/
/// local-tools describe an unreachable shell, and grounding-protocol's
/// people-file mechanics were replaced by the registry (its rules survive
/// distilled inside response-approach.md).
pub const CANON_ORDER: [&str; 9] = [
    "IDENTITY.md",
    "SOUL.md",
    "soul/model-identity.md",
    "soul/response-approach.md",
    "soul/self-improving.md",
    "USER.md",
    "USER_MODEL.md",
    "AGENTS.md",
    "MEMORY.md",
];

/// Per-doc budget for the composed persona: each canon doc keeps its head
/// up to this many chars (main.rs caps the *combined* prompt at
/// `PERSONA_MAX_CHARS` afterwards).
pub const PER_DOC_CAP_CHARS: usize = 6000;

/// `~/.jerith-mesh/workspace/`.
pub fn workspace_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Path::new(&home).join(".jerith-mesh").join("workspace")
}

/// The OpenClaw workspace to borrow from (read-only).
pub fn openclaw_workspace_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Path::new(&home).join(".openclaw").join("workspace")
}

/// README content — explains what lives here and how edits apply.
const README_MD: &str = r#"# jerith-bot workspace

Personality + canon docs for the mesh bot (OpenClaw workspace pattern).

| Doc | What it holds |
|---|---|
| `IDENTITY.md` | who the bot is (name, persona) |
| `SOUL.md` | character, voice, moral frame |
| `USER.md` | about the human(s) it talks to |
| `USER_MODEL.md` | how the bot models people / relationships |
| `AGENTS.md` | operating rules, routing notes |
| `MEMORY.md` | standing knowledge the bot carries every turn |
| `soul/model-identity.md` | honest model/stack identity rules |
| `soul/response-approach.md` | anti-autopilot response approach + pre-send self-check |
| `soul/self-improving.md` | how corrections land on the mesh (registry-based) |

Missing canon docs are staged from the migrated OpenClaw workspace on
startup; existing files are never overwritten. **Edit these files
directly** — changes apply on the next bot restart (the persona is
composed at startup, in this doc order). The post-cap combined prompt
also carries the live known-users export.

`~/.jerith-mesh/persona.md` (flat file) is only used when this workspace
holds no canon docs at all.
"#;

/// Stage canon docs when missing + the README. Returns which files were
/// newly written. Never overwrites existing files, never touches OpenClaw.
pub fn bootstrap(workspace: &Path, openclaw_ws: &Path) -> Vec<String> {
    let mut staged = Vec::new();
    let _ = std::fs::create_dir_all(workspace);
    for doc in CANON_ORDER {
        let dst = workspace.join(doc);
        if dst.exists() {
            continue;
        }
        let src = openclaw_ws.join(doc);
        match std::fs::read_to_string(&src) {
            // A too-short upstream file is probably a stub; skip it —
            // an empty doc would waste a persona slot.
            Ok(text) if text.trim().chars().count() >= 32 => {
                if std::fs::write(&dst, text).is_ok() {
                    staged.push(doc.to_string());
                }
            }
            _ => {}
        }
    }
    // README last, only when missing (a user may customize it too).
    let readme = workspace.join("README.md");
    if !readme.exists() {
        let _ = std::fs::write(&readme, README_MD);
    }
    staged
}

/// Compose the persona system prompt from the workspace canon docs, in
/// [`CANON_ORDER`], each doc independently capped. Returns `None` when the
/// workspace holds no canon docs (caller falls back to flat persona.md).
pub fn compose_persona(workspace: &Path, per_doc_max_chars: usize) -> Option<String> {
    let mut sections = Vec::new();
    for doc in CANON_ORDER {
        // A missing canon doc is skipped, never fatal — only a workspace
        // with NO canon docs at all yields None (flat persona fallback).
        let Ok(text) = std::fs::read_to_string(workspace.join(doc)) else {
            continue;
        };
        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }
        sections.push(format!(
            "## {doc}\n\n{}",
            cap_doc(trimmed, per_doc_max_chars)
        ));
    }
    if sections.is_empty() {
        return None;
    }
    Some(sections.join("\n\n"))
}

/// Cap a single doc: keep the head (canon is written first) and mark
/// truncation — reuses the persona capper so the marker text matches.
fn cap_doc(text: &str, max_chars: usize) -> String {
    crate::config::cap_system_prompt(text, max_chars)
}

/// Workspace-level bootstrap + persona hook; called from main.rs. Returns
/// (composed persona, docs newly staged). Persona is `None` when the
/// workspace holds no canon docs (caller falls back to persona.md).
pub fn load() -> (Option<String>, Vec<String>) {
    let ws = workspace_dir();
    let staged = bootstrap(&ws, &openclaw_workspace_dir());
    let persona = compose_persona(&ws, PER_DOC_CAP_CHARS);
    (persona, staged)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag() -> u32 {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        N.fetch_add(1, Ordering::SeqCst)
    }

    fn setup() -> (PathBuf, PathBuf) {
        let id = std::process::id();
        let ws = std::env::temp_dir().join(format!("ws-{}-{}-workspace", id, tag()));
        let oc = std::env::temp_dir().join(format!("ws-{}-{}-openclaw", id, tag()));
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::create_dir_all(&oc).unwrap();
        (ws, oc)
    }

    #[test]
    fn bootstrap_stages_only_missing_canon_and_a_readme() {
        let (ws, oc) = setup();
        std::fs::write(oc.join("SOUL.md"), "soul text ".repeat(10)).unwrap();
        std::fs::write(oc.join("IDENTITY.md"), "id text ".repeat(10)).unwrap();
        // A non-canon upstream doc must not be staged into the workspace.
        std::fs::write(oc.join("SOMETHING-ELSE.md"), "irrelevant".repeat(50)).unwrap();
        // A too-short upstream canon doc is treated as a stub and skipped.
        std::fs::write(oc.join("USER.md"), "tiny").unwrap();

        let staged = bootstrap(&ws, &oc);
        assert!(staged.contains(&"SOUL.md".to_string()));
        assert!(staged.contains(&"IDENTITY.md".to_string()));
        assert!(staged.iter().all(|d| CANON_ORDER.contains(&d.as_str())));
        assert!(!staged.contains(&"USER.md".to_string()));
        // Idempotent: second run stages nothing.
        assert!(bootstrap(&ws, &oc).is_empty());
        // README exists.
        assert!(ws.join("README.md").exists());
        // User edits survive a re-bootstrap.
        std::fs::write(ws.join("SOUL.md"), "edited!").unwrap();
        bootstrap(&ws, &oc);
        assert_eq!(
            std::fs::read_to_string(ws.join("SOUL.md")).unwrap(),
            "edited!"
        );
        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&oc);
    }

    #[test]
    fn persona_composes_in_canon_order_and_caps_docs() {
        let (ws, oc) = setup();
        std::fs::write(ws.join("IDENTITY.md"), "You are Jerith.").unwrap();
        std::fs::write(ws.join("SOUL.md"), "warm, feral, helpful").unwrap();
        std::fs::write(ws.join("MEMORY.md"), "x".repeat(3000)).unwrap();
        let persona = compose_persona(&ws, 800).expect("canon docs exist");
        assert!(persona.starts_with("## IDENTITY.md\n\nYou are Jerith."));
        let soul_pos = persona.find("## SOUL.md").unwrap();
        let mem_pos = persona.find("## MEMORY.md").unwrap();
        assert!(soul_pos < mem_pos);
        // The oversized doc carries the truncation marker.
        assert!(persona.contains("persona truncated at 800 chars"));
        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&oc);
    }

    #[test]
    fn soul_subdocs_compose_between_soul_and_user() {
        let (ws, oc) = setup();
        std::fs::write(ws.join("IDENTITY.md"), "id").unwrap();
        std::fs::write(ws.join("SOUL.md"), "soul").unwrap();
        std::fs::create_dir_all(ws.join("soul")).unwrap();
        std::fs::write(ws.join("soul/model-identity.md"), "model truth").unwrap();
        // gaps are skipped, never fatal — response-approach absent here
        std::fs::write(ws.join("USER.md"), "user").unwrap();
        let persona = compose_persona(&ws, 6000).expect("canon docs exist");
        let soul_pos = persona.find("## SOUL.md").unwrap();
        let mi_pos = persona.find("## soul/model-identity.md").unwrap();
        let user_pos = persona.find("## USER.md").unwrap();
        assert!(soul_pos < mi_pos && mi_pos < user_pos);
        assert!(!persona.contains("response-approach"));
        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&oc);
    }

    #[test]
    fn no_canon_docs_yields_none() {
        let (ws, oc) = setup();
        assert!(compose_persona(&ws, 800).is_none());
        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&oc);
    }
}
