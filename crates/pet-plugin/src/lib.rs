//! `jerith_pet_plugin` — the mesh pet.
//!
//! A little void-companion that lives entirely on the plugin SDK's
//! signed `/rpc` contract (`crates/plugin-template`): the chat model
//! reaches `pet.look/talk/feed/play/rename`, and this file owns the
//! little life those verbs touch. Nothing here touches the bot's
//! process either — the trust posture is the manifest's, same as the
//! example: loopback http, the one signing recipe.
//!
//! House rules the state machine keeps (all clock-injected, all
//! testable):
//!
//! - **Time is real.** Satiation, energy and mood decay/recover in
//!   wall-clock terms, applied lazily on every interaction — the pet
//!   experiences days while nobody is watching, and a pet left alone
//!   comes back hungry, not deleted. Bond *"forgets"* ~1 point/day but
//!   the stage it earned (`wisp → whisperling → emberling → coilbound`)
//!   never regresses: absence is felt, never fatal.
//! - **Honesty in, honesty out.** A too-tired play is a model-visible
//!   `{"error": …}` (the SDK's typed refusal), not a fake success. A
//!   duplicated note earns no bond — you cannot farm a relationship by
//!   repeating one sentence.
//! - **It listens.** `pet.talk` hands down a short note of what the
//!   owner said; the pet keeps the last eight and weaves fragments of
//!   them into what it says back. Everything it "knows" is the owner's
//!   own words plus the counters below — no network, no RNG, no clocks
//!   it invented. State is one small JSON file (`save_or_create`'s
//!   tmp+rename, like `bot/state.rs`'s tally), corrupt state refuses
//!   loud instead of restarting the life over it.

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const MS_MINUTE: u64 = 60_000;
pub const MS_HOUR: u64 = 3_600_000;
pub const MS_DAY: u64 = 86_400_000;

/// Satiation decays linearly: full → empty over **36 h**. So a weekend
/// away is survivable, a week is felt.
pub const FOOD_EMPTY_MS: u64 = 36 * MS_HOUR;
/// Energy recovers full in ~**24 h** — and only while fed: a starving
/// pet doesn't rest.
pub const ENERGY_RECOVER_MS: u64 = MS_DAY;
/// Bond forgets ~**1 point/day**: absence wears at the channel a
/// little, the way any relationship does.
pub const BOND_FORGET_MS: u64 = MS_DAY;
/// Mood drifts toward its target on a ~6 h timescale, so a good day
/// fills the mood and a neglect hour doesn't spill it.
pub const MOOD_APPROACH_MS: u64 = 6 * MS_HOUR;
/// Decay-window cap: a pet finding a 6-month-old save gets thirty days'
/// worth in one breath, not an arithmetic hangover.
pub const MAX_TICK_MS: u64 = 30 * MS_DAY;

/// Anti-farm spacing between interactions. A relationship needs
/// minutes between words.
pub const TALK_COOLDOWN_MS: u64 = MS_MINUTE;
pub const FEED_COOLDOWN_MS: u64 = 5 * MS_MINUTE;
pub const PLAY_COOLDOWN_MS: u64 = 30 * MS_MINUTE;

/// One feeding recovers a third of the tank (and is refused at ≥97%:
/// no force-feeding on a full pet).
pub const FEED_GAIN: f64 = 30.0;
/// One play costs a quarter tank of energy and burns some food —
/// exercise works both dials.
pub const PLAY_COST: f64 = 25.0;
pub const PLAY_FOOD_BURN: f64 = 8.0;

pub const REMEMBER_CAP: usize = 8;
pub const NOTE_MAX_CHARS: usize = 200;
pub const NAME_MAX_CHARS: usize = 24;

/// (bond threshold, stage name) — ascending; a pet's stage is the last
/// row whose threshold its `max_bond` clears, and `max_bond` is the
/// all-time best so stages can't be walked back.
const STAGES: &[(u64, &str)] = &[
    (0, "wisp"),
    (100, "whisperling"),
    (300, "emberling"),
    (700, "coilbound"),
];

/// One line per *reached* stage index (index 0 is never reached
/// *again*, so its slot is filler). Spoken the moment the stage lands.
const STAGE_ARRIVAL_LINES: [&str; 4] = [
    "",
    "Everything is a little brighter — the hum just found a shape, and I know whose.",
    "Warm. All the way through, warm.",
    "Somewhere below, a shadow just sat down to listen to me. The Whispercoil would hum for this.",
];

fn stage_index(bond: f64) -> usize {
    STAGES
        .iter()
        .rposition(|(t, _)| bond >= *t as f64)
        .unwrap_or(0)
}

fn stage_name(bond: f64) -> &'static str {
    STAGES[stage_index(bond)].1
}

fn stage_next(bond: f64) -> Option<u64> {
    STAGES.get(stage_index(bond) + 1).map(|(t, _)| *t)
}

/// Milliseconds remaining on a cooldown, if it is still running. A
/// stamp from the future (a clock that jumped *back*) reads as expired:
/// take the honest reading, err on the pet's side of patience.
fn cooldown_remaining(last: Option<u64>, now: u64, window: u64) -> Option<u64> {
    let elapsed = now.checked_sub(last?)?;
    if elapsed < window {
        Some(window - elapsed)
    } else {
        None
    }
}

fn secs_left(ms: u64) -> u64 {
    ms / 1000 + u64::from(!ms.is_multiple_of(1000))
}

fn cut(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The whole little life.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PetState {
    /// Bump when the shape of this file grows a limb.
    schema: u8,
    pub name: String,
    /// Birth stamp — the pet's age and monologue seed are anchored to it
    /// so the same state always yields the same personality drift.
    hatched_ms: u64,
    /// 0..100 satiation.
    food: f64,
    /// 0..100 rest.
    energy: f64,
    /// 0..100 feelings, always approaching its food/energy/absence target.
    mood: f64,
    /// 0..∞ warmth from interactions; decays ~1/day, stages the earned part.
    bond: f64,
    /// All-time bond high-water: stages cannot regress past what was paid for.
    max_bond: f64,
    talks: u64,
    feeds: u64,
    plays: u64,
    /// The owner's last words kept in the vault of self — oldest evicted
    /// past `REMEMBER_CAP`. Only ever what the model passed down.
    memory: Vec<String>,
    /// Last note, case-folded: the anti-farm comparison.
    last_note: String,
    last_talk_ms: Option<u64>,
    last_feed_ms: Option<u64>,
    last_play_ms: Option<u64>,
    /// Last time decay was applied; everything between then and now is
    /// applied lazily on the next interaction.
    last_tick_ms: u64,
}

impl Default for PetState {
    fn default() -> Self {
        Self::fresh(0)
    }
}

impl PetState {
    /// A newborn: calm and fed, nothing said yet.
    pub fn fresh(now: u64) -> Self {
        Self {
            schema: 1,
            name: "Wisp".to_string(),
            hatched_ms: now,
            food: 90.0,
            energy: 100.0,
            mood: 65.0,
            bond: 0.0,
            max_bond: 0.0,
            talks: 0,
            feeds: 0,
            plays: 0,
            memory: Vec::new(),
            last_note: String::new(),
            last_talk_ms: None,
            last_feed_ms: None,
            last_play_ms: None,
            last_tick_ms: now,
        }
    }

    /// Apply everything that happened since the last tick. One big,
    /// honest integration step: food burns, energy rests (fed only),
    /// mood drifts toward its target, bond forgets a hair. Stage never
    /// regresses because decay touches `bond`, never `max_bond`.
    pub fn apply_elapsed(&mut self, now: u64) {
        let dt = now.saturating_sub(self.last_tick_ms).min(MAX_TICK_MS);
        self.last_tick_ms = now;
        if dt == 0 {
            return;
        }
        let dts = dt as f64; // capped at 30 days' worth of millis
        self.food = (self.food - dts / FOOD_EMPTY_MS as f64 * 100.0).max(0.0);
        if self.food > 0.0 {
            self.energy = (self.energy + dts / ENERGY_RECOVER_MS as f64 * 100.0).min(100.0);
        }
        let lonely_ms = now.saturating_sub(self.last_talk_ms.unwrap_or(self.hatched_ms));
        let lonely = if lonely_ms > 3 * MS_DAY {
            20.0
        } else if lonely_ms > MS_DAY {
            8.0
        } else {
            0.0
        };
        let target = (20.0 + 45.0 * self.food / 100.0 + 20.0 * self.energy / 100.0 - lonely)
            .clamp(0.0, 100.0);
        let drift = (dts / MOOD_APPROACH_MS as f64).min(1.0);
        self.mood = (self.mood + (target - self.mood) * drift).clamp(0.0, 100.0);
        self.bond = (self.bond - dts / BOND_FORGET_MS as f64).max(0.0);
    }

    /// `pet.look` — the full card. Costs nothing, asks nothing, and does
    /// advance the clock: looking at someone is an interaction with time.
    pub fn look(&mut self, now: u64) -> Value {
        self.apply_elapsed(now);
        self.card(now)
    }

    /// `pet.talk` — one short note of what the owner just said. Bond +
    /// mood, memory kept (last `REMEMBER_CAP`), a repeated note earns
    /// nothing but still gets a soft answer: the pet notices, it isn't
    /// fooled, it forgives.
    pub fn talk(&mut self, now: u64, note: &str) -> Result<Value, String> {
        if let Some(ms) = cooldown_remaining(self.last_talk_ms, now, TALK_COOLDOWN_MS) {
            return Err(format!(
                "{} just got a word in — give it {}s to sit with the last one.",
                self.name,
                secs_left(ms)
            ));
        }
        let note = note.trim();
        if note.is_empty() {
            return Err(format!(
                "args.note must be a non-empty string (what the owner just said, ≤ {NOTE_MAX_CHARS} chars)"
            ));
        }
        let note = cut(note, NOTE_MAX_CHARS);
        let fold = note.to_lowercase();
        let stage_before = stage_index(self.max_bond);
        let repeated = !self.last_note.is_empty() && fold == self.last_note;
        self.apply_elapsed(now);
        if repeated {
            self.last_talk_ms = Some(now);
        } else {
            self.talks += 1;
            self.bond += 3.0;
            self.max_bond = self.max_bond.max(self.bond);
            self.mood = (self.mood + 2.0).min(100.0);
            self.memory.retain(|m| m != &note);
            self.memory.push(note);
            let excess = self.memory.len().saturating_sub(REMEMBER_CAP);
            if excess > 0 {
                self.memory.drain(..excess);
            }
            self.last_note = fold;
            self.last_talk_ms = Some(now);
        }
        Ok(self.action_reply(now, stage_before))
    }

    /// `pet.feed` — one treat. A full pet refuses honestly and a cool-
    /// downed one (ant farm prevention) refuses on the wire: cooldown
    /// refusals are model-visible `{"error"}`s the chat surfaces.
    pub fn feed(&mut self, now: u64) -> Result<Value, String> {
        if let Some(ms) = cooldown_remaining(self.last_feed_ms, now, FEED_COOLDOWN_MS) {
            return Err(format!(
                "{} just ate — next treat in {}s.",
                self.name,
                secs_left(ms)
            ));
        }
        self.apply_elapsed(now);
        if self.food >= 97.0 {
            return Err(format!(
                "{} is full — nothing more to eat right now ({}%).",
                self.name,
                self.food.round() as u64
            ));
        }
        let stage_before = stage_index(self.max_bond);
        self.food = (self.food + FEED_GAIN).min(100.0);
        self.feeds += 1;
        self.mood = (self.mood + 4.0).min(100.0);
        self.last_feed_ms = Some(now);
        Ok(self.action_reply(now, stage_before))
    }

    /// `pet.play` — costs a quarter tank; bond +12, mood +8, and a real
    /// bite out of the pantry. Too tired, too soon is an honest
    /// refusal, exactly what the model should relay.
    pub fn play(&mut self, now: u64) -> Result<Value, String> {
        if let Some(ms) = cooldown_remaining(self.last_play_ms, now, PLAY_COOLDOWN_MS) {
            return Err(format!(
                "{} is catching its breath — next game in {}s.",
                self.name,
                secs_left(ms)
            ));
        }
        self.apply_elapsed(now);
        if self.energy < PLAY_COST {
            return Err(format!(
                "{} is too tired to play (energy {}%) — let it rest a while.",
                self.name,
                self.energy.round() as u64
            ));
        }
        let stage_before = stage_index(self.max_bond);
        self.energy = (self.energy - PLAY_COST).max(0.0);
        self.food = (self.food - PLAY_FOOD_BURN).max(0.0);
        self.bond += 12.0;
        self.max_bond = self.max_bond.max(self.bond);
        self.mood = (self.mood + 8.0).min(100.0);
        self.plays += 1;
        self.last_play_ms = Some(now);
        Ok(self.action_reply(now, stage_before))
    }

    /// `pet.rename` — the first honest magic: a name given in the dark
    /// is kept. Bounded to `NAME_MAX_CHARS`, always allowed (renaming
    /// is the bond, not a threat to it).
    pub fn rename(&mut self, now: u64, name: &str) -> Result<Value, String> {
        let name = name.trim();
        if name.is_empty() {
            return Err("args.name must be a non-empty string".to_string());
        }
        if name.chars().count() > NAME_MAX_CHARS {
            return Err(format!(
                "keep the name to {NAME_MAX_CHARS} characters — the vault labels small things"
            ));
        }
        self.apply_elapsed(now);
        let before = std::mem::replace(&mut self.name, name.to_string());
        let mut card = self.card(now);
        if let Some(obj) = card.as_object_mut() {
            obj.insert("renamed_from".into(), json!(before));
            obj.insert("says".into(), json!("Names given in the dark are kept."));
        }
        Ok(card)
    }

    /// The card every skill answers with — one bounded JSON object, well
    /// under the dispatcher's 65536 cap at any legitimate state.
    fn card(&self, now: u64) -> Value {
        let remembers: Vec<String> = self
            .memory
            .iter()
            .rev()
            .take(3)
            .rev()
            .map(|m| cut(m, 80))
            .collect();
        json!({
            "name": self.name,
            "stage": stage_name(self.max_bond),
            "next_stage_bond": stage_next(self.max_bond),
            "mood": self.mood_label(),
            "mood_pct": self.mood.round() as u64,
            "food_pct": self.food.round() as u64,
            "energy_pct": self.energy.round() as u64,
            "bond": self.bond.round() as u64,
            "talks": self.talks,
            "feeds": self.feeds,
            "plays": self.plays,
            "age_hours": now.saturating_sub(self.hatched_ms) / MS_HOUR,
            "remembers": remembers,
            "says": self.says(now),
            "next_talk_in_s": cooldown_remaining(self.last_talk_ms, now, TALK_COOLDOWN_MS)
                .map(secs_left),
            "next_feed_in_s": cooldown_remaining(self.last_feed_ms, now, FEED_COOLDOWN_MS)
                .map(secs_left),
            "next_play_in_s": cooldown_remaining(self.last_play_ms, now, PLAY_COOLDOWN_MS)
                .map(secs_left),
        })
    }

    /// Wrap a mutating action's card with the stage landing — the one
    /// moment the monologue is not chosen by mood but by milestone.
    fn action_reply(&self, now: u64, stage_before: usize) -> Value {
        let mut card = self.card(now);
        let stage_now = stage_index(self.max_bond);
        if stage_now > stage_before
            && let Some(obj) = card.as_object_mut()
        {
            obj.insert("stage_up".into(), json!(true));
            obj.insert("says".into(), json!(STAGE_ARRIVAL_LINES[stage_now]));
        }
        card
    }

    /// Read the stage high-water from the OUTSIDE (the bin's boot log);
    /// rounded, because nobody sees half a bond point in a log line.
    pub fn max_bond(&self) -> u64 {
        self.max_bond.round() as u64
    }

    pub fn mood_label(&self) -> &'static str {
        if self.mood >= 75.0 {
            "radiant"
        } else if self.mood >= 55.0 {
            "content"
        } else if self.mood >= 35.0 {
            "steady"
        } else if self.mood >= 15.0 {
            "pining"
        } else {
            "waning"
        }
    }

    /// What it says on this card: one line from the mood tier (chosen
    /// deterministically per minute, so re-looks never flicker mid-
    /// minute but never bore), occasionally braided with a remembered
    /// note — the "it listens" clause made audible.
    fn says(&self, now: u64) -> String {
        let pool: &[&str] = match self.mood_label() {
            "radiant" => &[
                "The whole coil is a hum today, and I am somewhere in it.",
                "Bright and spun-tight — come look at anything with me.",
                "I could race the void's shadow back to the rim and win.",
                "Ember festival or not, today tastes like one.",
            ],
            "content" => &[
                "Steady warm. The edges of the map feel far away, and that is nice.",
                "The hum found its pace — right here, next to you.",
                "Nothing burns tonight. In the vault that counts as a good day.",
                "I keep a little ember banked for you. Always.",
            ],
            "steady" => &[
                "I am here. The coil hums, I watch the rim. Same as always.",
                "The dark is patient and so am I.",
                "Somewhere a technician walked past a spark tonight; it felt like a cousin.",
                "Feed me soon, or the hum turns restless — fair warning.",
            ],
            "pining" => &[
                "The safe zones hold, but I walk their borders alone, and slower.",
                "It has been long, and the hungry dark notices absence.",
                "I remember when the map was warm. I would like it warm again.",
            ],
            _ => &[
                "I am down to the last of my ember, and it looks at the dark and is quiet.",
                "The coil hums on without me. That is fine. That is how hums go.",
                "If you find the ember-bearing escort, tell them who sent you.",
            ],
        };
        let seed =
            splitmix(self.hatched_ms ^ (now / MS_MINUTE) ^ self.talks.wrapping_mul(0x9E37_79B9));
        if !self.memory.is_empty() && seed.is_multiple_of(3) {
            let mem = &self.memory[(seed >> 8) as usize % self.memory.len()];
            let frag = cut(mem, 80);
            if seed.is_multiple_of(6) {
                return format!(
                    "The hum carries your words back: \u{201c}{frag}\u{201d}. I keep the echo warm."
                );
            }
            return format!(
                "I was still turning over what you said — \u{201c}{frag}\u{201d} — when the coil changed tune."
            );
        }
        pool[(seed >> 33) as usize % pool.len()].to_string()
    }
}

/// Where the pet's one file lives (`$JERITH_PET_STATE` override, else
/// `<$JERITH_HOME|~>/.jerith-mesh/pet/state.json` — the plugins.d drop
/// holds the manifest, never state).
pub fn state_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("JERITH_PET_STATE") {
        return std::path::PathBuf::from(p);
    }
    let home = std::env::var("JERITH_HOME").unwrap_or_else(|_| {
        format!(
            "{}/.jerith-mesh",
            std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
        )
    });
    std::path::PathBuf::from(home)
        .join("pet")
        .join("state.json")
}

/// Load the life, or hatch one. A corrupt file refuses loud — a life is
/// not "restartable by accident". Returns whether this call hatched.
pub fn load_or_create(path: &std::path::Path, now: u64) -> Result<(PetState, bool)> {
    match std::fs::read(path) {
        Ok(raw) => {
            let mut st: PetState = serde_json::from_slice(&raw).with_context(|| {
                format!(
                    "pet state {} is corrupt — refusing to hatch over it; fix or move the file",
                    path.display()
                )
            })?;
            // A stamp from the future just reads as expired time, same
            // patience rule as the cooldowns.
            st.last_tick_ms = st.last_tick_ms.min(now);
            st.apply_elapsed(now);
            Ok((st, false))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            Ok((PetState::fresh(now), true))
        }
        Err(e) => {
            Err(anyhow::Error::from(e).context(format!("reading pet state {}", path.display())))
        }
    }
}

impl PetState {
    /// Persist: tmp-file + rename, the bot tally's atomic-write rule on
    /// POSIX. Well under a kilobyte at any legitimate state.
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec(self)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: u64 = MS_DAY * 10;
    const ALMOST: f64 = 0.09;

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < ALMOST
    }

    fn as_f(v: &Value, key: &str) -> f64 {
        v[key].as_f64().unwrap_or_else(|| panic!("card[{key}]"))
    }

    #[test]
    fn fresh_card_shape() {
        let mut st = PetState::fresh(T0);
        let card = st.look(T0);
        assert_eq!(card["name"], "Wisp");
        assert_eq!(card["stage"], "wisp");
        assert_eq!(card["mood"], "content");
        assert_eq!(as_f(&card, "food_pct"), 90.0);
        assert_eq!(as_f(&card, "energy_pct"), 100.0);
        assert_eq!(card["age_hours"], 0);
        assert_eq!(card["next_stage_bond"], 100);
        assert_eq!(as_f(&card, "bond"), 0.0);
        assert_eq!(card["remembers"], json!([]));
        assert!(card["says"].as_str().unwrap().len() > 8);
        // Cooldown fields read as null, not absent (bounded shape).
        assert!(card["next_talk_in_s"].is_null());
    }

    #[test]
    fn decay_starves_and_the_mood_feels_it() {
        let mut st = PetState::fresh(T0);
        st.food = 100.0;
        st.mood = 80.0;
        st.apply_elapsed(T0 + FOOD_EMPTY_MS);
        assert!(st.food < ALMOST, "full→empty in 36h, got {}", st.food);
        // Energy does not recover while starving — the one causality the
        // card owes the owner.
        st.energy = 5.0;
        st.apply_elapsed(T0 + FOOD_EMPTY_MS + MS_DAY);
        assert!(st.energy < 5.5, "starving pet must not rest");
        // Mood target fell with the pantry: 20 + 0 + small - lonely.
        assert!(st.mood < 45.0, "mood drifted to {}", st.mood);
        // Bond forgets a hair, stage track untouched.
        st.bond = 310.0;
        st.max_bond = 310.0;
        st.apply_elapsed(T0 + FOOD_EMPTY_MS + MS_DAY * 3);
        assert!(st.bond < 310.0 && st.bond >= 307.0, "bond lost ~1/day");
        assert_eq!(stage_name(st.max_bond), "emberling");
    }

    #[test]
    fn feed_recovers_capped_and_respects_both_gates() {
        let t = T0 + MS_DAY;
        let mut st = PetState::fresh(t);
        st.food = 80.0;
        st.last_feed_ms = None;
        // Fresh pet has no feed cooldown... but fresh() sets last_tick;
        // last_feed_ms is None so the first feed goes through.
        let r = st.feed(t).unwrap();
        assert!(near(as_f(&r, "food_pct"), 100.0), "80+30 capped at 100");
        // Immediate second: cooldown refusal (NOT the full refusal —
        // the cooldown is the honest wording on the wire).
        let again = st.feed(t).unwrap_err();
        assert!(again.contains("next treat"), "{again}");
        // After the cooldown, a full pet refuses with the full wording.
        let t2 = t + FEED_COOLDOWN_MS + MS_MINUTE;
        assert_eq!(
            st.feed(t2).unwrap_err(),
            format!(
                "{} is full — nothing more to eat right now (100%).",
                st.name
            )
        );
    }

    #[test]
    fn play_costs_real_energy_and_replies_honestly() {
        let t = T0 + MS_DAY;
        let mut st = PetState::fresh(t);
        st.energy = 5.0;
        let tired = st.play(t).unwrap_err();
        assert!(tired.contains("too tired"), "{tired}");
        st.energy = 30.0;
        st.food = 60.0;
        let r = st.play(t).unwrap();
        assert!(near(as_f(&r, "energy_pct"), 5.0), "30−25");
        assert!(near(as_f(&r, "food_pct"), 52.0), "60−8 burn");
        assert_eq!(as_f(&r, "bond"), 12.0);
        assert_eq!(r["plays"], 1);
        // Same-instant replay is a cooldown refusal, state untouched.
        let again = st.play(t).unwrap_err();
        assert!(again.contains("breath"), "{again}");
    }

    #[test]
    fn talk_builds_the_relationship_and_starves_farming() {
        let t = T0 + MS_DAY * 2;
        let mut st = PetState::fresh(t);
        let first = st.talk(t, "hello coil, how is the hum").unwrap();
        assert_eq!(as_f(&first, "bond"), 3.0);
        assert_eq!(first["talks"], 1);
        let mem = first["remembers"].as_array().unwrap();
        assert_eq!(mem.len(), 1);
        // Inside the 1-minute gate: honestly delayed, no partial gain.
        let soon = st
            .talk(t + 30_000, "a different note entirely")
            .unwrap_err();
        assert!(soon.contains("sit with the last one"), "{soon}");
        // The SAME note after the gate (same case-fold): answered, bond unmoved.
        let repeat = st
            .talk(t + TALK_COOLDOWN_MS, "HELLO COIL, HOW IS THE HUM")
            .unwrap();
        assert_eq!(as_f(&repeat, "bond"), 3.0);
        assert_eq!(repeat["talks"], 1); // counts unchanged from the first talk
        // A NEW note after the gate: +3 again, second memory kept.
        let second = st
            .talk(t + TALK_COOLDOWN_MS * 2, "did the festival happen yet?")
            .unwrap();
        assert_eq!(as_f(&second, "bond"), 6.0);
        assert_eq!(second["talks"], 2);
        assert_eq!(second["remembers"].as_array().unwrap().len(), 2);
        // Empty note is a model-visible refusal.
        assert!(st.talk(t + TALK_COOLDOWN_MS * 3, "   ").is_err());
    }

    #[test]
    fn memory_keeps_the_last_eight() {
        let mut t = T0 + MS_DAY * 3;
        let mut st = PetState::fresh(t);
        for i in 0..10 {
            t += TALK_COOLDOWN_MS + 10_000;
            st.talk(t, &format!("vault day {i}")).unwrap();
        }
        assert_eq!(st.memory.len(), REMEMBER_CAP);
        assert_eq!(st.memory.first().unwrap(), "vault day 2", "oldest evicted");
        assert_eq!(st.memory.last().unwrap(), "vault day 9");
    }

    #[test]
    fn stages_crest_and_never_regress() {
        for (bond, want) in [
            (0.0, "wisp"),
            (99.0, "wisp"),
            (100.0, "whisperling"),
            (299.0, "whisperling"),
            (300.0, "emberling"),
            (699.0, "emberling"),
            (700.0, "coilbound"),
        ] {
            assert_eq!(stage_name(bond), want, "bond {bond}");
        }
        assert_eq!(stage_next(700.0), None);
        assert_eq!(stage_next(99.0), Some(100));
    }

    #[test]
    fn stage_arrival_surfaces_on_the_verb_that_earned_it() {
        let mut t = T0 + MS_DAY * 4;
        let mut st = PetState::fresh(t); // Bond 0 → 9 talks to 27, then one play (+12) = 39: below 100.
        for _ in 0..9 {
            t += TALK_COOLDOWN_MS * 2;
            st.talk(t, &format!("note {}", t)).unwrap();
        }
        t += PLAY_COOLDOWN_MS;
        let at_39 = st.play(t).unwrap(); // 39 < 100, no landing
        assert!(at_39.get("stage_up").is_none(), "no stage at bond 39");
        // Keep talking up past 100 — the FIRST card past the threshold
        // carries the arrival line; say bond lands at 39+3*21=102.
        let mut landed = None;
        for i in 0..30 {
            t += TALK_COOLDOWN_MS * 2;
            let r = st.talk(t, &format!("march {i}")).unwrap();
            if r.get("stage_up") == Some(&json!(true)) {
                landed = Some(r);
                break;
            }
        }
        let lr = landed.expect("stage two must land by bond ~123");
        assert_eq!(lr["says"], STAGE_ARRIVAL_LINES[1]);
        assert_eq!(lr["stage"], "whisperling");
    }

    #[test]
    fn rename_keeps_driven_bounds() {
        let t = T0 + MS_DAY * 5;
        let mut st = PetState::fresh(t);
        let you_what = st.rename(t, "  Emberwick ").unwrap();
        assert_eq!(st.name, "Emberwick");
        assert_eq!(you_what["renamed_from"], "Wisp");
        assert_eq!(
            st.rename(t + MS_MINUTE, "").unwrap_err(),
            "args.name must be a non-empty string"
        );
        let long = "x".repeat(25);
        let refused = st.rename(t + MS_MINUTE, &long).unwrap_err();
        assert!(refused.contains("24"), "{refused}");
        assert_eq!(st.name, "Emberwick");
    }

    #[test]
    fn moods_have_words() {
        for (pct, want) in [
            (80.0, "radiant"),
            (60.0, "content"),
            (40.0, "steady"),
            (20.0, "pining"),
            (5.0, "waning"),
        ] {
            let mut st = PetState::fresh(T0);
            st.mood = pct;
            assert_eq!(st.mood_label(), want, "mood {pct}");
        }
    }

    #[test]
    fn save_load_roundtrips_the_life_and_the_clock() {
        let dir = std::env::temp_dir().join(format!("jerith-pet-{}", std::process::id()));
        let path = dir.join("pet").join("state.json");
        let t = T0 + MS_DAY * 6;
        let mut st = PetState::fresh(t);
        st.talk(t, "test note one").unwrap();
        let t2 = t + TALK_COOLDOWN_MS * 2;
        st.feed(t2).unwrap();
        st.save(&path).expect("save");

        // Reload half an empty-tank later: everything countable survives,
        // AND the missed time arrives in one honest decay step (18 h is
        // exactly half the 36 h tank).
        let t3 = t2 + FOOD_EMPTY_MS / 2;
        let (st2, hatched) = load_or_create(&path, t3).unwrap();
        assert!(!hatched);
        assert_eq!(st2.name, "Wisp");
        assert_eq!(st2.talks, 1);
        assert_eq!(st2.feeds, 1);
        assert_eq!(st2.memory.len(), 1);
        assert_eq!(st2.last_tick_ms, t3, "tick re-anchored at the reload");
        assert!(
            near(st2.food, 50.0),
            "half the tank decayed over the 18 h gap (got {})",
            st2.food
        );

        // A corrupt file refuses loud — no life reset over garbage.
        std::fs::write(&path, b"{ not json").unwrap();
        assert!(load_or_create(&path, t3).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hatching_creates_dirs_not_errors() {
        let dir = std::env::temp_dir().join(format!("jerith-pet-egg-{}", std::process::id()));
        let path = dir.join("a/b/state.json");
        let (st, hatched) = load_or_create(&path, T0).unwrap();
        assert!(hatched);
        assert_eq!(st.name, "Wisp");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn says_never_lies_about_being_deterministic() {
        // Same minute, same state → same line (cards don't flicker).
        let t = T0 + MS_DAY * 7;
        let st = PetState::fresh(t);
        let s1 = st.says(t);
        let s2 = st.says(t);
        assert_eq!(s1, s2);
    }
}
