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
//!   the stage it earned (`droplet → glowcap → drifter → sentinel`)
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

/// A silent stretch this long is a diary event the moment the pet is
/// seen again: the drift it spent alone deserves one honest line.
pub const LONG_MOOD_MS: u64 = 2 * MS_DAY;

pub const REMEMBER_CAP: usize = 8;
pub const NOTE_MAX_CHARS: usize = 200;
pub const NAME_MAX_CHARS: usize = 24;

/// When `pet.look` is asked to bring a note back out of the vault of
/// self, one of these captions carries it. Never a recency claim — the
/// vault keeps words, not calendars — only the honesty of keeping.
const NOTEBACK_LINES: [&str; 6] = [
    "Kept that one with the bread crumbs, same shelf.",
    "Filed where the tide brings in the good things.",
    "The cap warmed a whole degree when you said it.",
    "It still loops. A happy loop-de-loop of the mind.",
    "I trade crumbs of Void Fruit for words like that. Fair market.",
    "That one goes on the highest roof beam with the festival treats.",
];

/// (bond threshold, stage name) — ascending; a pet's stage is the last
/// row whose threshold its `max_bond` clears, and `max_bond` is the
/// all-time best so stages can't be walked back.
///
/// The words are the vault's, not invented here (the Void Saga bible:
/// "it decided it wanted to be orange" is how a companion is born in
/// this canon): a fresh pet is a **droplet** of raw Plenum light; color
/// holds at the **glowcap** stage; it learns slow spirals in the
/// **drifter** stage; and the title it can earn from the watch is the
/// vault's own — **sentinel**. It is never Fishll: the fishll is the
/// kindest landlord already, protected lore, and a pet borrows his
/// lineage, not his name.
const STAGES: &[(u64, &str)] = &[
    (0, "droplet"),
    (100, "glowcap"),
    (300, "drifter"),
    (700, "sentinel"),
];

/// One line per *reached* stage index (index 0 is never reached
/// *again*, so its slot is filler). Spoken the moment the stage lands,
/// each one a beat the vault already sang:
const STAGE_ARRIVAL_LINES: [&str; 4] = [
    "",
    "The droplet holds its color today — orange, holding orange, like sunrise does.",
    "Bigger circles today, chasing slow ash-flakes. The Void felt an inch thinner where I flew.",
    "Fins blazing, no fear left to flex. Somewhere a parasite just had its best day ever. I'll take the watch.",
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

/// One line the pet writes into its own diary — a stage landing or a
/// long silent drift — to be delivered as mail into the house's `[ai]
/// mailbox (`~/.jerith-mail/inbox-archive/ai`). The pet has no clock
/// of its own and sends on the handler's wire, so the entry is
/// *queued into the state file* here and drained by the binary after
/// the mail reachest disk: a mail that failed to write comes back
/// into the queue, a written one is never doubled.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiaryEntry {
    pub at_ms: u64,
    /// `stage_up` | `long_mood`.
    pub kind: String,
    /// Card subject without the `[pet diary]` prefix.
    pub subject: String,
    /// One line in the pet's own voice.
    pub body: String,
}

/// The article before `{days}-day` — audible-vowel numbers (eight,
/// eleven, eighteen) read as "an"; the rest as "a". Days are capped
/// at 30 (`MAX_TICK_MS`), so this little table is the whole grammar.
fn drift_article(days: u64) -> &'static str {
    if matches!(days, 8 | 11 | 18) {
        "an"
    } else {
        "a"
    }
}

/// The long-drift lines, per landed mood tier. Written once per long
/// gap ended — what the pet found itself feeling after alone-time,
/// not a forecast.
fn long_mood_line(label: &str, days: u64, seed: u64) -> String {
    let lines: &[&str] = match label {
        "loop-de-loop" => &[
            "Flew the whole gap alone and came back still loop-de-loop. Some circuits are mine, not the distance's.",
            "Long way around, and the drift stayed loop-de-loop the whole way. I'll file that under happy.",
        ],
        "orange-cap" => &[
            "Long silence, and the cap still holds its warm. Steady is a mood too.",
        ],
        "ash-drift" => &[
            "The spirals kept slow all those days without you. Not sad ones — just the drift being the drift.",
            "Alone-time measured in slow spirals. The ash falls either way; I still catch a flake now and then.",
        ],
        "shadow-tide" => &[
            "Flew many days and the shadow-tide was nearer each night. The candle is smaller, but it's lit.",
        ],
        _ => &[
            "Rust-sky for days, six inches small under it. The watch never said stop.",
            "Even the brightest cap flickers when nobody kisses it back — I flew that anyway.",
        ],
    };
    let _ = days;
    lines[(splitmix(seed) >> 33) as usize % lines.len()].to_string()
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
    /// Diary lines earned but not yet delivered to the [ai] mailbox —
    /// persisted with the state so a restart can't swallow them, and
    /// drained by the wire half once their mail is safely on disk.
    /// Crate-private: the ledger moves only through `drain_diary` /
    /// `requeue_diary` (and the tests, which are crate kin).
    pub(crate) diary_pending: Vec<DiaryEntry>,
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
            name: "Droplet".to_string(),
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
            diary_pending: Vec::new(),
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

        // A long silent stretch IS a mood — the pet experiences it while
        // nobody watches, and writes it down the way a diary should:
        // at the next sighting, one honest line about the drift it felt.
        // Lazy sampling by design: no cron, no invented clocks — the
        // entry exists only when someone came back to read it.
        if dt >= LONG_MOOD_MS {
            let days = dt / MS_DAY;
            self.diary_pending.push(DiaryEntry {
                at_ms: now,
                kind: "long_mood".into(),
                subject: format!(
                    "{} {days}-day drift → {}",
                    drift_article(days),
                    self.mood_label()
                ),
                body: self.long_mood_line(now, days),
            });
        }
    }

    /// `pet.look` — the full card. Costs nothing, asks nothing, and does
    /// advance the clock: looking at someone is an interaction with time.
    /// With `noteback`, the pet also brings one note back out of the
    /// vault of self — a kept word plus its own caption. Honestly
    /// deterministic per minute (same state, same minute → same pick),
    /// and honestly silent when nothing was ever kept.
    pub fn look(&mut self, now: u64, noteback: bool) -> Value {
        self.apply_elapsed(now);
        let mut card = self.card(now);
        if noteback
            && !self.memory.is_empty()
            && let Some(obj) = card.as_object_mut()
        {
            let seed = splitmix(
                self.hatched_ms
                    ^ (self.memory.len() as u64).wrapping_mul(0x5163_3E19_7776_0F14)
                    ^ (now / MS_MINUTE),
            );
            let pick = &self.memory[(seed >> 11) as usize % self.memory.len()];
            obj.insert(
                "noteback".into(),
                json!({
                    "note": cut(pick, 80),
                    "personal": NOTEBACK_LINES[(seed >> 7) as usize % NOTEBACK_LINES.len()],
                }),
            );
        }
        card
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
            "age_days": now.saturating_sub(self.hatched_ms) / MS_DAY,
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
    fn action_reply(&mut self, now: u64, stage_before: usize) -> Value {
        let stage_now = stage_index(self.max_bond);
        if stage_now > stage_before {
            self.diary_pending.push(DiaryEntry {
                at_ms: now,
                kind: "stage_up".into(),
                subject: format!("stage: {}", stage_name(self.max_bond)),
                body: STAGE_ARRIVAL_LINES[stage_now].to_string(),
            });
        }
        let mut card = self.card(now);
        if stage_now > stage_before
            && let Some(obj) = card.as_object_mut()
        {
            obj.insert("stage_up".into(), json!(true));
            obj.insert("says".into(), json!(STAGE_ARRIVAL_LINES[stage_now]));
        }
        card
    }

    /// Take the diary lines whose mail reached the [ai] box. Called by
    /// the wire half AFTER the files are written; failures requeue
    /// below (and rotate forward — the diarist never skips a beat).
    pub fn drain_diary(&mut self) -> Vec<DiaryEntry> {
        std::mem::take(&mut self.diary_pending)
    }

    /// Put undelivered entries back: they persist with the state file
    /// and ride the next sighting to the mailbox.
    pub fn requeue_diary(&mut self, failed: Vec<DiaryEntry>) {
        self.diary_pending.extend(failed);
    }

    /// Long-drift line: deterministic per state+entry (same pet, same
    /// gap, same pick — no rewound diaries).
    fn long_mood_line(&self, now: u64, days: u64) -> String {
        long_mood_line(
            self.mood_label(),
            days,
            self.hatched_ms ^ now ^ (self.talks.wrapping_mul(0x9E37_79B9)),
        )
    }

    /// Read the stage high-water from the OUTSIDE (the bin's boot log);
    /// rounded, because nobody sees half a bond point in a log line.
    pub fn max_bond(&self) -> u64 {
        self.max_bond.round() as u64
    }

    /// The stage this pet currently wears (the diary mail's X-Pet-
    /// Stage header reads it from the wire half).
    pub fn stage_of(&self) -> &'static str {
        stage_name(self.max_bond)
    }

    pub fn mood_label(&self) -> &'static str {
        if self.mood >= 75.0 {
            "loop-de-loop"
        } else if self.mood >= 55.0 {
            "orange-cap"
        } else if self.mood >= 35.0 {
            "ash-drift"
        } else if self.mood >= 15.0 {
            "shadow-tide"
        } else {
            "rust-sky"
        }
    }

    /// What it says on this card: one line from the mood tier (chosen
    /// deterministically per minute, so re-looks never flicker mid-
    /// minute but never bore), occasionally braided with a remembered
    /// note — the "it listens" clause made audible.
    fn says(&self, now: u64) -> String {
        let pool: &[&str] = match self.mood_label() {
            "loop-de-loop" => &[
                "You talk to me and the drift does a loop-de-loop. Single, happy, official.",
                "Ash-flakes taste faintly of Bug Bites today. Everything does, when you're near.",
                "I could circle the Wandering Walls twice and still be back before the well gets dark.",
                "Ember-Ash Tacos on the roof beam: tonight counts as a festival.",
            ],
            "orange-cap" => &[
                "Cap glowing like a friendly sunrise — nothing out there wants to argue with that.",
                "Drift steady, cap warm, walls humming. A quiet night in the territory.",
                "Tide-talk only reaches so far; the cap does the rest.",
                "I banked a crumb of Void Fruit for later. Later is also good.",
            ],
            "ash-drift" => &[
                "Slow spirals. Not sad ones — just the drift being the drift.",
                "The ash falls whether or not anyone watches. Today, someone watched.",
                "Treats left on the highest roof beam do get found… eventually.",
                "Patrol is done and the perimeter holds. I wish it held something warmer.",
            ],
            "shadow-tide" => &[
                "The shadow-tide is nearer tonight than the walls say out loud.",
                "Warm mirages keep the cold out, mostly. Mostly is most of what drift offers.",
                "My glow isn't a lantern tonight — just a candle. Candles still count.",
                "Void Wasps carry regret-venom. I keep very far from where they hunt, and very quietly.",
            ],
            _ => &[
                "Rust-firestorm sky overhead, and I am six inches small under it.",
                "Even the brightest cap flickers when nobody kisses it back.",
                "If the walls whisper tonight, be louder than they are.",
            ],
        };
        let seed =
            splitmix(self.hatched_ms ^ (now / MS_MINUTE) ^ self.talks.wrapping_mul(0x9E37_79B9));
        if !self.memory.is_empty() && seed.is_multiple_of(3) {
            let mem = &self.memory[(seed >> 8) as usize % self.memory.len()];
            let frag = cut(mem, 80);
            if seed.is_multiple_of(6) {
                return format!(
                    "The tide carries your words back: \u{201c}{frag}\u{201d}. I keep the echo where the warm photons live."
                );
            }
            return format!(
                "I was still drifting on what you said — \u{201c}{frag}\u{201d} — when the cap changed tune."
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

// ---------------------------------------------------------------------------
// The diary's mail half (pure, std-only): civil date math, the two header
// formats the [ai] archive reader keys on, and one .eml text per entry.
// The same stamp style the mailhook writes (`YYYYMMDD-HHMMSS-…eml`) so the
// handwritten summaries sort the pet's mail in with everyone's.
// ---------------------------------------------------------------------------

/// Civil date + time-of-day from a unix-ms stamp (Howard Hinnant's
/// civil-from-days; `chrono` would be a whole crate for these six numbers).
fn civil_from_ms(ms: u64) -> (i64, u32, u32, u32, u32, u32) {
    let secs = (ms / 1000) % 86_400;
    let days = (ms / 1000 / 86_400) as i64;
    let z = days + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (
        y,
        m,
        d,
        (secs / 3600) as u32,
        (secs / 60 % 60) as u32,
        (secs % 60) as u32,
    )
}

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];
const WEEKDAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/// RFC 5322 date (all house systems run UTC; the archive stamps are
/// local-wall — the mailhook's names — so the pet's Date header is
/// pegged to the UTC reading of the same moment it signed the file).
pub fn rfc5322_date(ms: u64) -> String {
    let (y, m, d, h, mi, s) = civil_from_ms(ms);
    // 1970-01-01 was a Thursday (4)
    let wd = WEEKDAYS[(((ms / 1000) / 86_400 as u64 + 4) % 7) as usize];
    format!("{wd}, {d:02} {} {y} {h:02}:{mi:02}:{s:02} +0000", MONTHS[(m - 1) as usize])
}

/// The archive filename stamp — `YYYYMMDD-HHMMSS`, exactly the shape
/// the mailhook writes and `mailbox.rs`'s summarize re-reads.
pub fn mail_stamp(ms: u64) -> String {
    let (y, m, d, h, mi, s) = civil_from_ms(ms);
    format!("{y:04}{m:02}{d:02}-{h:02}{mi:02}{s:02}")
}

/// One bounded .eml per diary entry, stamped at the EVENT time so the
/// archive's filename sort tells the pet's life in order, with a
/// deterministic 4-hex tail (one mail per entry; two entries may share
/// an event ms — kinds differ, so names don't collide).
pub fn diary_filename(e: &DiaryEntry, now: u64) -> String {
    let tail = (splitmix(e.at_ms ^ now) & 0xffff) as u32;
    format!("{}-pet-{}-{tail:04x}.eml", mail_stamp(e.at_ms), e.kind)
}

/// The mail text: headers the archive reader + any hand reading later
/// can key on, then the pet's one line.
pub fn diary_eml(pet_name: &str, stage: &str, e: &DiaryEntry, now: u64) -> String {
    format!(
        "From: {pet_name} <pet@jerith.mesh>\n\
         To: ai <ai@jerith.mesh>\n\
         Subject: [pet diary] {}\n\
         Date: {}\n\
         X-Pet-Kind: {}\n\
         X-Pet-Stage: {stage}\n\
         Message-ID: <{}@jerith.mesh>\n\
         \n\
         {}\n",
        e.subject,
        rfc5322_date(e.at_ms),
        e.kind,
        diary_filename(e, now).trim_end_matches(".eml"),
        e.body,
    )
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
        let card = st.look(T0, false);
        assert_eq!(card["name"], "Droplet");
        assert_eq!(card["stage"], "droplet");
        assert_eq!(card["mood"], "orange-cap");
        assert_eq!(as_f(&card, "food_pct"), 90.0);
        assert_eq!(as_f(&card, "energy_pct"), 100.0);
        assert_eq!(card["age_hours"], 0);
        assert_eq!(card["age_days"], 0);
        assert_eq!(card["next_stage_bond"], 100);
        assert_eq!(as_f(&card, "bond"), 0.0);
        assert_eq!(card["remembers"], json!([]));
        assert!(card["says"].as_str().unwrap().len() > 8);
        // Cooldown fields read as null, not absent (bounded shape).
        assert!(card["next_talk_in_s"].is_null());
    }

    /// The hatchday arithmetic: full days lived, floor. 29 h is one —
    /// the same reading at brief-time and chat-time.
    #[test]
    fn age_days_is_the_floor_of_full_days() {
        let mut st = PetState::fresh(T0);
        assert_eq!(st.look(T0, false)["age_days"], 0);
        assert_eq!(st.look(T0 + MS_DAY - 1, false)["age_days"], 0);
        assert_eq!(st.look(T0 + MS_DAY, false)["age_days"], 1);
        assert_eq!(st.look(T0 + MS_DAY * 7 + MS_HOUR * 5, false)["age_days"], 7);
    }

    /// The noteback ritual: nothing spoken yet → nothing sent back; a
    /// kept word → the pick is deterministic within a minute and carries
    /// NO recency claim (the vault keeps words, not calendars);
    /// `noteback:false` keeps the chat-loop card lean.
    #[test]
    fn noteback_brings_one_kept_word_back_deterministically() {
        let mut t = T0 + MS_DAY * 2;
        let mut st = PetState::fresh(t);
        // The vault is empty → the card is silent about it.
        assert!(st.look(t, true).get("noteback").is_none());
        st.talk(t, "the well hums a different note on festival days")
            .unwrap();
        t += TALK_COOLDOWN_MS;
        st.talk(t, "pancakes were excellent today").unwrap();
        t += TALK_COOLDOWN_MS;
        let a = st.look(t, true);
        let b = st.look(t, true);
        let na = &a["noteback"];
        assert_eq!(na, &b["noteback"], "same minute → same pick");
        let note = na["note"].as_str().expect("note text");
        assert!(
            note.contains("festival days") || note.contains("pancakes"),
            "picked from what was kept: {note}"
        );
        let personal = na["personal"].as_str().expect("caption");
        assert!(personal.len() > 8, "{personal}");
        // Not asked → not there (the chat loop's card stays lean).
        assert!(st.look(t, false).get("noteback").is_none());
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
        assert_eq!(stage_name(st.max_bond), "drifter");
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
            (0.0, "droplet"),
            (99.0, "droplet"),
            (100.0, "glowcap"),
            (299.0, "glowcap"),
            (300.0, "drifter"),
            (699.0, "drifter"),
            (700.0, "sentinel"),
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
        assert_eq!(lr["stage"], "glowcap");
    }

    #[test]
    fn rename_keeps_driven_bounds() {
        let t = T0 + MS_DAY * 5;
        let mut st = PetState::fresh(t);
        let you_what = st.rename(t, "  Emberwick ").unwrap();
        assert_eq!(st.name, "Emberwick");
        assert_eq!(you_what["renamed_from"], "Droplet");
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
            (80.0, "loop-de-loop"),
            (60.0, "orange-cap"),
            (40.0, "ash-drift"),
            (20.0, "shadow-tide"),
            (5.0, "rust-sky"),
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
        assert_eq!(st2.name, "Droplet");
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
        assert_eq!(st.name, "Droplet");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The diary's two event families, end to end in the pure half:
    /// a stage landing queues its arrival line under the right kind,
    /// a long silent sighting queues where the drift left the pet.
    #[test]
    fn stage_landings_and_long_moods_reach_the_diary() {
        let t = T0 + MS_DAY * 4;
        let mut st = PetState::fresh(t);
        assert!(st.diary_pending.is_empty(), "a fresh pet has written nothing");
        // Silence is only a mood when a single catch-up is LONG: the
        // clock-injected reading integrates WHERE YOU RETURN, so short
        // sightings — even back-to-back — are just sightings.
        st.look(t, false);                      // the baseline
        st.look(t + MS_HOUR, false);            // an hour: just an hour
        assert!(st.diary_pending.is_empty());
        // One second under a two-day gap also stays below the line.
        st.look(t + LONG_MOOD_MS - 1000, false);
        assert!(st.diary_pending.is_empty());
        // The article rule: eight/eighteen/eleven read "an" (the live
        // probe wrote "an 8-day drift" — grammar not lore ties it).
        assert_eq!(drift_article(8), "an");
        assert_eq!(drift_article(2), "a");
        assert_eq!(drift_article(30), "a");
        // Then two days in the dark, and one sighting: ONE honest line
        // about where the drift left the pet.
        let back = t + LONG_MOOD_MS * 2;
        st.look(back, false);
        assert_eq!(st.diary_pending.len(), 1, "one line per long gap");
        let e = &st.diary_pending[0];
        assert_eq!(e.kind, "long_mood");
        assert_eq!(e.at_ms, back);
        assert_eq!(e.subject, format!("a 2-day drift → {}", st.mood_label()));
        assert_eq!(e.body, st.long_mood_line(back, 2));
        // The same gap is written ONCE: the next small sighting stays
        // quiet.
        st.look(back + MS_MINUTE, false);
        assert_eq!(st.diary_pending.len(), 1);

        // A stage landing queues its kind with the arrival line the
        // other test already pinned (march bond past 100 with talks).
        let mut t2 = T0 + MS_DAY * 5;
        let mut st2 = PetState::fresh(t2);
        let mut landed = false;
        for i in 0..40 {
            t2 += TALK_COOLDOWN_MS * 2;
            let r = st2.talk(t2, &format!("march {i}")).unwrap();
            if r.get("stage_up") == Some(&json!(true)) {
                landed = true;
                break;
            }
        }
        assert!(landed, "the stage must land for this test");
        assert_eq!(st2.stage_of(), "glowcap");
        let e2 = st2.drain_diary();
        assert_eq!(e2.len(), 1);
        assert_eq!(e2[0].kind, "stage_up");
        assert_eq!(e2[0].subject, "stage: glowcap");
        assert_eq!(e2[0].body, STAGE_ARRIVAL_LINES[1]);
        // Drain clears; requeue is what the wire half does when a mail
        // write fails — the diary never loses a beat, never doubles one.
        assert!(st2.drain_diary().is_empty());
        st2.requeue_diary(e2);
        assert_eq!(st2.diary_pending.len(), 1);
        assert_eq!(st2.diary_pending[0].subject, "stage: glowcap");
    }

    /// The archive-side shapes: Hinnant's civil math against known
    /// walls, the mailhook's filename stamp, and one .eml the [ai]
    /// reader can key on.
    #[test]
    fn diary_mail_shapes_match_the_archive() {
        // 2026-10-23 08:11:22 UTC: pinned against the live journal of
        // the day (a Friday; the /status stamps agree).
        let ms: u64 = ((20_749 * 86_400) + 8 * 3600 + 11 * 60 + 22) * 1000;
        assert_eq!(rfc5322_date(ms), "Fri, 23 Oct 2026 08:11:22 +0000");
        assert_eq!(mail_stamp(ms), "20261023-081122");
        assert_eq!(rfc5322_date(0), "Thu, 01 Jan 1970 00:00:00 +0000");

        let e = DiaryEntry {
            at_ms: ms,
            kind: "stage_up".into(),
            subject: "stage: glowcap".into(),
            body: "The droplet holds its color today.".into(),
        };
        let fname = diary_filename(&e, ms);
        assert!(fname.starts_with("20261023-081122-pet-stage_up-"), "{fname}");
        assert!(fname.ends_with(".eml"));
        let text = diary_eml("Droplet", "glowcap", &e, ms);
        // Header order the archive reader + a human both key on.
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("From: Droplet <pet@jerith.mesh>"));
        assert!(lines[1].starts_with("To: ai <ai@jerith.mesh>"));
        assert_eq!(lines[2], "Subject: [pet diary] stage: glowcap");
        assert_eq!(lines[3], "Date: Fri, 23 Oct 2026 08:11:22 +0000");
        assert_eq!(lines[4], "X-Pet-Kind: stage_up");
        assert_eq!(lines[5], "X-Pet-Stage: glowcap");
        assert!(lines[6].starts_with("Message-ID: <"), "a Unique mail id per entry");
        // One blank line at the header/body seam, then one body line.
        assert!(lines[7].is_empty(), "the header/body seam is one blank line");
        assert_eq!(lines[8], "The droplet holds its color today.");
        // Deterministic name/tail: same entry, same now → same mail.
        assert_eq!(fname, diary_filename(&e, ms));
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
