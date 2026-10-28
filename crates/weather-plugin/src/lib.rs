//! The weather skill's pure half (OpenClaw `weather` skill family,
//! re-homed over the plugin SDK with real upstream data).
//!
//! Everything here is clockless, networkless parsing over the public
//! open-meteo.com APIs (no api key tracked anywhere):
//!
//! - geocoding: `geocoding-api.open-meteo.com/v1/search?name=…` —
//!   count=1 best match for a human place name;
//! - forecast: `api.open-meteo.com/v1/forecast?latitude=…` —
//!   `current` + `daily` series, metric, tz=auto;
//!
//! plus the two leading URL builders and the human "card" text each
//! skill result carries (the model gets BOTH: structured JSON to quote
//! and a card to paste). Errors are typed strings (the signed /rpc
//! contract's refusal shape), never panics: a dead upstream, an empty
//! geocode result or a shape drift answer with an honest line.

use serde::Deserialize;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Upstream wire schema (serde views; every field the handlers slice)
// ---------------------------------------------------------------------------

/// One geocode best-match row (`count=1`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Place {
    pub name: String,
    #[serde(default)]
    pub latitude: f64,
    #[serde(default)]
    pub longitude: f64,
    #[serde(default)]
    pub country: Option<String>,
    #[serde(default)]
    pub admin1: Option<String>,
    #[serde(default)]
    pub timezone: Option<String>,
    #[serde(default)]
    pub population: Option<u64>,
}

/// Parsed geocode API body (we keep only `results[0]`).
#[derive(Debug, Deserialize)]
struct GeocodeBody {
    #[serde(default)]
    results: Vec<Place>,
}

/// `current` block of a forecast call, metric units.
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct Current {
    #[serde(default)]
    pub time: String,
    #[serde(default)]
    pub temperature_2m: Option<f64>,
    #[serde(default)]
    pub apparent_temperature: Option<f64>,
    #[serde(default)]
    pub relative_humidity_2m: Option<f64>,
    #[serde(default)]
    pub weather_code: Option<u8>,
    #[serde(default)]
    pub wind_speed_10m: Option<f64>,
    #[serde(default)]
    pub wind_direction_10m: Option<f64>,
}

/// One `daily` series row.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Day {
    #[serde(default)]
    pub date: String,
    #[serde(default)]
    pub weather_code: Option<u8>,
    #[serde(default)]
    pub temperature_2m_max: Option<f64>,
    #[serde(default)]
    pub temperature_2m_min: Option<f64>,
    /// Null upstream when the day is beyond the precip model horizon —
    /// None "not stated", never "0%".
    #[serde(default)]
    pub precipitation_probability_max: Option<u8>,
}

/// Full forecast call body (`current` + `daily` + context).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ForecastBody {
    #[serde(default)]
    pub latitude: f64,
    #[serde(default)]
    pub longitude: f64,
    #[serde(default)]
    pub timezone: Option<String>,
    #[serde(default)]
    pub current: Current,
    #[serde(default)]
    pub daily: Daily,
}

#[derive(Debug, Default, Clone, PartialEq, Deserialize)]
pub struct Daily {
    #[serde(default)]
    pub time: Vec<String>,
    #[serde(default)]
    pub weather_code: Vec<Option<u8>>,
    #[serde(default)]
    pub temperature_2m_max: Vec<Option<f64>>,
    #[serde(default)]
    pub temperature_2m_min: Vec<Option<f64>>,
    #[serde(default)]
    pub precipitation_probability_max: Vec<Option<u8>>,
}

// ---------------------------------------------------------------------------
// URL builders (pure — the handlers feed reqwest with these)
// ---------------------------------------------------------------------------

/// Percent-encode a query parameter (reserved chars always). Mirrors
/// gifs.rs's urlencode — a shared shape, not a new convention.
pub fn urlencode(s: &str) -> String {
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

/// The count=1 geocode search URL (open-meteo geocoding, English
/// names, JSON body).
pub fn geocode_url(place: &str) -> String {
    format!(
        "https://geocoding-api.open-meteo.com/v1/search?name={}&count=1&language=en&format=json",
        urlencode(place.trim())
    )
}

/// The forecast URL for one coordinate: metric, auto-tz, `current`
/// block + `days` days of the daily series the cards need.
pub fn forecast_url(latitude: f64, longitude: f64, days: u32) -> String {
    format!(
        "https://api.open-meteo.com/v1/forecast?latitude={:.4}&longitude={:.4}\
         &current=temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m\
         &daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max\
         &forecast_days={days}&timezone=auto",
        latitude, longitude
    )
}

/// Operator-controlled day count: clamp 1–7 (upstream's own horizon) —
/// 0/day-8 asks are refusals at the edges, never upstream 400s.
pub fn clamp_days(n: u32) -> u32 {
    n.clamp(1, 7)
}

// ---------------------------------------------------------------------------
// WMO 4677 weather-code table (what the model and cards show)
// ---------------------------------------------------------------------------

/// One-line human text for the standard WMO 4677 code set up-meteo
/// returns. Unknown code → honest "code N".
pub fn code_text(code: u8) -> &'static str {
    match code {
        0 => "clear sky",
        1 => "mainly clear",
        2 => "partly cloudy",
        3 => "overcast",
        45 => "fog",
        48 => "depositing rime fog",
        51 => "light drizzle",
        53 => "moderate drizzle",
        55 => "dense drizzle",
        56 => "light freezing drizzle",
        57 => "dense freezing drizzle",
        61 => "slight rain",
        63 => "moderate rain",
        65 => "heavy rain",
        66 => "light freezing rain",
        67 => "heavy freezing rain",
        71 => "slight snowfall",
        73 => "moderate snowfall",
        75 => "heavy snowfall",
        77 => "snow grains",
        80 => "slight rain showers",
        81 => "moderate rain showers",
        82 => "violent rain showers",
        85 => "slight snow showers",
        86 => "heavy snow showers",
        95 => "thunderstorm",
        96 => "thunderstorm with slight hail",
        99 => "thunderstorm with heavy hail",
        _ => "unknown condition",
    }
}

// ---------------------------------------------------------------------------
// Body parsing (typed errors, feed both card AND JSON paths)
// ---------------------------------------------------------------------------

/// The geocode body's best match. An empty `results` (upstream's
/// honest "no such place") errors typed — the caller surfaces "no
/// place matched 'xyz' — check spelling".
pub fn parse_geocode(body: &str) -> Result<Place, String> {
    let geo: GeocodeBody =
        serde_json::from_str(body).map_err(|e| format!("geocode body not a search result: {e}"))?;
    geo.results.into_iter().next().ok_or_else(|| {
        "no place matched — check the spelling (e.g. \"New York\", not \"ny\")".to_string()
    })
}

/// The forecast body's full series. Shape drift (missing `current` or
/// `daily`) fails typed; null fields stay Option (upstream may omit).
pub fn parse_forecast(body: &str) -> Result<ForecastBody, String> {
    let body: ForecastBody =
        serde_json::from_str(body).map_err(|e| format!("forecast body not a forecast: {e}"))?;
    if body.current.temperature_2m.is_none() {
        return Err("forecast body has no current observations".to_string());
    }
    if body.daily.time.is_empty() {
        return Err("forecast body has no daily series".to_string());
    }
    Ok(body)
}

// ---------------------------------------------------------------------------
// Cards (the human-paste text riding under each JSON answer)
// ---------------------------------------------------------------------------

/// One-line place label ("Prague, Czechia" — admin1 joins when the
/// country field carries none).
pub fn place_label(p: &Place) -> String {
    let mut label = p.name.clone();
    if let Some(admin) = &p.admin1
        && *admin != p.name
    {
        label = format!("{label}, {admin}");
    }
    if let Some(country) = &p.country {
        label = format!("{label}, {country}");
    }
    label
}

fn trunc(s: &str, n: usize) -> String {
    let mut c = n;
    while !s.is_char_boundary(c) {
        c -= 1;
    }
    s[..c].to_string()
}

/// The user-facing edit on a label for card use: bound to 40 chars so
/// a geocoded monster like `Llanfair­pwllgwyngyll­gogery­chwyrndrobwll­llantysilio­gogogoch`
/// still renders a one-line card.
fn card_label(p: &Place) -> String {
    let label = place_label(p);
    if label.chars().count() <= 40 {
        return label;
    }
    format!("{}…", trunc(&label, 39))
}

fn f(x: &Option<f64>) -> String {
    match x {
        Some(v) => format!("{v:.1}"),
        None => "?".to_string(),
    }
}

/// The `weather.now` card: three lines, metric, time-stamped.
pub fn now_card(place: &Place, tz: &str, now: &Current) -> String {
    let head = format!(
        "🌤️ {} — {}",
        card_label(place),
        code_text(now.weather_code.unwrap_or(255))
    );
    let mut lines = vec![
        head,
        format!(
            "🌡️ {} °C (feels {} °C), 💧 {}%, 💨 {} km/h",
            f(&now.temperature_2m),
            f(&now.apparent_temperature),
            now.relative_humidity_2m
                .map(|v| format!("{v:.0}"))
                .unwrap_or_else(|| "?".into()),
            f(&now.wind_speed_10m)
        ),
    ];
    if !now.time.is_empty() {
        lines.push(format!("⏱️ observed {} local ({tz})", now.time));
    }
    lines.join("\n")
}

/// The `weather.forecast` card: one line per day, high/low + pop %
/// where the model states it.
pub fn forecast_cards(place: &Place, tz: &str, days: &[Day]) -> String {
    let mut lines = vec![format!(
        "📅 {} — next {} day(s) local ({tz})",
        card_label(place),
        days.len()
    )];
    for d in days {
        let date = if d.date.is_empty() {
            "?"
        } else {
            d.date.as_str()
        };
        let hi = d
            .temperature_2m_max
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "?".into());
        let lo = d
            .temperature_2m_min
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "?".into());
        let pop = match d.precipitation_probability_max {
            Some(v) => format!("{v:.0}%"),
            None => "–".to_string(),
        };
        lines.push(format!(
            "• {date}: {} °C / {} °C, 💧 {pop}, {}",
            hi,
            lo,
            code_text(d.weather_code.unwrap_or(255))
        ));
    }
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// JSON assembly (what the XML expands to — the model's tool content)
// ---------------------------------------------------------------------------

/// Build the `weather.now` RESULT: provider + place + current block +
/// the card. `Value`-only: no serde struct needs the wire contract.
pub fn now_result(place: &Place, tz: &str, now: &Current) -> Value {
    serde_json::json!({
        "provider": "open-meteo",
        "place": {
            "name": place.name,
            "country": place.country,
            "latitude": place.latitude,
            "longitude": place.longitude,
            "timezone": tz,
        },
        "current": {
            "observed_at": now.time,
            "condition": code_text(now.weather_code.unwrap_or(255)),
            "temperature_c": now.temperature_2m,
            "apparent_c": now.apparent_temperature,
            "humidity_percent": now.relative_humidity_2m,
            "wind_speed_kmh": now.wind_speed_10m,
            "wind_direction_deg": now.wind_direction_10m,
        },
        "card": now_card(place, tz, now),
    })
}

/// Build the `weather.forecast` RESULT: place + day rows + the card.
pub fn forecast_result(place: &Place, tz: &str, days: &[Day]) -> Value {
    serde_json::json!({
        "provider": "open-meteo",
        "place": {
            "name": place.name,
            "country": place.country,
            "latitude": place.latitude,
            "longitude": place.longitude,
            "timezone": tz,
        },
        "forecast": days.iter().map(|d| serde_json::json!({
            "date": d.date,
            "condition": code_text(d.weather_code.unwrap_or(255)),
            "high_c": d.temperature_2m_max,
            "low_c": d.temperature_2m_min,
            "precipitation_probability_percent": d.precipitation_probability_max,
        })).collect::<Vec<_>>(),
        "card": forecast_cards(place, tz, days),
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urlencode_always_keeps_reserved_out() {
        assert_eq!(urlencode("New York"), "New%20York");
        assert_eq!(urlencode("Reykjavík"), "Reykjav%C3%ADk");
        assert_eq!(urlencode("a&b=1"), "a%26b%3D1");
        assert_eq!(urlencode("río_-~."), "r%C3%ADo_-~.");
    }

    #[test]
    fn urls_pin_upstream_shapes() {
        // Place names trim before encoding (" São Paulo " is "São Paulo"
        // to upstream — spealer whitespace is user noise, not part of a
        // place name).
        let g = geocode_url(" São Paulo ");
        assert!(g.starts_with("https://geocoding-api.open-meteo.com/v1/search?name=S%C3%A3o%20Paulo&count=1&language=en&format=json"), "{g}");
        let f = forecast_url(50.087, 14.421, 3);
        assert!(
            f.contains("latitude=50.0870")
                && f.contains("longitude=14.4210")
                && f.contains("&forecast_days=3&timezone=auto")
                && f.contains("current=temperature_2m,apparent_temperature"),
            "{f}"
        );
    }

    #[test]
    fn days_clamp_at_the_upstream_horizon() {
        assert_eq!(clamp_days(0), 1);
        assert_eq!(clamp_days(3), 3);
        assert_eq!(clamp_days(12), 7);
    }

    #[test]
    fn wmo_spots_cover_every_family() {
        assert_eq!(code_text(0), "clear sky");
        assert_eq!(code_text(3), "overcast");
        assert_eq!(code_text(45), "fog");
        assert_eq!(code_text(65), "heavy rain");
        assert_eq!(code_text(75), "heavy snowfall");
        assert_eq!(code_text(95), "thunderstorm");
        assert_eq!(code_text(99), "thunderstorm with heavy hail");
        assert_eq!(code_text(123), "unknown condition");
        assert_eq!(code_text(100), "unknown condition");
    }

    #[test]
    fn geocode_parses_best_match_and_empties_fail_typed() {
        let body = r#"{"results":[{
            "id":536,"name":"Prahu","latitude":-6.12144,"longitude":106.1336,
            "elevation":5,"feature_code":"PPLC","country_code":"ID",
            "country":"Indonesia","admin1":"Central Bangka","timezone":"Asia/Jakarta",
            "population":1400000
        }]}"#;
        let p = parse_geocode(body).expect("one row parses");
        assert_eq!(p.name, "Prahu");
        assert_eq!(p.country.as_deref(), Some("Indonesia"));
        assert_eq!(p.population, Some(1_400_000));
        let empty = r#"{"results":[]}"#;
        let err = parse_geocode(empty).unwrap_err();
        assert!(err.contains("no place matched"), "{err}");
        // Garbage bodies fail typed (never a panic).
        assert!(parse_geocode("{oops}").is_err());
    }

    #[test]
    fn forecast_parse_uses_both_series_and_tolerates_nils() {
        // compacted, shape-faithful: two days, second has no precip prob.
        let body = r#"{
          "latitude":50.5,"longitude":14.4,"timezone":"Europe/Prague",
          "current":{"time":"2026-10-27T14:00","temperature_2m":12.5,
                     "apparent_temperature":10.4,"relative_humidity_2m":79.0,
                     "weather_code":2,"wind_speed_10m":11.9,"wind_direction_10m":272.0},
          "daily":{"time":["2026-10-27","2026-10-28"],
                   "weather_code":[2,3],
                   "temperature_2m_max":[14.5,13.2],
                   "temperature_2m_min":[7.7,6.4],
                   "precipitation_probability_max":[40,null]}
        }"#;
        let fc = parse_forecast(body).expect("parses");
        assert_eq!(fc.current.temperature_2m, Some(12.5));
        assert_eq!(fc.timezone.as_deref(), Some("Europe/Prague"));
        assert_eq!(fc.daily.time.len(), 2);
        assert_eq!(fc.daily.precipitation_probability_max[1], None);

        // Shape drift: no `current` block fails typed, never a panic.
        let bad = r#"{"daily":{"time":["2026-10-27"],"weather_code":[0],"temperature_2m_max":[1.0],"temperature_2m_min":[0.0],"precipitation_probability_max":[null]}}"#;
        let err = parse_forecast(bad).unwrap_err();
        assert!(err.contains("no current observations"), "{err}");
        // No daily series fails typed too.
        assert!(parse_forecast(r#"{"current":{"time":"t","temperature_2m":1.0}}"#).is_err());
    }

    #[test]
    fn cards_are_human_paste() {
        let place = Place {
            name: "Prague".into(),
            latitude: 50.09,
            longitude: 14.42,
            country: Some("Czechia".into()),
            admin1: Some("Prague".into()),
            timezone: None,
            population: None,
        };
        let now = Current {
            time: "2026-10-27T14:30".into(),
            temperature_2m: Some(12.543),
            apparent_temperature: Some(10.4),
            relative_humidity_2m: Some(79.0),
            weather_code: Some(2),
            wind_speed_10m: Some(11.9),
            wind_direction_10m: Some(272.0),
        };
        let card = now_card(&place, "Europe/Prague", &now);
        assert!(card.contains("Prague"), "{card}");
        assert!(card.contains("partly cloudy"), "{card}");
        assert!(card.contains("12.5 °C"), "{card}");
        assert!(card.contains("observed "), "{card}");
        // Dedup: admin1 = name never double-prints Prague, Prague.
        assert!(!card.contains("Prague, Prague"));

        let days = vec![Day {
            date: "2026-10-28".into(),
            weather_code: Some(96),
            temperature_2m_max: Some(13.2),
            temperature_2m_min: Some(6.4),
            precipitation_probability_max: Some(80),
        }];
        let fcard = forecast_cards(&place, "Europe/Prague", &days);
        assert!(fcard.contains("2026-10-28"), "{fcard}");
        assert!(fcard.contains("13.2 °C / 6.4 °C"), "{fcard}");
        assert!(fcard.contains("80%"), "{fcard}");
        assert!(fcard.contains("thunderstorm with slight hail"), "{fcard}");
        // A precip-less day is a "–", never "0%".
        let nopop = Day {
            precipitation_probability_max: None,
            ..days[0].clone()
        };
        let fcard2 = forecast_cards(&place, "Europe/Prague", &[nopop]);
        assert!(fcard2.contains("💧 –"), "{fcard2}");
    }

    #[test]
    fn monster_labels_cannot_wreck_a_card() {
        let place = Place {
            name: "Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch".into(),
            latitude: 53.22,
            longitude: -4.21,
            country: Some("United Kingdom".into()),
            admin1: Some("Wales".into()),
            timezone: None,
            population: None,
        };
        let now = Current::default();
        let card = now_card(&place, "UTC", &now);
        // The LABEL is capped at 40+1; the head line still carries the
        // emoji + condition text — bound at 70 keeps the full card to
        // 3 lines on a narrow terminal, monster name included.
        assert!(card.lines().next().unwrap().chars().count() < 70, "{card}");
        assert!(card.contains("…"), "{card}");
    }

    #[test]
    fn result_json_carries_both_structured_and_card() {
        let place = Place {
            name: "Prahu".into(),
            latitude: -6.1,
            longitude: 106.1,
            country: Some("Indonesia".into()),
            admin1: None,
            timezone: None,
            population: None,
        };
        let now = Current {
            time: "2026-10-27T19:00".into(),
            temperature_2m: Some(27.0),
            apparent_temperature: Some(30.0),
            relative_humidity_2m: Some(80.0),
            weather_code: Some(0),
            wind_speed_10m: Some(5.0),
            wind_direction_10m: None,
        };
        let v = now_result(&place, "Asia/Jakarta", &now);
        assert_eq!(v["provider"], "open-meteo");
        assert_eq!(v["place"]["timezone"], "Asia/Jakarta");
        assert_eq!(v["current"]["temperature_c"], 27.0);
        assert_eq!(v["current"]["wind_direction_deg"], Value::Null);
        assert!(v["card"].as_str().unwrap().contains("clear sky"));

        let days = vec![Day {
            date: "2026-10-28".into(),
            weather_code: Some(3),
            temperature_2m_max: Some(10.0),
            temperature_2m_min: Some(2.0),
            precipitation_probability_max: None,
        }];
        let f = forecast_result(&place, "Asia/Jakarta", &days);
        assert_eq!(
            f["forecast"][0]["precipitation_probability_percent"],
            Value::Null
        );
        assert!(f["card"].as_str().unwrap().contains("next 1 day(s)"));
    }
}
