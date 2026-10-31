//! The credential-free read path, pinned: shapes, parsing, and the honesty
//! rule that a feed read never invents a score Reddit did not send.

use jerith_reddit_plugin::{
    ANON_BASE, Caught, CheckReport, Post, Watch, load_watch, parse_atom_comments, parse_atom_posts,
    parse_listing, parse_rfc3339_secs, post_json, rss_listing_url, rss_search_url, rss_thread_url,
    save_watch, score_label,
};

/// Trimmed to the fields Reddit's public feed actually sends: a link post
/// and a self post.
const FEED: &str = r#"<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom"><category term="rust" label="r/rust"/>
<entry><author><name>/u/seino_chan</name><uri>https://www.reddit.com/user/seino_chan</uri></author><category term="rust" label="r/rust"/><content type="html">&amp;#32; submitted by &amp;#32; &lt;a href=&quot;https://www.reddit.com/user/seino_chan&quot;&gt; /u/seino_chan &lt;/a&gt; &lt;br/&gt; &lt;span&gt;&lt;a href=&quot;https://this-week-in-rust.org/blog/2026/10/28/x/&quot;&gt;[link]&lt;/a&gt;&lt;/span&gt; &amp;#32; &lt;span&gt;&lt;a href=&quot;https://www.reddit.com/r/rust/comments/1xi7826/twir/&quot;&gt;[comments]&lt;/a&gt;&lt;/span&gt;</content><id>t3_1xi7826</id><link href="https://www.reddit.com/r/rust/comments/1xi7826/twir/" /><updated>2026-10-29T01:49:15+00:00</updated><published>2026-10-29T01:49:15+00:00</published><title>This Week in Rust &amp; #675</title></entry>
<entry><author><name>/u/UnclothedButFree</name><uri>https://www.reddit.com/user/UnclothedButFree</uri></author><category term="selfhosted" label="r/selfhosted"/><content type="html">&lt;!-- SC_OFF --&gt;&lt;div class=&quot;md&quot;&gt;&lt;p&gt;Booru project&amp;#39;s submissions are closed.&lt;/p&gt; &lt;p&gt;I don&amp;#39;t code.&lt;/p&gt; &lt;/div&gt;&lt;!-- SC_ON --&gt; &amp;#32; submitted by &amp;#32; &lt;a href=&quot;https://www.reddit.com/user/UnclothedButFree&quot;&gt; /u/UnclothedButFree &lt;/a&gt;</content><id>t3_1xkmyjl</id><link href="https://www.reddit.com/r/selfhosted/comments/1xkmyjl/is_there_a_way/" /><updated>2026-10-31T09:02:00+00:00</updated><published>2026-10-31T09:02:00+00:00</published><title>Is there a way &lt;cheap&gt; to do this?</title></entry>
</feed>"#;

const COMMENTS: &str = r#"<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><author><name>/u/helpful</name></author><content type="html">&lt;p&gt;Use &lt;code&gt;docker&lt;/code&gt;, honestly.&lt;/p&gt;</content><id>t1_abc</id><link href="https://www.reddit.com/r/rust/comments/1xi7826/twir/abc/" /><published>2026-10-29T02:10:05+00:00</published><title>/u/helpful on x</title></entry>
</feed>"#;

#[test]
fn feed_urls_pin_the_shapes_reddit_serves() {
    assert_eq!(
        rss_listing_url(ANON_BASE, "rust", "new", 25, "day"),
        "https://www.reddit.com/r/rust/new/.rss?limit=25"
    );
    assert_eq!(
        rss_listing_url(ANON_BASE, "", "top", 10, "week"),
        "https://www.reddit.com/r/all/top/.rss?limit=10&t=week"
    );
    // An unknown sort falls back to new rather than inventing a path.
    assert!(rss_listing_url(ANON_BASE, "rust", "bogus", 5, "day").contains("/new/.rss"));

    assert_eq!(
        rss_search_url(ANON_BASE, Some("rust"), "async await", "new", 10, "day"),
        "https://www.reddit.com/r/rust/search.rss?q=async%20await&restrict_sr=1&sort=new&limit=10"
    );
    assert_eq!(
        rss_search_url(ANON_BASE, None, "pgvector", "top", 5, "month"),
        "https://www.reddit.com/search.rss?q=pgvector&sort=top&limit=5&t=month"
    );

    assert_eq!(
        rss_thread_url(ANON_BASE, "t3_1xi7826", 15),
        "https://www.reddit.com/comments/1xi7826/.rss?limit=15"
    );
    assert_eq!(
        rss_thread_url(
            ANON_BASE,
            "https://www.reddit.com/r/rust/comments/1xi7826/twir/",
            5
        ),
        "https://www.reddit.com/comments/1xi7826/.rss?limit=5"
    );
}

#[test]
fn atom_entries_become_the_same_post_rows() {
    let posts = parse_atom_posts(FEED).expect("the shipped feed shape parses");
    assert_eq!(posts.len(), 2);

    let link = &posts[0];
    assert_eq!(link.id, "1xi7826");
    assert_eq!(link.fullname(), "t3_1xi7826");
    assert_eq!(link.subreddit, "rust");
    assert_eq!(link.author, "seino_chan");
    assert_eq!(
        link.title, "This Week in Rust & #675",
        "entities decoded once"
    );
    assert!(
        link.permalink
            .starts_with("https://www.reddit.com/r/rust/comments/")
    );
    assert!(!link.is_self, "a link post has no body");
    assert_eq!(link.created_utc, 1_793_238_555.0);

    let self_post = &posts[1];
    assert_eq!(self_post.subreddit, "selfhosted");
    assert!(self_post.is_self);
    assert_eq!(
        self_post.selftext, "Booru project's submissions are closed. I don't code.",
        "html stripped, entities decoded, boilerplate dropped: {}",
        self_post.selftext
    );
    assert_eq!(self_post.title, "Is there a way <cheap> to do this?");

    // The feed carries no scores: the row must say so, not say zero.
    for p in &posts {
        assert!(!p.metrics_known);
        assert_eq!(p.score, 0);
    }
}

#[test]
fn an_empty_feed_is_an_empty_page_and_html_is_an_error() {
    assert!(
        parse_atom_posts("<feed xmlns=\"x\"></feed>")
            .unwrap()
            .is_empty()
    );
    assert!(parse_atom_comments("<feed></feed>").unwrap().is_empty());
    let err = parse_atom_posts("<html><body>blocked</body></html>").unwrap_err();
    assert!(err.contains("no Atom"), "{err}");
}

#[test]
fn comment_feed_parses_to_comment_rows() {
    let comments = parse_atom_comments(COMMENTS).expect("comment feed parses");
    assert_eq!(comments.len(), 1);
    let c = &comments[0];
    assert_eq!(c.id, "abc");
    assert_eq!(c.author, "helpful");
    assert_eq!(c.body, "Use docker, honestly.");
    assert_eq!(c.created_utc, 1_793_239_805.0);
    assert!(!c.metrics_known);
    assert!(
        c.permalink
            .as_deref()
            .unwrap_or("")
            .contains("/comments/1xi7826/")
    );
}

#[test]
fn unknown_metrics_render_as_a_question_not_a_zero() {
    let posts = parse_atom_posts(FEED).unwrap();
    let json = post_json(&posts[0], 1_790_003_000.0);
    assert!(json["score"].is_null(), "{}", json);
    assert!(json["comments"].is_null(), "{}", json);
    assert_eq!(json["metrics_known"], false);
    assert_eq!(score_label(0, false), "▲?");
    assert_eq!(score_label(0, true), "▲0");

    // The morning digest carries the flag too: a caught feed row must not
    // read back as a post nobody upvoted.
    let caught = Caught::of(&posts[0], 1_790_003_000_000);
    assert!(!caught.metrics_known);
    assert_eq!(score_label(caught.score, caught.metrics_known), "▲?");
}

#[test]
fn the_api_path_keeps_its_scores() {
    let body = r#"{"kind":"Listing","data":{"children":[{"kind":"t3","data":{"id":"a","name":"t3_a","title":"t","subreddit":"rust","score":42,"num_comments":7}}]}}"#;
    let page = parse_listing(body).expect("api listing parses");
    assert!(page.posts[0].metrics_known, "JSON carries real numbers");
    assert_eq!(page.posts[0].score, 42);
    let json = post_json(&page.posts[0], 0.0);
    assert_eq!(json["score"], 42);
    assert_eq!(json["comments"], 7);
}

#[test]
fn timestamps_parse_offline() {
    assert_eq!(parse_rfc3339_secs("1970-01-01T00:00:00+00:00"), Some(0.0));
    assert_eq!(
        parse_rfc3339_secs("2026-10-29T01:49:15+00:00"),
        Some(1_793_238_555.0)
    );
    assert_eq!(
        parse_rfc3339_secs("2026-10-31T09:02:00+00:00"),
        Some(1_793_437_320.0)
    );
    assert_eq!(
        parse_rfc3339_secs("2026-10-29T01:49:15Z"),
        parse_rfc3339_secs("2026-10-29T01:49:15+00:00")
    );
    assert_eq!(parse_rfc3339_secs("not a date"), None);
    let _ = Post::default();
}

/// The signed-out lookout polls one sub per tick, so a THROTTLED attempt has
/// to rotate the queue — otherwise the first sub is retried forever and the
/// ones behind it never get read. `last_try` is that memory, and it must
/// survive a restart.
#[test]
fn rotation_counts_throttled_attempts_and_persists() {
    let mut w = Watch::default();
    w.add_sub("selfhosted");
    w.add_sub("LocalLLaMA");
    assert_eq!(w.last_try_of("localllama"), 0.0, "never attempted = 0");

    w.tried("selfhosted", 100.0);
    let mut queue = w.subs.clone();
    queue.sort_by(|a, b| {
        w.last_try_of(a)
            .partial_cmp(&w.last_try_of(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Sub names are normalized to lowercase on the way in.
    assert_eq!(queue[0], "localllama", "the unattempted sub goes first");

    let path = std::env::temp_dir().join(format!("jerith-watch-{}.json", std::process::id()));
    save_watch(&path, &w).expect("watch file writes");
    let back = load_watch(&path);
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        back.last_try_of("selfhosted"),
        100.0,
        "rotation survives a restart"
    );
    assert_eq!(back.last_try_of("localllama"), 0.0);
}

/// The briefing reads the digest through `recent`, which never checks
/// anything — so the report of what the last tick could NOT read has to
/// live in the file, or "nothing matched" and "one sub was read" read the
/// same in the morning.
#[test]
fn the_last_tick_report_survives_a_restart() {
    let mut w = Watch::default();
    w.add_sub("selfhosted");
    assert!(w.last_check.is_none(), "no report before the first tick");
    w.last_check = Some(CheckReport {
        at_ms: 1_790_000_000_000,
        checked: vec!["selfhosted".into()],
        deferred: vec!["rust".into()],
        errors: vec!["r/godot: throttled".into()],
    });

    let path = std::env::temp_dir().join(format!("jerith-check-{}.json", std::process::id()));
    save_watch(&path, &w).expect("watch file writes");
    let back = load_watch(&path);
    let _ = std::fs::remove_file(&path);

    let lc = back.last_check.expect("report persisted");
    assert_eq!(lc.at_ms, 1_790_000_000_000);
    assert_eq!(lc.checked, vec!["selfhosted".to_string()]);
    assert_eq!(lc.deferred, vec!["rust".to_string()]);
    assert_eq!(lc.errors, vec!["r/godot: throttled".to_string()]);
}
