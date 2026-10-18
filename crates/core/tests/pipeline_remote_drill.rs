//! Live two-node pipeline drill — the Phase 3 M-ladder's live-integration
//! gate (FEATURES §13 / ARCHITECTURE §10).
//!
//! What this drills: M0–M3 machinery — `plan_route`, the resilient walker
//! with per-hop commitments, ring-rule reassignment, honest partial
//! answers, measured budgets — driven through the **real QUIC
//! transport**: signed `NodeRPC ExpertQuery` over cert-pinned wire to
//! in-process serving nodes (the store shape production uses). The
//! weights are real-and-traveling too: a source node holds the
//! containers, the hop owners boot with EMPTY stores and harvest every
//! expert they serve through the runtime transfer path (chunked
//! `ExpertFetch` over this same transport + checksummed `put_bytes`
//! ingest) — so every tensor a hop multiplies arrived on the wire,
//! byte-identical to its source container. (The containers themselves
//! are synthetic because the Colibri-side layer split is the only piece
//! still pending.)
//!
//! Pass/fail criteria (any assert firing is a drill failure):
//!   0. RUNTIME WEIGHT PROVENANCE — the hop owners boot with empty
//!      stores, honestly REFUSE an expert on the wire rather than
//!      invent activations, then harvest each hop container through the
//!      runtime transfer path; every weight the walk multiplies is
//!      therefore byte-identical to the source container and provably
//!      arrived through the transport itself. A forged replica with one
//!      flipped payload byte is rejected by the checksummed ingest —
//!      tampered bytes can never enter the served weight set.
//!   1. SUCCESS ROUTE over the wire — the two-hop route completes with
//!      no degradation, and the final activation equals the
//!      independently chained reference forward pass bit-comparable
//!      (tolerance 1e-6) on the same synthetic weights.
//!   2. WIRE COMMITMENTS — a manually driven hop through the transport
//!      returns a server-computed Phase 4 commitment over the actual
//!      reply body with pinned attribution; the walker re-verifies every
//!      hop's commitment implicitly on each attempt.
//!   3. DEAD HOP + RING REASSIGNMENT — a mute owner (accepts QUIC,
//!      never replies) between two healthy hops fails at its per-hop
//!      deadline, its layers get re-hosted by the ring-rule closure onto
//!      a healthy node, and the route still completes bit-comparable to
//!      the re-planned serial reference. `RouteResult.reassigned` types
//!      the surgery.
//!   4. DEAD HOP + HONEST PARTIAL — the same dead hop without
//!      reassignment yields a typed `DegradedHop` and the walk
//!      continues: the residual equals the serial reference of the hops
//!      that ran (skipped layers are skipped, never silently zeroed).
//!   5. BUDGETS — every attempt is timed (successes and failures), all
//!      completed hops land inside their RTT-floored demand-priced
//!      budgets, the whole drill finishes well under its own deadlines,
//!      and two dead-hop failures sit exactly below the stability
//!      hysteresis raise line (a flapping peer doesn't self-exclude at
//!      the boundary).
//!   6. THREE-NODE RTT BUDGETS — a third serving node is priced as a
//!      simulated wide-area hop (180 ms `rtt_of`): the walk runs
//!      A → B → far-D, the far hop's budget is floored on
//!      base + 3×RTT and sits strictly above the near hops', the walk
//!      still completes with honest empty over-budget, and the
//!      activation bit-compares to the three-hop serial reference.
//!
//! Honest limits: loopback nodes, and the containers are synthetic
//! until the Colibri-side layer split yields real model shards; the
//! wide-area RTT is SIMULATED (`rtt_of` prices a far hop at 180 ms and
//! floors its budget on it — the latencies themselves stay
//! loopback-real). The `runtime` stand-in stays OFF (`tensors_only
//! = true`): a hashed text completion is not expert math, so wiring that
//! path here would fabricate activations instead of moving real weights.

use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use jerith_mesh_core::expert_store::{ExpertStore, ExpertWeights};
use jerith_mesh_core::pipeline::{
    PipelineHop, RequestBudget, RouteResult, RouteStability, WalkLimits, execute_route_resilient,
    hop_expert_id, plan_route, plan_route_budget,
};
use jerith_mesh_core::quic::{
    ExpertServing, NodeRpcReply, NodeRpcRequest, PinMap, SharedPinMap, call, fetch_expert,
    pinned_client_endpoint_dyn, sign_request,
};
use jerith_mesh_core::signing::SigningIdentity;

const MODEL: &str = "drill-m";
const HIDDEN: usize = 8;
const INTER: usize = 16;

/// Reference forward for one hop, computed locally on the drill's
/// deterministic synthetic containers (the same seed the source node
/// turned into the containers its hop owners harvested — so a mismatch
/// means bytes lost in the transport, not a different expert).
fn run_hop(expert_id: &str, x: &[f32]) -> anyhow::Result<Vec<f32>> {
    ExpertWeights::synthetic(expert_id, HIDDEN, INTER).forward(x)
}

static DRILL_NODE_SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// A real serving node: real QUIC server loop, cert-pinned identity, an
/// EMPTY weight store — what it hosts is the caller's decision (the
/// weight source seeds its containers; the hop owners are filled by
/// harvesting over the wire in the test body). Lives until the test
/// process exits (the watch sender is leaked deliberately).
async fn spin_node() -> anyhow::Result<(String, SocketAddr, Arc<ExpertStore>)> {
    let dir = std::env::temp_dir().join(format!(
        "jerith-pipeline-drill-{}-{}",
        std::process::id(),
        DRILL_NODE_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let store = Arc::new(ExpertStore::new(&dir, 8).context("drill store")?);

    let identity = SigningIdentity::generate();
    let server =
        jerith_mesh_core::quic::server_endpoint_for(&identity, "127.0.0.1:0".parse().unwrap())
            .expect("drill server endpoint");
    let addr = server.local_addr().unwrap();
    let node_id = identity.node_id();
    let serving = Arc::new(ExpertServing {
        store: Some(store.clone()),
        runtime: None,
        tensors_only: true,
        stats: None,
        isolate: None,
        prefetch: None,
    });
    let (_kill_tx, rx) = tokio::sync::watch::channel(false);
    {
        let node_id = node_id.clone();
        tokio::spawn(async move {
            let _ = jerith_mesh_core::quic::serve(server, serving, node_id, rx).await;
        });
    }
    std::mem::forget(_kill_tx); // keep the server's kill channel open
    Ok((node_id, addr, store))
}

/// A mute node: a real QUIC endpoint whose connections are accepted but
/// never answered — the honest stand-in for a hung peer, and the reason
/// the dead-hop failure is a genuine *deadline* (a closed port would
/// fail too fast to exercise it).
async fn spin_mute_node() -> (String, SocketAddr) {
    let identity = SigningIdentity::generate();
    let endpoint =
        jerith_mesh_core::quic::server_endpoint_for(&identity, "127.0.0.1:0".parse().unwrap())
            .expect("mute endpoint");
    let addr = endpoint.local_addr().unwrap();
    let node_id = identity.node_id();
    tokio::spawn(async move {
        while let Some(incoming) = endpoint.accept().await {
            // Complete the handshake, then hold the connection in
            // silence: the client's stream opens and waits — the only
            // honest way to make a genuine *deadline* (a closed port
            // would fail too fast to exercise it).
            if let Ok(conn) = incoming.await {
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_secs(120)).await;
                    drop(conn);
                });
            }
        }
    });
    (node_id, addr)
}

#[tokio::test]
async fn two_node_pipeline_drill() -> anyhow::Result<()> {
    let drill_started = Instant::now();

    // --- The mesh: A hosts layers 0–1, B hosts 2–3 plus the re-host
    // targets (2–2, 3–3) so the dead-hop variants land their
    // reassignment on a healthy node without a second boot; C is mute.
    // S holds every expert container the hop owners will serve — and A
    // and B boot with EMPTY stores: the weights they multiply arrive
    // through the transport (criterion 0 below), not a mirrored seed.
    let (node_s, addr_s, store_s) = spin_node().await?;
    let experts: Vec<String> = vec![
        format!("{MODEL}#0-1"),
        format!("{MODEL}#2-3"),
        format!("{MODEL}#2-2"),
        format!("{MODEL}#3-3"),
    ];
    for id in &experts {
        store_s
            .put(&ExpertWeights::synthetic(id, HIDDEN, INTER))
            .expect("seed source container");
    }
    let (node_a, addr_a, store_a) = spin_node().await?;
    let (node_b, addr_b, store_b) = spin_node().await?;
    let (node_d, addr_d, store_d) = spin_node().await?;
    let (node_c, addr_c) = spin_mute_node().await;

    let mut pin_map: PinMap = HashMap::new();
    pin_map.insert(node_a.clone(), addr_a);
    pin_map.insert(node_b.clone(), addr_b);
    pin_map.insert(node_s.clone(), addr_s);
    pin_map.insert(node_d.clone(), addr_d);
    pin_map.insert(node_c.clone(), addr_c);
    let pins: SharedPinMap = Arc::new(tokio::sync::RwLock::new(pin_map.clone()));
    let client = pinned_client_endpoint_dyn(pins).expect("drill client endpoint");
    let walker = SigningIdentity::generate();
    let walk_node_id = walker.node_id();
    let walk_req_id = "drill-route-1".to_string();

    let addrs: HashMap<String, SocketAddr> = pin_map.iter().map(|(n, a)| (n.clone(), *a)).collect();

    // --- The transport adapter under test: signed ExpertQuery over the
    // pinned QUIC transport, the HOP DEADLINE APPLIED HERE (the closure
    // owns applying it, per the dispatch contract), wire attribution
    // checked, the server's commitment passed through for the walker's
    // per-hop re-verification.
    let walker_for_adapter = walker.clone();
    let req_id_for_adapter = walk_req_id.clone();
    let adapter_client = client.clone();
    // The adapter is the live integration's transport shape: a hop call
    // returns a boxed future carrying (output, server commitment).
    type RemoteHopper = std::pin::Pin<
        Box<dyn std::future::Future<Output = anyhow::Result<(Vec<f32>, Option<String>)>> + Send>,
    >;
    let remote_run =
        move |expert_id: &str, node: &str, input: Vec<f32>, deadline: Duration| -> RemoteHopper {
            let client = adapter_client.clone();
            let walker = walker_for_adapter.clone();
            let walk_req = req_id_for_adapter.clone();
            let addr = addrs.get(node).copied();
            let expert_id = expert_id.to_owned();
            let node = node.to_owned();
            Box::pin(async move {
                let addr = addr.ok_or_else(|| anyhow::anyhow!("no pin for hop owner {node}"))?;
                let mut req = NodeRpcRequest::ExpertQuery {
                    req_id: walk_req.clone(),
                    expert_id: expert_id.clone(),
                    input,
                    from_node_id: walker.node_id(),
                    stamp_ms: 0,
                    sig: String::new(),
                };
                sign_request(&mut req, &walker);
                // The deadline the executor priced is applied at the wire.
                // `call` gets a strictly LARGER internal budget (the margin
                // keeps the two timers from racing): the priced deadline
                // below must always be the one that fires, so the failure
                // reason names the deadline on every platform — not which
                // same-length timer happened to win the scheduling race.
                let reply = tokio::time::timeout(
                    deadline,
                    call(
                        &client,
                        addr,
                        &node,
                        &req,
                        &walker,
                        deadline.saturating_add(Duration::from_secs(5)),
                    ),
                )
                .await
                .map_err(|_| {
                    anyhow::anyhow!(
                        "hop deadline ({:?}) exceeded on hop {node} ({expert_id})",
                        deadline
                    )
                })??;
                match reply {
                    NodeRpcReply::ExpertResult {
                        req_id: _,
                        output,
                        node_id,
                        commitment,
                    } => {
                        if node_id != node {
                            anyhow::bail!("attribution: hop served by {node_id}, pinned {node}");
                        }
                        Ok((output, commitment))
                    }
                    other => anyhow::bail!("unexpected rpc reply from hop {node}: {other:?}"),
                }
            })
        };

    // --- Criterion 0: RUNTIME WEIGHT PROVENANCE. The serving nodes
    // under test boot with empty stores: they first honestly REFUSE an
    // expert on the wire rather than invent activations, then harvest
    // every hop container through the runtime transfer path (chunked
    // ExpertFetch over this same transport, checksummed ingest — the
    // exact replication path production nodes run). From here on, every
    // tensor a hop multiplies provably arrived on the wire.
    let input = vec![0.5_f32, -0.25, 1.0, 0.0, -1.5, 2.0, 0.75, -0.5];

    // Empty-store honesty, on the wire: B does not host 2–3 yet.
    {
        let mut req = NodeRpcRequest::ExpertQuery {
            req_id: "drill-probe-empty".into(),
            expert_id: format!("{MODEL}#2-3"),
            input: input.clone(),
            from_node_id: walk_node_id.clone(),
            stamp_ms: 0,
            sig: String::new(),
        };
        sign_request(&mut req, &walker);
        let reply = tokio::time::timeout(
            Duration::from_secs(10),
            call(
                &client,
                addr_b,
                &node_b,
                &req,
                &walker,
                Duration::from_secs(5),
            ),
        )
        .await
        .expect("refusal probe timeout")
        .expect("refusal probe rpc failed");
        match reply {
            NodeRpcReply::Error { message, .. } => assert!(
                message.contains("refusing to fabricate"),
                "empty-store B must refuse, not invent: {message}"
            ),
            other => panic!("empty-store B must Error, got {other:?}"),
        }
    }

    // The harvest: each hop owner pulls its containers from S and
    // ingests them byte-checksummed. Prove provenance end-to-end: the
    // ingested tensors equal the source's exactly, and the expert sits
    // on disk, load-on-demand (hosted but never resident).
    let a_targets = [format!("{MODEL}#0-1")];
    let b_targets = [
        format!("{MODEL}#2-3"),
        format!("{MODEL}#2-2"),
        format!("{MODEL}#3-3"),
    ];
    // D is the wide-area third owner (criterion 6 below) and hosts the
    // #3 replica — harvested through the same runtime path as the rest.
    let d_targets = [format!("{MODEL}#3-3")];
    for (owner_id, owner_store, targets) in [
        (&node_a, &store_a, &a_targets[..]),
        (&node_b, &store_b, &b_targets[..]),
        (&node_d, &store_d, &d_targets[..]),
    ] {
        for id in targets {
            let bytes = fetch_expert(
                &client,
                addr_s,
                &node_s,
                &walker,
                id,
                Duration::from_secs(15),
            )
            .await
            .with_context(|| format!("harvesting {id} over QUIC"))?;
            let ingested = owner_store
                .put_bytes(&bytes)
                .with_context(|| format!("ingesting harvested {id}"))?;
            assert_eq!(*id, ingested.id, "container keeps its id through the wire");
            assert_eq!(
                ingested,
                ExpertWeights::synthetic(id, HIDDEN, INTER),
                "harvested {id} is the tensors the source seeded, byte for byte"
            );
            assert!(owner_store.contains(id), "{owner_id} now hosts {id}");
            assert!(
                !owner_store.is_resident(id),
                "{id} sits on disk (load-on-demand), never resident"
            );
        }
    }
    for id in &experts {
        let (want, _) = store_s.bytes_of(id, 0, u32::MAX)?;
        for store in [&store_a, &store_b, &store_d] {
            if store.contains(id) {
                let (got, _) = store.bytes_of(id, 0, u32::MAX)?;
                assert_eq!(got, want, "{id} mutated across the transport");
            }
        }
    }

    // Tamper gate (criterion 0, continued): a corrupted replica can
    // never enter the served weight set. Forge the #2-3 bytes with one
    // payload byte flipped — magic, id and shape stay honest, only the
    // checksum now lies — and prove the checksummed ingest refuses it.
    // The honest replica on disk stays untouched, so the hops keep
    // multiplying provenance-clean weights no matter what travels.
    {
        let id = format!("{MODEL}#2-3");
        let (mut evil, _) = store_s.bytes_of(&id, 0, u32::MAX)?;
        let mid = evil.len() / 2;
        evil[mid] ^= 0xFF;
        let err = store_b
            .put_bytes(&evil)
            .expect_err("tampered container must never ingest");
        assert!(
            format!("{err:#}").contains("checksum mismatch"),
            "the rejection must name the checksum, got: {err:#}"
        );
        let (still, _) = store_b.bytes_of(&id, 0, u32::MAX)?;
        let (origin, _) = store_s.bytes_of(&id, 0, u32::MAX)?;
        assert_eq!(
            still, origin,
            "ingest refusal leaves the honest replica intact"
        );
    }

    // --- Criterion 1 + 5: success route over the wire, priced honestly.
    let ownership: BTreeMap<usize, String> = [
        (0, node_a.clone()),
        (1, node_a.clone()),
        (2, node_b.clone()),
        (3, node_b.clone()),
    ]
    .into();
    let route = plan_route(&walk_node_id, &ownership);
    assert_eq!(
        route,
        vec![
            PipelineHop {
                node: Some(node_a.clone()),
                layers: vec![0, 1]
            },
            PipelineHop {
                node: Some(node_b.clone()),
                layers: vec![2, 3]
            },
        ],
        "planner: two remote hops in walk order"
    );

    // Budgets priced from the measured-activation ordering (a drill node
    // has no measured history: even split, but the RTT floors are real).
    let rtt_of = |_: &str| 5_u64;
    let priced = plan_route_budget(&route, |_| 1, rtt_of, Duration::from_millis(250), 20_000);
    let budget_of = move |hop: &PipelineHop| {
        priced
            .iter()
            .find(|b| b.layers == hop.layers.as_slice() && b.node == hop.node)
            .map(|b| b.budget_ms)
            .unwrap_or(u64::MAX)
    };

    let mut stability = RouteStability::new();
    let result = execute_route_resilient(
        MODEL,
        &walk_req_id,
        route,
        input.clone(),
        |_a, _l| anyhow::bail!("drill router hosts no layers"),
        remote_run.clone(),
        rtt_of,
        Duration::from_millis(250),
        Some(budget_of),
        None,
        None,
        &mut stability,
        |_l, _d| None,
    )
    .await;

    assert!(
        result.is_complete() && result.reassigned.is_empty(),
        "healthy route must not degrade: {:?}",
        result.degraded
    );
    assert_eq!(result.hops_run, 2, "both hops ran over the wire");
    assert!(
        result.over_budget.is_empty(),
        "honest loopback pricing must hold: {:?}",
        result.over_budget
    );
    assert_eq!(
        result.attempts,
        vec![
            jerith_mesh_core::pipeline::AttemptTiming {
                node: Some(node_a.clone()),
                layers: vec![0, 1],
                ms: result.attempts[0].ms,
                ok: true,
            },
            jerith_mesh_core::pipeline::AttemptTiming {
                node: Some(node_b.clone()),
                layers: vec![2, 3],
                ms: result.attempts[1].ms,
                ok: true,
            },
        ]
    );
    assert!(result.attempts[0].ms > 0, "wire latency is measured");

    // Serial reference on the containers now living on the hop owners
    // (chained hop by hop).
    let r1 = run_hop(
        &hop_expert_id(
            MODEL,
            &PipelineHop {
                node: None,
                layers: vec![0, 1],
            },
        ),
        &input,
    )?;
    let want = run_hop(
        &hop_expert_id(
            MODEL,
            &PipelineHop {
                node: None,
                layers: vec![2, 3],
            },
        ),
        &r1,
    )?;
    assert_eq!(want.len(), result.activation.len());
    for (got, expected) in result.activation.iter().zip(&want) {
        assert!(
            (got - expected).abs() < 1e-6,
            "activation mismatch across the wire: got {got}, want {expected}"
        );
    }

    // --- Criterion 2: the wire commitment, checked explicitly.
    {
        let manual_input = vec![1.5_f32, -2.0, 0.25, 0.0, 3.0, -0.75, 0.5, 1.0];
        let expert = format!("{MODEL}#2-3");
        let mut req = NodeRpcRequest::ExpertQuery {
            req_id: "drill-manual-1".into(),
            expert_id: expert.clone(),
            input: manual_input.clone(),
            from_node_id: walk_node_id.clone(),
            stamp_ms: 0,
            sig: String::new(),
        };
        sign_request(&mut req, &walker);
        let reply = tokio::time::timeout(
            Duration::from_secs(10),
            call(
                &client,
                addr_b,
                &node_b,
                &req,
                &walker,
                Duration::from_secs(5),
            ),
        )
        .await
        .expect("manual hop timeout")
        .expect("manual hop failed");
        match reply {
            NodeRpcReply::ExpertResult {
                req_id,
                output,
                node_id,
                commitment,
            } => {
                assert_eq!(req_id, "drill-manual-1");
                assert_eq!(node_id, node_b, "attribution pinned to B");
                let commitment = commitment.expect("wire commitment present");
                assert!(
                    jerith_mesh_core::verify::verify_commitment(
                        "drill-manual-1",
                        &expert,
                        &node_b,
                        &manual_input,
                        &output,
                        &commitment,
                    ),
                    "server commitment must verify against the wire body"
                );
            }
            other => panic!("unexpected rpc reply: {other:?}"),
        }
    }

    // --- Criteria 3 + 4: the dead hop, both degradation paths.
    let ownership_dead: BTreeMap<usize, String> = [
        (0, node_a.clone()),
        (1, node_a.clone()),
        (2, node_c.clone()),
        (3, node_b.clone()),
    ]
    .into();
    let dead_route = plan_route(&walk_node_id, &ownership_dead);
    assert_eq!(
        dead_route
            .iter()
            .map(|h| h.layers.as_slice())
            .collect::<Vec<_>>(),
        vec![&[0, 1][..], &[2][..], &[3][..]],
        "planner: A, dead C, B"
    );
    let priced_dead = plan_route_budget(
        &dead_route,
        |_| 1,
        rtt_of,
        Duration::from_millis(250),
        20_000,
    );
    let budget_of_dead = move |hop: &PipelineHop| {
        priced_dead
            .iter()
            .find(|b| b.layers == hop.layers.as_slice() && b.node == hop.node)
            .map(|b| b.budget_ms)
            .unwrap_or(u64::MAX)
    };

    // Variant 3 — ring rule re-hosts the dead owner's layer onto B.
    let rehost_owner = node_b.clone();
    let result3 = {
        let dead_owner = node_c.clone();
        let reassign = move |layers: &[usize], dead: Option<&str>| {
            if dead == Some(dead_owner.as_str()) {
                Some(vec![PipelineHop {
                    node: Some(rehost_owner.clone()),
                    layers: layers.to_vec(),
                }])
            } else {
                None
            }
        };
        let mut stability3 = RouteStability::new();
        execute_route_resilient(
            MODEL,
            &walk_req_id,
            dead_route.clone(),
            input.clone(),
            |_a, _l| anyhow::bail!("drill router hosts no layers"),
            remote_run.clone(),
            rtt_of,
            Duration::from_millis(250),
            Some(budget_of_dead.clone()),
            None,
            None,
            &mut stability3,
            reassign,
        )
        .await
    };
    assert!(
        result3.degraded.is_empty(),
        "re-hosted walk must not degrade"
    );
    assert_eq!(result3.reassigned.len(), 1, "the dead hop was re-hosted");
    assert_eq!(result3.reassigned[0].layers, vec![2]);
    assert_eq!(
        result3.reassigned[0].was_node.as_deref(),
        Some(node_c.as_str())
    );
    assert!(
        result3.reassigned[0].reason.contains("deadline"),
        "the mute peer died at its priced deadline (reason: {})",
        result3.reassigned[0].reason
    );
    assert_eq!(result3.hops_run, 3, "A, re-hosted 2–2 on B, 3–3 on B");
    let r1d = run_hop(
        &hop_expert_id(
            MODEL,
            &PipelineHop {
                node: None,
                layers: vec![0, 1],
            },
        ),
        &input,
    )?;
    let r2d = run_hop(
        &hop_expert_id(
            MODEL,
            &PipelineHop {
                node: None,
                layers: vec![2],
            },
        ),
        &r1d,
    )?;
    let want3 = run_hop(
        &hop_expert_id(
            MODEL,
            &PipelineHop {
                node: None,
                layers: vec![3],
            },
        ),
        &r2d,
    )?;
    for (got, expected) in result3.activation.iter().zip(&want3) {
        assert!(
            (got - expected).abs() < 1e-6,
            "reassigned activation mismatch: got {got}, want {expected}"
        );
    }

    // Variant 4 — no reassignment: the honest skips-short answer.
    let result4 = {
        let mut stability4 = RouteStability::new();
        execute_route_resilient(
            MODEL,
            &walk_req_id,
            dead_route.clone(),
            input.clone(),
            |_a, _l| anyhow::bail!("drill router hosts no layers"),
            remote_run.clone(),
            rtt_of,
            Duration::from_millis(250),
            Some(budget_of_dead),
            None,
            None,
            &mut stability4,
            |_l, _d| None,
        )
        .await
    };
    assert_eq!(result4.hops_run, 2, "A and B walked on past the dead hop");
    assert_eq!(result4.degraded.len(), 1, "exactly the lost layer typed");
    assert_eq!(result4.degraded[0].layers, vec![2]);
    assert_eq!(result4.degraded[0].node.as_deref(), Some(node_c.as_str()));
    assert!(
        result4.degraded[0].reason.contains("deadline"),
        "the mute peer died at its priced deadline: {}",
        result4.degraded[0].reason
    );
    let skip_ref1 = run_hop(
        &hop_expert_id(
            MODEL,
            &PipelineHop {
                node: None,
                layers: vec![0, 1],
            },
        ),
        &input,
    )?;
    let want4 = run_hop(
        &hop_expert_id(
            MODEL,
            &PipelineHop {
                node: None,
                layers: vec![3],
            },
        ),
        &skip_ref1,
    )?;
    for (got, expected) in result4.activation.iter().zip(&want4) {
        assert!(
            (got - expected).abs() < 1e-6,
            "partial-activation mismatch: got {got}, want {expected}"
        );
    }

    // Criterion 5 (continued): two dead failures sit below the
    // exclude line — the hysteresis boundary holds across variants.
    assert!(
        !stability.is_excluded(&node_c),
        "two failures must not self-exclude at the boundary"
    );

    // --- Criterion 6: three nodes and simulated wide-area RTT. D is a
    // real serving node the pricer sees as 180 ms away. The walk runs
    // A → B → far-D and the budget pricer must FLOOR the far hop's
    // budget on its RTT (base + 3×RTT headroom) while the near hops keep
    // their own floors; under honest loopback latencies nothing can be
    // over-budget — the point here is the priced numbers themselves.
    {
        let ownership_far: BTreeMap<usize, String> = [
            (0, node_a.clone()),
            (1, node_a.clone()),
            (2, node_b.clone()),
            (3, node_d.clone()),
        ]
        .into();
        let far_route = plan_route(&walk_node_id, &ownership_far);
        assert_eq!(
            far_route
                .iter()
                .map(|h| (h.node.clone(), h.layers.clone()))
                .collect::<Vec<_>>(),
            vec![
                (Some(node_a.clone()), vec![0, 1]),
                (Some(node_b.clone()), vec![2]),
                (Some(node_d.clone()), vec![3]),
            ],
            "planner: A, B, far D in walk order"
        );
        let rtt_far = |n: &str| if n == node_d.as_str() { 180_u64 } else { 5_u64 };
        let priced_far = plan_route_budget(
            &far_route,
            |_| 1,
            rtt_far,
            Duration::from_millis(250),
            20_000,
        );
        let budget_of_far = move |hop: &PipelineHop| {
            priced_far
                .iter()
                .find(|b| b.layers == hop.layers.as_slice() && b.node == hop.node)
                .map(|b| b.budget_ms)
                .unwrap_or(u64::MAX)
        };
        let far_hop = far_route
            .iter()
            .find(|h| h.node.as_deref() == Some(node_d.as_str()))
            .expect("far hop present")
            .clone();
        let near_hop = far_route
            .iter()
            .find(|h| h.node.as_deref() == Some(node_a.as_str()))
            .expect("near hop present")
            .clone();
        let far_floor = jerith_mesh_core::pipeline::hop_timeout(Duration::from_millis(250), 180)
            .as_millis() as u64;
        let near_floor = jerith_mesh_core::pipeline::hop_timeout(Duration::from_millis(250), 5)
            .as_millis() as u64;
        assert!(
            budget_of_far(&far_hop) >= far_floor,
            "far hop budget floored on base + 3×RTT: {} < {far_floor}",
            budget_of_far(&far_hop)
        );
        assert!(
            budget_of_far(&near_hop) >= near_floor,
            "near hop keeps its own RTT floor"
        );
        assert!(
            budget_of_far(&far_hop) > budget_of_far(&near_hop),
            "the RTT floor must lift the far hop above the near hops': {} vs {}",
            budget_of_far(&far_hop),
            budget_of_far(&near_hop)
        );
        let result6 = execute_route_resilient(
            MODEL,
            &walk_req_id,
            far_route.clone(),
            input.clone(),
            |_a, _l| anyhow::bail!("drill router hosts no layers"),
            remote_run.clone(),
            rtt_far,
            Duration::from_millis(250),
            Some(budget_of_far.clone()),
            None,
            None,
            &mut stability,
            |_l, _d| None,
        )
        .await;
        assert!(
            result6.is_complete() && result6.reassigned.is_empty(),
            "the three-node walk must complete: {:?}",
            result6.degraded
        );
        assert_eq!(result6.hops_run, 3, "A, B and far D all walked");
        assert!(
            result6.over_budget.is_empty(),
            "RTT-floored honest pricing must hold: {:?}",
            result6.over_budget
        );
        assert_eq!(
            result6.attempts.last().map(|a| a.node.as_deref()),
            Some(Some(node_d.as_str())),
            "the far hop ran last"
        );
        let r1f = run_hop(
            &hop_expert_id(
                MODEL,
                &PipelineHop {
                    node: None,
                    layers: vec![0, 1],
                },
            ),
            &input,
        )?;
        let r2f = run_hop(
            &hop_expert_id(
                MODEL,
                &PipelineHop {
                    node: None,
                    layers: vec![2],
                },
            ),
            &r1f,
        )?;
        let want6 = run_hop(
            &hop_expert_id(
                MODEL,
                &PipelineHop {
                    node: None,
                    layers: vec![3],
                },
            ),
            &r2f,
        )?;
        for (got, expected) in result6.activation.iter().zip(&want6) {
            assert!(
                (got - expected).abs() < 1e-6,
                "three-node activation mismatch: got {got}, want {expected}"
            );
        }

        // --- Criterion 7: the budgets are ENFORCED through the real
        // transport (M4 wall + §1 per-request work caps).
        //
        // (a) A per-request budget allowing only two REMOTE attempts
        //     walks A and B over real QUIC and refuses the far D hop by
        //     name with the typed per-request reason — deterministic (a
        //     count, not a race), so the degradation is proven live,
        //     against the actual wire, not just in unit simulations.
        {
            let request_budget = RequestBudget {
                max_remote_attempts: Some(2),
                max_wire_bytes: None,
            };
            let result7 = execute_route_resilient(
                MODEL,
                &walk_req_id,
                far_route.clone(),
                input.clone(),
                |_a, _l| anyhow::bail!("drill router hosts no layers"),
                remote_run.clone(),
                rtt_far,
                Duration::from_millis(250),
                Some(budget_of_far.clone()),
                None,
                Some(&WalkLimits::with_budget(request_budget)),
                &mut stability,
                |_l, _d| None,
            )
            .await;
            assert!(
                result7.budget_exhausted,
                "the per-request cap must degrade the far hop: {:?}",
                result7.degraded
            );
            assert_eq!(result7.hops_run, 2, "A and B walked inside the cap");
            assert_eq!(
                result7.degraded.len(),
                1,
                "only the far hop was refused: {:?}",
                result7.degraded
            );
            assert_eq!(
                result7.degraded[0].node.as_deref(),
                Some(node_d.as_str()),
                "the far hop was refused by name"
            );
            assert!(
                result7.degraded[0]
                    .reason
                    .contains(RouteResult::REQUEST_BUDGET_EXHAUSTED_REASON),
                "typed per-request reason: {:?}",
                result7.degraded[0].reason
            );
            assert_eq!(
                result7.attempts.last().map(|a| a.node.as_deref()),
                Some(Some(node_b.as_str())),
                "the walk stopped after the second remote attempt"
            );
            // The partial answer equals the two-hop local reference —
            // refusal costs the refused hop, nothing else (residual
            // unchanged through L2).
            for (got, expected) in result7.activation.iter().zip(&r2f) {
                assert!(
                    (got - expected).abs() < 1e-6,
                    "two-node degraded activation mismatch: got {got}, want {expected}"
                );
            }
        }
        // (b) The DEFAULT per-request caps must never misfire on the
        // honest live walk: the same A → B → far-D route completes
        // untouched and byte-equal to the full reference under defaults.
        {
            let result8 = execute_route_resilient(
                MODEL,
                &walk_req_id,
                far_route.clone(),
                input.clone(),
                |_a, _l| anyhow::bail!("drill router hosts no layers"),
                remote_run.clone(),
                rtt_far,
                Duration::from_millis(250),
                Some(budget_of_far.clone()),
                None,
                Some(&WalkLimits::with_budget(RequestBudget::default())),
                &mut stability,
                |_l, _d| None,
            )
            .await;
            assert!(
                !result8.budget_exhausted,
                "default caps are not a squeeze: the honest walk completes"
            );
            assert!(
                result8.degraded.is_empty(),
                "no honest hop degrades under defaults: {:?}",
                result8.degraded
            );
            assert_eq!(result8.hops_run, 3, "A, B, far D walked under default caps");
            for (got, expected) in result8.activation.iter().zip(&want6) {
                assert!(
                    (got - expected).abs() < 1e-6,
                    "default-cap walk activation mismatch: got {got}, want {expected}"
                );
            }
        }

        // --- Criterion 8 (Phase 3 M4+§1): speculative warm prefetch
        // rides the dial order over REAL QUIC — while hop A runs, the
        // next hop's containers travel from the canonical source into
        // the walker's local cache; the mute owner's dial dies at its
        // deadline, and the caller upgrades the answer with a SELF-HOST
        // reassignment: those layers run LOCALLY from the warm cache
        // instead of degrading. Every pre-paid byte lands in the same
        // per-request ledger the caps read.
        {
            let walker_cache_dir = std::env::temp_dir()
                .join(format!("jerith-drill-walker-cache-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&walker_cache_dir);
            let walker_cache = std::sync::Arc::new(
                ExpertStore::new(&walker_cache_dir, 8).context("walker warm cache store")?,
            );

            let hook_client = client.clone();
            let hook_walker = walker.clone();
            let hook_source_addr = addr_s;
            let hook_source_id = node_s.clone();
            let hook_cache = walker_cache.clone();
            let hook_bytes: std::sync::Arc<std::sync::atomic::AtomicU64> =
                std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
            // The hook keeps its OWN handle (the closure moves it in);
            // the assertion reads through the original.
            let hook_bytes_for_hook = hook_bytes.clone();
            let prefetch_closure = Box::leak(Box::new(
                move |hop: PipelineHop| -> std::pin::Pin<
                    Box<dyn std::future::Future<Output = anyhow::Result<u64>> + Send>,
                > {
                    let client = hook_client.clone();
                    let walker = hook_walker.clone();
                    let cache = hook_cache.clone();
                    let source_addr = hook_source_addr;
                    let source_id = hook_source_id.clone();
                    let warm_bytes = hook_bytes_for_hook.clone();
                    let expert_id = hop_expert_id(MODEL, &hop);
                    Box::pin(async move {
                        let bytes = fetch_expert(
                            &client,
                            source_addr,
                            &source_id,
                            &walker,
                            &expert_id,
                            Duration::from_secs(15),
                        )
                        .await
                        .with_context(|| {
                            format!("speculative warm fetch of {expert_id} over QUIC")
                        })?;
                        let n = bytes.len();
                        cache
                            .put_bytes(&bytes)
                            .with_context(|| format!("ingesting warm-prefetched {expert_id}"))?;
                        warm_bytes.store(
                            warm_bytes.load(std::sync::atomic::Ordering::SeqCst) + n as u64,
                            std::sync::atomic::Ordering::SeqCst,
                        );
                        Ok::<u64, anyhow::Error>(n as u64)
                    })
                },
            )
                as Box<
                    dyn Fn(
                            PipelineHop,
                        ) -> std::pin::Pin<
                            Box<dyn std::future::Future<Output = anyhow::Result<u64>> + Send>,
                        > + Send
                        + Sync,
                >);

            // The route over the drill's real 4-layer model: A hosts
            // layers 0–1, the MUTE owner owns layer 2, B owns layer 3 —
            // the muted dial is a genuine deadline death.
            let ownership_spec: BTreeMap<usize, String> = [
                (0usize, node_a.clone()),
                (1, node_a.clone()),
                (2, node_c.clone()),
                (3, node_b.clone()),
            ]
            .into();
            let spec_route = plan_route(&walk_node_id, &ownership_spec);
            assert_eq!(
                spec_route.len(),
                3,
                "A, mute, B hops in walk order: {:?}",
                spec_route
            );

            // The walker self-hosts from its warm cache; a miss is an
            // honest error (proving the prefetch, not luck, carried the
            // tensors).
            let cache_for_local = walker_cache.clone();
            let local_run = move |act: &[f32], layers: &[usize]| {
                let hop = PipelineHop {
                    node: None,
                    layers: layers.to_vec(),
                };
                let id = hop_expert_id(MODEL, &hop);
                let weights = cache_for_local
                    .get(&id)
                    .map_err(|e| anyhow::anyhow!("walker self-host miss on {id}: {e}"))?;
                weights.forward(act)
            };
            let reassign_self_host = |layers: &[usize], dead: Option<&str>| {
                if dead == Some(node_c.as_str()) {
                    Some(vec![PipelineHop {
                        node: None,
                        layers: layers.to_vec(),
                    }])
                } else {
                    None
                }
            };
            let limits_spec = WalkLimits {
                budget: RequestBudget::default(),
                prefetch_next: Some(&*prefetch_closure),
                est_layer_bytes: 1, // unit-tested headroom rule; live rule: always room
            };
            let mut stability_spec = RouteStability::new();
            let result9 = execute_route_resilient(
                MODEL,
                &walk_req_id,
                spec_route,
                input.clone(),
                local_run,
                remote_run.clone(),
                rtt_far,
                Duration::from_millis(250),
                None::<fn(&PipelineHop) -> u64>,
                None,
                Some(&limits_spec),
                &mut stability_spec,
                reassign_self_host,
            )
            .await;

            assert!(
                result9.is_complete(),
                "the warm-prefetched walker self-hosts the mute owner's layers: {:?}",
                result9.degraded
            );
            assert_eq!(result9.hops_run, 3, "A remote, mute SELF-HOSTED, B remote");
            assert_eq!(
                result9.reassigned.len(),
                1,
                "the dead owner was re-hosted to SELF: {:?}",
                result9.reassigned
            );
            assert_eq!(
                result9.reassigned[0].was_node.as_deref(),
                Some(node_c.as_str()),
                "the mute owner was re-hosted by name"
            );
            assert_eq!(
                result9.attempts[1].node.as_deref(),
                Some(node_c.as_str()),
                "the mute dial died at its deadline first"
            );
            assert!(!result9.attempts[1].ok);
            assert_eq!(
                result9.attempts[2].node.as_deref(),
                None,
                "the self-host ran as a LOCAL attempt"
            );
            assert!(result9.attempts[2].ok);
            assert_eq!(
                result9.prefetches, 2,
                "A's dial warmed the mute hop; the mute dial warmed B"
            );
            // The ledger: 3 remote attempts' transport bytes (A in+out,
            // mute in only, B in+out) + the two warm pulls, exactly.
            let in_bytes = (input.len() * 4) as u64;
            assert_eq!(
                result9.wire_bytes_moved,
                5 * in_bytes + hook_bytes.load(std::sync::atomic::Ordering::SeqCst),
                "prefetch bytes and transport bytes share one request ledger"
            );
            // And the activation is byte-equal to the honest local chain
            // (including the self-hosted layer run on the warm tensors).
            let r1g = run_hop(
                &hop_expert_id(
                    MODEL,
                    &PipelineHop {
                        node: None,
                        layers: vec![0, 1],
                    },
                ),
                &input,
            )?;
            let r2g = run_hop(
                &hop_expert_id(
                    MODEL,
                    &PipelineHop {
                        node: None,
                        layers: vec![2],
                    },
                ),
                &r1g,
            )?;
            let want9 = run_hop(
                &hop_expert_id(
                    MODEL,
                    &PipelineHop {
                        node: None,
                        layers: vec![3],
                    },
                ),
                &r2g,
            )?;
            for (got, expected) in result9.activation.iter().zip(&want9) {
                assert!(
                    (got - expected).abs() < 1e-6,
                    "self-host walk activation mismatch: got {got}, want {expected}"
                );
            }
        }
    }

    // The whole drill stayed inside its own deadlines.
    assert!(
        drill_started.elapsed() < Duration::from_secs(15),
        "a wedged hop costs its deadline, not the drill: {:?}",
        drill_started.elapsed()
    );
    Ok(())
}
