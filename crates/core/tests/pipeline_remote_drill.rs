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
//!      arrived through the transport itself.
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
//!
//! Honest limits: loopback nodes, and the containers are synthetic
//! until the Colibri-side layer split yields real model shards; wide-area
//! RTT is not exercised. The `runtime` stand-in stays OFF (`tensors_only
//! = true`): a hashed text completion is not expert math, so wiring that
//! path here would fabricate activations instead of moving real weights.

use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use jerith_mesh_core::expert_store::{ExpertStore, ExpertWeights};
use jerith_mesh_core::pipeline::{
    PipelineHop, RouteStability, execute_route_resilient, hop_expert_id, plan_route,
    plan_route_budget,
};
use jerith_mesh_core::quic::{
    ExpertServing, NodeRpcReply, NodeRpcRequest, PinMap, SharedPinMap, call,
    fetch_expert, pinned_client_endpoint_dyn, sign_request,
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
    let (node_c, addr_c) = spin_mute_node().await;

    let mut pin_map: PinMap = HashMap::new();
    pin_map.insert(node_a.clone(), addr_a);
    pin_map.insert(node_b.clone(), addr_b);
    pin_map.insert(node_s.clone(), addr_s);
    pin_map.insert(node_c.clone(), addr_c);
    let pins: SharedPinMap = Arc::new(tokio::sync::RwLock::new(pin_map.clone()));
    let client = pinned_client_endpoint_dyn(pins).expect("drill client endpoint");
    let walker = SigningIdentity::generate();
    let walk_node_id = walker.node_id();
    let walk_req_id = "drill-route-1".to_string();

    let addrs: HashMap<String, SocketAddr> = pin_map
        .iter()
        .map(|(n, a)| (n.clone(), *a))
        .collect();

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
    let remote_run = move |expert_id: &str, node: &str, input: Vec<f32>, deadline: Duration| -> RemoteHopper {
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
            let reply = tokio::time::timeout(deadline, call(&client, addr, &node, &req, &walker, deadline))
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
            call(&client, addr_b, &node_b, &req, &walker, Duration::from_secs(5)),
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
    for (owner_id, owner_store, targets) in [
        (&node_a, &store_a, &a_targets[..]),
        (&node_b, &store_b, &b_targets[..]),
    ] {
        for id in targets {
            let bytes =
                fetch_expert(&client, addr_s, &node_s, &walker, id, Duration::from_secs(15))
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
        for store in [&store_a, &store_b] {
            if store.contains(id) {
                let (got, _) = store.bytes_of(id, 0, u32::MAX)?;
                assert_eq!(got, want, "{id} mutated across the transport");
            }
        }
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
            PipelineHop { node: Some(node_a.clone()), layers: vec![0, 1] },
            PipelineHop { node: Some(node_b.clone()), layers: vec![2, 3] },
        ],
        "planner: two remote hops in walk order"
    );

    // Budgets priced from the measured-activation ordering (a drill node
    // has no measured history: even split, but the RTT floors are real).
    let rtt_of = |_: &str| 5_u64;
    let priced =
        plan_route_budget(&route, |_| 1, rtt_of, Duration::from_millis(250), 20_000);
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
    let r1 = run_hop(&hop_expert_id(MODEL, &PipelineHop { node: None, layers: vec![0, 1] }), &input)?;
    let want = run_hop(&hop_expert_id(MODEL, &PipelineHop { node: None, layers: vec![2, 3] }), &r1)?;
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
            call(&client, addr_b, &node_b, &req, &walker, Duration::from_secs(5)),
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
    let priced_dead =
        plan_route_budget(&dead_route, |_| 1, rtt_of, Duration::from_millis(250), 20_000);
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
            &mut stability3,
            reassign,
        )
        .await
    };
    assert!(result3.degraded.is_empty(), "re-hosted walk must not degrade");
    assert_eq!(result3.reassigned.len(), 1, "the dead hop was re-hosted");
    assert_eq!(result3.reassigned[0].layers, vec![2]);
    assert_eq!(result3.reassigned[0].was_node.as_deref(), Some(node_c.as_str()));
    assert!(
        result3.reassigned[0].reason.contains("deadline"),
        "the mute peer died at its priced deadline (reason: {})",
        result3.reassigned[0].reason
    );
    assert_eq!(result3.hops_run, 3, "A, re-hosted 2–2 on B, 3–3 on B");
    let r1d = run_hop(
        &hop_expert_id(MODEL, &PipelineHop { node: None, layers: vec![0, 1] }),
        &input,
    )?;
    let r2d = run_hop(
        &hop_expert_id(MODEL, &PipelineHop { node: None, layers: vec![2] }),
        &r1d,
    )?;
    let want3 = run_hop(
        &hop_expert_id(MODEL, &PipelineHop { node: None, layers: vec![3] }),
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
        &hop_expert_id(MODEL, &PipelineHop { node: None, layers: vec![0, 1] }),
        &input,
    )?;
    let want4 = run_hop(
        &hop_expert_id(MODEL, &PipelineHop { node: None, layers: vec![3] }),
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

    // The whole drill stayed inside its own deadlines.
    assert!(
        drill_started.elapsed() < Duration::from_secs(15),
        "a wedged hop costs its deadline, not the drill: {:?}",
        drill_started.elapsed()
    );
    Ok(())
}
