//! End-to-end tests for the observational one-token inference probe.
//!
//! The probe exists to make visible the failure described in issue #158: a
//! worker whose process is up — and which the health checker therefore keeps in
//! rotation — but which cannot actually complete a generation.
//!
//! The invariant these tests protect is that the probe *reports only*. It must
//! never mark a worker unhealthy, never mark one healthy, and never touch load
//! accounting, so that a defect in the probe cannot withdraw a good worker from
//! routing.

mod common;

use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig};
use std::sync::Arc;
use vllm_router_rs::config::InferenceProbeConfig;
use vllm_router_rs::core::inference_probe::probe_inference;
use vllm_router_rs::core::{BasicWorker, Worker, WorkerRegistry, WorkerType};

/// A worker whose `/health` answers 200 while every inference request fails.
///
/// This is exactly the shape of #158: the liveness signal says healthy, the
/// inference path says otherwise.
#[tokio::test]
async fn probe_reports_worker_that_is_healthy_but_cannot_serve() {
    let mut mock = MockWorker::new(MockWorkerConfig {
        health_status: HealthStatus::Healthy,
        fail_rate: 1.0,
        ..Default::default()
    });
    let url = mock.start().await.expect("mock worker starts");

    let worker = BasicWorker::new(url, WorkerType::Regular);

    // The health checker's own probe is satisfied: `/health` is fine. This is
    // why the node stays in rotation and keeps receiving traffic.
    assert!(worker.check_health_async().await.is_ok());
    assert!(
        worker.is_healthy(),
        "the liveness probe sees a healthy worker"
    );

    // The inference probe disagrees, which is the whole point.
    let outcome = probe_inference(&worker, 5).await;
    assert!(
        !outcome.ok,
        "a worker that cannot complete a generation must be reported as failing"
    );
    assert_eq!(outcome.reason, "http_error");

    // ...and reports only. Nothing about routing may change as a result.
    assert!(
        worker.is_healthy(),
        "the inference probe must not mark the worker unhealthy"
    );
    assert_eq!(worker.load(), 0, "the probe must not touch load accounting");

    mock.stop().await;
}

/// A serving worker must be reported as healthy, but a passing probe must not
/// flip a worker's health either.
#[tokio::test]
async fn probe_reports_a_serving_worker_without_changing_health() {
    let mut mock = MockWorker::new(MockWorkerConfig {
        health_status: HealthStatus::Healthy,
        fail_rate: 0.0,
        ..Default::default()
    });
    let url = mock.start().await.expect("mock worker starts");

    let worker = BasicWorker::new(url, WorkerType::Regular);
    worker.set_healthy(false);

    let outcome = probe_inference(&worker, 5).await;

    assert!(outcome.ok, "unexpected failure: {}", outcome.detail);
    assert_eq!(outcome.reason, "ok");
    assert!(
        !worker.is_healthy(),
        "a passing probe reports; it must not mark the worker healthy"
    );

    mock.stop().await;
}

/// Driving the periodic sweep over a registry must leave every worker's health
/// exactly as the health checker left it.
#[tokio::test]
async fn periodic_sweep_never_mutates_worker_health() {
    let mut mock = MockWorker::new(MockWorkerConfig {
        health_status: HealthStatus::Healthy,
        fail_rate: 1.0,
        ..Default::default()
    });
    let url = mock.start().await.expect("mock worker starts");

    let worker = Arc::new(BasicWorker::new(url, WorkerType::Regular));
    worker.set_healthy(true);

    let registry = WorkerRegistry::new();
    registry.register(worker.clone() as Arc<dyn Worker>);

    let handle = registry.start_inference_probe(InferenceProbeConfig {
        enabled: true,
        interval_secs: 1,
        timeout_secs: 5,
    });

    // Let at least one sweep run, twice over, to catch a mutating probe.
    tokio::time::sleep(tokio::time::Duration::from_millis(2500)).await;
    handle.stop();

    assert!(
        worker.is_healthy(),
        "a full sweep of a failing worker must not change its health"
    );
    assert_eq!(worker.load(), 0, "probes must not touch load accounting");

    mock.stop().await;
}
