import os
import time
import threading

PROM_ENABLED = False

PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "").strip()
PROM_PUSH_INTERVAL = int(os.getenv("PROM_PUSH_INTERVAL", "10"))
PROM_JOB_NAME = os.getenv("PROM_JOB_NAME", "sentinel_edge")
PROM_INSTANCE = os.getenv("NODE_ID") or os.getenv("LOCATION") or "unknown"
EDGE_LOCATION = os.getenv("LOCATION", "unknown")

registry = None
JOBS_CREATED = None
WORKER_RESULTS = None
WORKER_LATENCY = None
JOB_COMPLETED = None
JOB_TIMEOUTS = None
JOB_LATENCY = None
UPLOAD_LATENCY = None
UPLOAD_FAILURES = None
REDIS_STREAM_LEN = None
REDIS_GROUP_PENDING = None
REDIS_GROUP_LAG = None
REDIS_RECLAIMS = None

try:
    from prometheus_client import (  # type: ignore
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        push_to_gateway,
    )

    registry = CollectorRegistry()

    JOBS_CREATED = Counter(
        "edge_jobs_created_total",
        "Total jobs created on edge",
        ["location", "vehicle_type"],
        registry=registry,
    )
    WORKER_RESULTS = Counter(
        "edge_worker_results_total",
        "Worker results by status",
        ["location", "worker", "status", "vehicle_type"],
        registry=registry,
    )
    WORKER_LATENCY = Histogram(
        "edge_worker_latency_seconds",
        "Worker processing latency",
        ["location", "worker"],
        registry=registry,
    )
    JOB_COMPLETED = Counter(
        "edge_jobs_completed_total",
        "Completed jobs uploaded to central",
        ["location", "vehicle_type"],
        registry=registry,
    )
    JOB_TIMEOUTS = Counter(
        "edge_jobs_timeout_total",
        "Jobs finalized with fallback due to timeout",
        ["location", "vehicle_type"],
        registry=registry,
    )
    JOB_LATENCY = Histogram(
        "edge_job_latency_seconds",
        "End-to-end job latency on edge",
        ["location", "vehicle_type"],
        registry=registry,
    )
    UPLOAD_LATENCY = Histogram(
        "edge_upload_latency_seconds",
        "Upload latency to central",
        ["location"],
        registry=registry,
    )
    UPLOAD_FAILURES = Counter(
        "edge_upload_failures_total",
        "Upload failures",
        ["location"],
        registry=registry,
    )
    REDIS_STREAM_LEN = Gauge(
        "edge_redis_stream_length",
        "Redis stream length",
        ["location", "stream"],
        registry=registry,
    )
    REDIS_GROUP_PENDING = Gauge(
        "edge_redis_group_pending",
        "Redis pending entries by group",
        ["location", "stream", "group"],
        registry=registry,
    )
    REDIS_GROUP_LAG = Gauge(
        "edge_redis_group_lag",
        "Redis lag by group",
        ["location", "stream", "group"],
        registry=registry,
    )
    REDIS_RECLAIMS = Counter(
        "edge_redis_reclaims_total",
        "Redis pending reclaims",
        ["location", "stream", "group"],
        registry=registry,
    )

    PROM_ENABLED = True
except Exception:
    PROM_ENABLED = False

tracer = None

def _push_once() -> None:
    if not PROM_ENABLED or not PUSHGATEWAY_URL:
        return
    push_to_gateway(
        PUSHGATEWAY_URL,
        job=PROM_JOB_NAME,
        registry=registry,
        grouping_key={"instance": PROM_INSTANCE},
    )


def start_metrics_pusher() -> bool:
    if not PROM_ENABLED or not PUSHGATEWAY_URL:
        return False

    def _loop():
        while True:
            _push_once()
            time.sleep(PROM_PUSH_INTERVAL)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return True


def record_job_created(location: str, vehicle_type: str) -> None:
    if PROM_ENABLED and JOBS_CREATED:
        JOBS_CREATED.labels(location=location, vehicle_type=vehicle_type).inc()


def record_worker_result(worker: str, status: str, vehicle_type: str) -> None:
    if PROM_ENABLED and WORKER_RESULTS:
        WORKER_RESULTS.labels(
            location=EDGE_LOCATION,
            worker=worker,
            status=status,
            vehicle_type=vehicle_type,
        ).inc()


def observe_worker_latency(worker: str, seconds: float) -> None:
    if PROM_ENABLED and WORKER_LATENCY:
        WORKER_LATENCY.labels(location=EDGE_LOCATION, worker=worker).observe(seconds)


def record_job_completed(location: str, vehicle_type: str, latency_s: float) -> None:
    if PROM_ENABLED and JOB_COMPLETED and JOB_LATENCY:
        JOB_COMPLETED.labels(location=location, vehicle_type=vehicle_type).inc()
        JOB_LATENCY.labels(location=location, vehicle_type=vehicle_type).observe(latency_s)


def record_job_timeout(location: str, vehicle_type: str) -> None:
    if PROM_ENABLED and JOB_TIMEOUTS:
        JOB_TIMEOUTS.labels(location=location, vehicle_type=vehicle_type).inc()


def record_upload_latency(location: str, seconds: float, success: bool) -> None:
    if PROM_ENABLED and UPLOAD_LATENCY:
        UPLOAD_LATENCY.labels(location=location).observe(seconds)
    if PROM_ENABLED and UPLOAD_FAILURES and not success:
        UPLOAD_FAILURES.labels(location=location).inc()


def record_reclaim(stream: str, group: str, count: int) -> None:
    if PROM_ENABLED and REDIS_RECLAIMS:
        REDIS_RECLAIMS.labels(location=EDGE_LOCATION, stream=stream, group=group).inc(count)


def update_redis_metrics(stream_snapshot: dict) -> None:
    if not PROM_ENABLED:
        return

    for stream, data in stream_snapshot.items():
        if data is None:
            continue
        if REDIS_STREAM_LEN:
            REDIS_STREAM_LEN.labels(location=EDGE_LOCATION, stream=stream).set(data.get("messages", 0))
        groups = data.get("groups", []) or []
        for group in groups:
            group_name = group.get("name") if isinstance(group, dict) else None
            pending = group.get("pending", 0) if isinstance(group, dict) else 0
            lag = group.get("lag", 0) if isinstance(group, dict) else 0
            if group_name and REDIS_GROUP_PENDING:
                REDIS_GROUP_PENDING.labels(location=EDGE_LOCATION, stream=stream, group=group_name).set(pending)
            if group_name and REDIS_GROUP_LAG:
                REDIS_GROUP_LAG.labels(location=EDGE_LOCATION, stream=stream, group=group_name).set(lag)
