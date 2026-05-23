import redis
import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

def get_redis_connection():
    return redis.Redis(
        host=REDIS_HOST, 
        port=REDIS_PORT, 
        db=REDIS_DB, 
        decode_responses=True
    )

# Stream Names
VEHICLE_JOBS_STREAM = "vehicle_jobs"
VEHICLE_RESULTS_STREAM = "vehicle_results"
VEHICLE_ACK_STREAM = "vehicle_ack"

# Consumer Groups
OCR_GROUP = "ocr_workers"
COLOR_GROUP = "color_workers"
LOGO_GROUP = "logo_workers"
VIOLATION_GROUP = "violation_workers"
AGGREGATOR_GROUP = "aggregator" 
INGEST_GROUP = "ingest"

# Processing Configuration
BATCH_SIZE = 10          # messages per batch
BLOCK_TIME = 1000        # ms to block on XREADGROUP
ACK_TIMEOUT = 30000      # ms before reclaim (30 seconds)
MAX_RETRIES = 3
JOB_TIMEOUT_SEC = int(os.getenv("JOB_TIMEOUT_SEC", "60"))
RECLAIM_BATCH = int(os.getenv("RECLAIM_BATCH", "10"))

# Worker Types
WORKER_TYPES = {
    "ocr": ["car", "bus", "motorcycle", "truck", "auto"],
    "color": ["car"],
    "logo": ["car"],
    "violation": ["motorcycle"]
}

def get_expected_workers(vehicle_type):
    """Get list of workers expected for a vehicle type"""
    if vehicle_type == "car":
        return ["ocr", "color", "logo"]
    elif vehicle_type == "motorcycle":
        return ["ocr", "violation"]
    else:
        return ["ocr"]

def should_worker_process(worker_type, vehicle_type):
    """Check if worker should process this vehicle type"""
    return vehicle_type in WORKER_TYPES.get(worker_type, [])

def reclaim_pending_messages(r, group, consumer, stream, min_idle_ms, count=RECLAIM_BATCH):
    """Reclaim idle pending messages for this consumer group."""
    try:
        if hasattr(r, "xautoclaim"):
            _next_id, msgs = r.xautoclaim(stream, group, consumer, min_idle_ms, count=count)
            return msgs or []
        result = r.execute_command("XAUTOCLAIM", stream, group, consumer, min_idle_ms, "0-0", "COUNT", count)
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            return result[1] or []
    except Exception:
        pass
    return []
