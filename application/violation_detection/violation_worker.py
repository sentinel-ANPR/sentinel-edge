import time
import signal
import threading
import sys
import os
from ultralytics import YOLO
from db_redis.sentinel_redis_config import *
from model_config import resolve_model_path
from telemetry import (
    observe_worker_latency,
    record_reclaim,
    record_worker_result,
)

shutdown_event = threading.Event()

def handle_shutdown(signum, frame):
    print(f"\nReceived signal {signum}, shutting down Violation worker gracefully...")
    shutdown_event.set()

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

MODEL_PATH = resolve_model_path("MODEL_VIOLATION_PATH", "models/violations_yolo11n.pt")

# post-processing controls for duplicate / edge-box filtering
CONF_THRESHOLD = 0.40
DUPLICATE_IOU_THRESHOLD = 0.55
EDGE_MARGIN_RATIO = 0.02
MIN_NO_HELMET_AREA_RATIO = 0.0015
MIN_NO_HELMET_HEIGHT_RATIO = 0.035

print(f"Loading Violation Model from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print(f"Model Classes: {model.names}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def deduplicate_by_iou(detections, iou_threshold=DUPLICATE_IOU_THRESHOLD):
    """Keep highest-confidence box among highly-overlapping detections."""
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
    kept = []

    for det in sorted_dets:
        is_duplicate = False
        for existing in kept:
            if compute_iou(det["xyxy"], existing["xyxy"]) >= iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(det)

    return kept


def is_partial_edge_nohelmet(box, img_w, img_h):
    """Treat small border-touching boxes as likely cropped/off-screen false positives."""
    x1, y1, x2, y2 = box
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    box_area = box_w * box_h
    img_area = float(max(1, img_w * img_h))

    margin_x = img_w * EDGE_MARGIN_RATIO
    margin_y = img_h * EDGE_MARGIN_RATIO
    touches_edge = (
        x1 <= margin_x or
        y1 <= margin_y or
        x2 >= (img_w - margin_x) or
        y2 >= (img_h - margin_y)
    )

    area_ratio = box_area / img_area
    height_ratio = box_h / float(max(1, img_h))

    return touches_edge and (
        area_ratio < MIN_NO_HELMET_AREA_RATIO or
        height_ratio < MIN_NO_HELMET_HEIGHT_RATIO
    )

def get_violation_code(frame_path):
    if not frame_path or not os.path.exists(frame_path):
        print(f"Error: Image path invalid {frame_path}")
        return 0

    try:
        results = model(frame_path, verbose=False)
        
        if not results or results[0].boxes is None:
            return 0

        result = results[0]
        boxes = result.boxes
        img_h, img_w = result.orig_shape if hasattr(result, "orig_shape") else (0, 0)
        
        # ids nad counters
        HELMET_ID = 0 
        NO_HELMET_ID = 1        
        helmet_detections = []
        nohelmet_detections = []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = tuple(float(v) for v in box.xyxy[0].tolist())

            if cls_id == HELMET_ID:
                helmet_detections.append({"conf": conf, "xyxy": xyxy})
            
            elif cls_id == NO_HELMET_ID:
                if conf > CONF_THRESHOLD:
                    if img_w > 0 and img_h > 0 and is_partial_edge_nohelmet(xyxy, img_w, img_h):
                        print(f"   Ignored No-Helmet (Conf: {conf:.2f}) - PARTIAL/OFFSCREEN")
                        continue
                    nohelmet_detections.append({"conf": conf, "xyxy": xyxy})
                else:
                    print(f"   Ignored No-Helmet (Conf: {conf:.2f}) - TOO LOW")

        helmet_detections = deduplicate_by_iou(helmet_detections)
        nohelmet_detections = deduplicate_by_iou(nohelmet_detections)

        num_helmets = len(helmet_detections)
        num_no_helmets = len(nohelmet_detections)

        if num_no_helmets > 0:
            print(f"   Counted No-Helmet after dedupe/filter: {num_no_helmets}")

        total_people = num_helmets + num_no_helmets
        
        # for code 0-3
        has_no_helmet = num_no_helmets > 0
        is_triple_riding = total_people >= 3
        
        violation_code = 0
        details = ""
        
        if has_no_helmet and is_triple_riding:
            violation_code = 3
            details = f"BOTH: No Helmet ({num_no_helmets}) + 3x Riding ({total_people})"
        elif is_triple_riding:
            violation_code = 2
            details = f"3x Riding ({total_people} pax)"
        elif has_no_helmet:
            violation_code = 1
            details = f"No Helmet ({num_no_helmets})"
        else:
            violation_code = 0
            details = "Clean"

        # trace log
        print(f"  -> [WORKER OUTPUT] Code: {violation_code} | {details}")
        return violation_code

    except Exception as e:
        print(f"Inference Error: {e}")
        return 0

def violation_worker():
    r = get_redis_connection()
    worker_id = "violation_worker_1"
    
    # Ensure Consumer Group Exists
    try:
        r.xgroup_create(VEHICLE_JOBS_STREAM, VIOLATION_GROUP, id="0", mkstream=True)
    except:
        pass 

    print(f"[Violation] Worker started: {worker_id}")
    
    while not shutdown_event.is_set():
        try:
            # Read from Redis Stream
            reclaimed = reclaim_pending_messages(
                r, VIOLATION_GROUP, worker_id, VEHICLE_JOBS_STREAM, ACK_TIMEOUT
            )
            if reclaimed:
                record_reclaim(VEHICLE_JOBS_STREAM, VIOLATION_GROUP, len(reclaimed))
                messages = [(VEHICLE_JOBS_STREAM, reclaimed)]
            else:
                messages = r.xreadgroup(
                    VIOLATION_GROUP, worker_id, 
                    {VEHICLE_JOBS_STREAM: ">"}, 
                    count=1, block=BLOCK_TIME
                )
            
            if not messages:
                continue
            
            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    job_id = fields.get("job_id")
                    vehicle_type = fields.get("vehicle_type")
                    frame_path = fields.get("frame_path")
                    plate_path = fields.get("plate_path")
                    vehicle_id = fields.get("vehicle_id")
                    
                    print(f"[Violation] Processing: {vehicle_id} ({vehicle_type})")
                    
                    if should_worker_process("violation", vehicle_type):
                        try:
                            # Get Integer Code
                            start_ts = time.time()
                            v_code = get_violation_code(frame_path)
                            
                            # Publish Result
                            # Send 'result' as string for consistency, but the payload contains the code
                            payload = {
                                "job_id": job_id,
                                "vehicle_id": vehicle_id,
                                "job_msg_id": msg_id,
                                "worker": "violation",
                                "result": str(v_code),
                                "status": "ok",
                                "frame_path": frame_path, 
                                "plate_path": plate_path  
                            }
                            r.xadd(VEHICLE_RESULTS_STREAM, payload)
                            
                            r.xack(VEHICLE_JOBS_STREAM, VIOLATION_GROUP, msg_id)
                            observe_worker_latency("violation", time.time() - start_ts)
                            record_worker_result("violation", "ok", vehicle_type)
                            
                        except Exception as e:
                            print(f"[Violation] Failed for {job_id}: {e}")
                            # send 0 on error to avoid blocking the aggregator
                            error_payload = {
                                "job_id": job_id,
                                "vehicle_id": vehicle_id,
                                "job_msg_id": msg_id,
                                "worker": "violation",
                                "result": "0",
                                "status": "error"
                            }
                            r.xadd(VEHICLE_RESULTS_STREAM, error_payload)
                            r.xack(VEHICLE_JOBS_STREAM, VIOLATION_GROUP, msg_id)
                            record_worker_result("violation", "error", vehicle_type)
                    else:
                        print(f"[Violation] Skipping {vehicle_type}")
                        r.xack(VEHICLE_JOBS_STREAM, VIOLATION_GROUP, msg_id)
                        record_worker_result("violation", "skipped", vehicle_type)
                        
        except Exception as e:
            print(f"[Violation] Worker loop error: {e}")
            time.sleep(1)
    
    print("[Violation] Shutdown complete.")

if __name__ == "__main__":
    violation_worker()