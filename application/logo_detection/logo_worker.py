import time
import signal
import threading
import cv2
from ultralytics import YOLO
from pathlib import Path
from db_redis.sentinel_redis_config import *

shutdown_event = threading.Event()

def handle_shutdown(signum, frame):
    print(f"\nReceived signal {signum}, shutting down Logo worker gracefully...")
    shutdown_event.set()

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGOS_PATH = PROJECT_ROOT / "static" / "logos"
LOGOS_PATH.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "models/logo-detector-yolo.pt"
LOGO_CONF_THRESHOLD = 0.25
LOGO_IMGSZ = 640
LOGO_IOU_THRESHOLD = 0.45
LOGO_MAX_DET = 10

print(f"[Logo] Loading YOLO logo model from {MODEL_PATH}...")
try:
    logo_model = YOLO(MODEL_PATH)
    print(f"[Logo] Model classes loaded: {len(logo_model.names)}")
except Exception as e:
    print(f"[Logo] ERROR: Failed to load model: {e}")
    logo_model = None


def normalize_logo_class(raw_name):
    if not raw_name:
        return "Unknown"
    base = raw_name.strip().lower()
    if "_" in base and base.rsplit("_", 1)[-1].isdigit():
        base = base.rsplit("_", 1)[0]
    return base

def process_logo(frame_path, vehicle_id):
    """
    Detect and crop car logo from keyframe image.
    Returns: (make_name, logo_path)
    """
    if logo_model is None:
        print(f"[Logo] Model not loaded, returning default values")
        return "Unknown", None
    
    # Load the keyframe image
    image = cv2.imread(frame_path)
    if image is None:
        print(f"[Logo] Error: Could not load image from {frame_path}")
        return "Unknown", None
    
    (img_h, img_w) = image.shape[:2]

    try:
        results = logo_model(
            image,
            verbose=False,
            conf=LOGO_CONF_THRESHOLD,
            iou=LOGO_IOU_THRESHOLD,
            imgsz=LOGO_IMGSZ,
            max_det=LOGO_MAX_DET,
        )
    except Exception as e:
        print(f"[Logo] Inference error: {e}")
        return "Unknown", None

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        print(f"[Logo] No logo detected in {frame_path}")
        return "Unknown", None

    best_det = None
    best_conf = -1.0
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < LOGO_CONF_THRESHOLD:
            continue

        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        if x2 <= x1 or y2 <= y1:
            continue

        if conf > best_conf:
            best_conf = conf
            best_det = (cls_id, x1, y1, x2, y2)

    if best_det is None:
        print(f"[Logo] No logo passed threshold in {frame_path}")
        return "Unknown", None

    cls_id, x1, y1, x2, y2 = best_det
    cropped_logo = image[y1:y2, x1:x2]
    if cropped_logo.size == 0:
        print(f"[Logo] Empty crop for {frame_path}")
        return "Unknown", None

    raw_name = str(logo_model.names.get(cls_id, "Unknown"))
    logo_make = normalize_logo_class(raw_name)

    logo_filename = f"{vehicle_id}_logo.jpg"
    logo_path = LOGOS_PATH / logo_filename
    write_ok = cv2.imwrite(str(logo_path), cropped_logo)
    if not write_ok:
        print(f"[Logo] Failed to save cropped logo: {logo_path}")
        return "Unknown", None

    print(f"[Logo] Detected {logo_make} (confidence: {best_conf:.2f})")
    print(f"[Logo] Saved cropped logo to: {logo_path}")

    return logo_make, str(logo_path)

def logo_worker():
    r = get_redis_connection()
    worker_id = "logo_worker_1"
    
    print(f"[Logo] Worker started: {worker_id}")
    
    while not shutdown_event.is_set():
        try:
            messages = r.xreadgroup(
                LOGO_GROUP, worker_id,
                {VEHICLE_JOBS_STREAM: ">"}, 
                count=1, block=BLOCK_TIME
            )
            
            if not messages:
                continue
            
            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    job_id = fields.get("job_id")
                    vehicle_type = fields.get("vehicle_type")
                    vehicle_id = fields.get("vehicle_id")
                    frame_path = fields.get("frame_path")
                    plate_path = fields.get("plate_path")
                    
                    print(f"[Logo] Processing job: {job_id} ({vehicle_type})")
                    
                    if should_worker_process("logo", vehicle_type):
                        try:
                            make, logo_path = process_logo(frame_path, vehicle_id)
                            
                            r.xadd(VEHICLE_RESULTS_STREAM, {
                                "job_id": job_id,
                                "vehicle_id": vehicle_id,
                                "worker": "logo",
                                "result": make,  # car make/brand name
                                "logo_path": logo_path if logo_path else "N/A",  # path to cropped logo
                                "status": "ok",
                                "frame_path": frame_path, 
                                "plate_path": plate_path  
                            })
                            print(f"[Logo] Completed: {job_id} -> {make}")
                            if logo_path:
                                print(f"[Logo] Logo saved at: {logo_path}")
                            r.xack(VEHICLE_JOBS_STREAM, LOGO_GROUP, msg_id)
                        except Exception as e:
                            print(f"[Logo] Failed for {job_id}: {e}")
                            r.xadd(VEHICLE_RESULTS_STREAM, {
                                "job_id": job_id,
                                "vehicle_id": vehicle_id,
                                "worker": "logo",
                                "result": "Unknown",
                                "logo_path": "N/A",
                                "status": "error",
                                "error": str(e)
                            })
                            r.xack(VEHICLE_JOBS_STREAM, LOGO_GROUP, msg_id)
                    else:
                        print(f"[Logo] Skipping {vehicle_type} (not in scope)")
                        r.xack(VEHICLE_JOBS_STREAM, LOGO_GROUP, msg_id)
                        
        except Exception as e:
            print(f"[Logo] Worker error: {e}")
            time.sleep(1)
    
    print("[Logo] Shutdown complete.")

if __name__ == "__main__":
    logo_worker()
