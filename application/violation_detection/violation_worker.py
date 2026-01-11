import time
import signal
import threading
import sys
import os
from ultralytics import YOLO
from db_redis.sentinel_redis_config import *

shutdown_event = threading.Event()

def handle_shutdown(signum, frame):
    print(f"\nReceived signal {signum}, shutting down Violation worker gracefully...")
    shutdown_event.set()

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

MODEL_PATH = "models/violations_yolo11n_openvino_model/"

print(f"Loading Violation Model from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH, task="detect")
    print(f"Model Classes: {model.names}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

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
        
        # ids nad counters
        HELMET_ID = 0 
        NO_HELMET_ID = 1        
        num_helmets = 0
        num_no_helmets = 0
        
        # cehck confidence before setting 
        CONF_THRESHOLD = 0.40  

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id == HELMET_ID:
                num_helmets += 1
            
            elif cls_id == NO_HELMET_ID:
                if conf > CONF_THRESHOLD:
                    num_no_helmets += 1
                    print(f"   Found No-Helmet (Conf: {conf:.2f}) - COUNTED")
                else:
                    print(f"   Ignored No-Helmet (Conf: {conf:.2f}) - TOO LOW")

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
                            v_code = get_violation_code(frame_path)
                            
                            # Publish Result
                            # Send 'result' as string for consistency, but the payload contains the code
                            r.xadd(VEHICLE_RESULTS_STREAM, {
                                "job_id": job_id,
                                "vehicle_id": vehicle_id,
                                "worker": "violation",
                                "result": str(v_code),
                                "status": "ok",
                                "frame_path": frame_path, 
                                "plate_path": plate_path  
                            })
                            
                            r.xack(VEHICLE_JOBS_STREAM, VIOLATION_GROUP, msg_id)
                            
                        except Exception as e:
                            print(f"[Violation] Failed for {job_id}: {e}")
                            # send 0 on error to avoid blocking the aggregator
                            r.xadd(VEHICLE_RESULTS_STREAM, {
                                "job_id": job_id,
                                "vehicle_id": vehicle_id,
                                "worker": "violation",
                                "result": "0",
                                "status": "error"
                            })
                            r.xack(VEHICLE_JOBS_STREAM, VIOLATION_GROUP, msg_id)
                    else:
                        print(f"[Violation] Skipping {vehicle_type}")
                        r.xack(VEHICLE_JOBS_STREAM, VIOLATION_GROUP, msg_id)
                        
        except Exception as e:
            print(f"[Violation] Worker loop error: {e}")
            time.sleep(1)
    
    print("[Violation] Shutdown complete.")

if __name__ == "__main__":
    violation_worker()