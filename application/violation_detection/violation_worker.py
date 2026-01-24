import time
import signal
import threading
import sys
import os
import cv2
import numpy as np
from db_redis.sentinel_redis_config import *
import logging

shutdown_event = threading.Event()

def handle_shutdown(signum, frame):
    print(f"\nReceived signal {signum}, shutting down Violation worker gracefully...")
    shutdown_event.set()

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

MODEL_PATH = "models/violations_yolo11n.onnx" 
CONF_THRESHOLD = 0.45  
NMS_THRESHOLD = 0.45
IOU_THRESHOLD = 0.60 # preven double boxes on the same head
INPUT_SIZE = (640, 640)

print(f"Loading Violation Model from {MODEL_PATH}...")
try:
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    
    #  use GPU if available else CPU
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        logging.iNMS_THRESHOLDnfo("Using OpenCV DNN with CUDA (GPU)")
    except:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logging.info("Using OpenCV DNN on CPU")
        
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    sys.exit(1)

def get_violation_code(frame_path):
    if not os.path.exists(frame_path):
        return 0

    img = cv2.imread(frame_path)
    if img is None: 
        return 0
    
    # opencv preprocess with blobfromiamge -> handels resizing normaalzingg and swapping channels etc
    blob = cv2.dnn.blobFromImage(img, 1/255.0, INPUT_SIZE, swapRB=True, crop=False)
    net.setInput(blob)

    # inference
    try:
        # runs the model
        outputs = net.forward()
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        return 0

    # posr processinng -> filtering the results
    # YOLO output format: [1, 6, 8400] -> Needs transposing to [8400, 6]
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # parse the rows 
    for i in range(rows):
        row = outputs[0][i]
        confidence = row[4:].max() # find highest score among classes
        
        if confidence >= CONF_THRESHOLD:
            class_id = row[4:].argmax()
            
            # extract box coordinates (center_x, center_y, w, h)
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            
            # scale box back to original image size
            h_factor = img.shape[0] / INPUT_SIZE[1]
            w_factor = img.shape[1] / INPUT_SIZE[0]
            
            left = int((cx - w/2) * w_factor)
            top = int((cy - h/2) * h_factor)
            width = int(w * w_factor)
            height = int(h * h_factor)
            
            boxes.append([left, top, width, height])
            scores.append(float(confidence))
            class_ids.append(class_id)

    # nms to remove duples    
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)

    # Count results
    num_helmets = 0
    num_no_helmets = 0
    
    if len(indices) > 0:
        for i in indices.flatten():
            cid = class_ids[i]
            if cid == 0: num_helmets += 1
            elif cid == 1: num_no_helmets += 1

    # logic to get teh code
    total_people = num_helmets + num_no_helmets
    has_no_helmet = num_no_helmets > 0
    is_triple_riding = total_people >= 3
    
    violation_code = 0
    details = "Clean"

    if has_no_helmet and is_triple_riding:
        violation_code = 3
        details = f"BOTH: No Helmet ({num_no_helmets}) + 3x Riding ({total_people})"
    elif is_triple_riding:
        violation_code = 2
        details = f"3x Riding ({total_people} pax)"
    elif has_no_helmet:
        violation_code = 1
        details = f"No Helmet ({num_no_helmets})"
        
    logging.info(f"-> Result: Code {violation_code} | {details}")
    return violation_code

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