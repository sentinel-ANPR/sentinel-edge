import time
import signal
import sys
import threading
import cv2
import numpy as np
import os
from collections import Counter
from sklearn.cluster import KMeans
from ultralytics import YOLO
from db_redis.sentinel_redis_config import *

YOLO_MODEL_PATH = "models/color/colour-yolo_openvino_model/"

# if final score is lower it goes to Other
CONF_THRESH = 0.55 

# output classes
CLASSES = ['Black', 'Blue', 'Gray', 'White', 'Red', 'Night', 'Other']

shutdown_event = threading.Event()

def handle_shutdown(signum, frame):
    print(f"\nReceived signal {signum}, shutting down Color Worker gracefully...")
    shutdown_event.set()

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# gamma correction to brighten
def adjust_gamma(image, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# detect if an image is in IR mode
def is_monochrome(image_bgr, mean_thresh=0.03, std_thresh=0.03):
    small = cv2.resize(image_bgr, (64, 64), interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype("float32") / 255.0
    return (S.mean() < mean_thresh) and (S.std() < std_thresh)

# get region to find dominant hex
def extract_color_roi(img_rgb):
    h, w, _ = img_rgb.shape
    x1 = int(0.05 * w)
    x2 = int(0.95 * w)
    bottom_ignore = int(0.10 * h)
    sample_height = int(0.25 * h)
    offset_up = int(0.03 * h)
    y2 = h - bottom_ignore - offset_up
    y1 = max(0, y2 - sample_height)
    return img_rgb[y1:y2, x1:x2]

# get dominant hex
def get_hex_color(image_bgr, k=1):
    img = cv2.resize(image_bgr, (100, 100))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3)
    
    # remove near-black pixels
    pixels = pixels[np.any(pixels > 20, axis=1)]
    
    if len(pixels) == 0: 
        return "#000000"

    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    try:
        _, _, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(centers[0])
        return "#{:02x}{:02x}{:02x}".format(center[0], center[1], center[2])
    except Exception:
        return "#000000"

def load_model():
    print("[INFO] Loading YOLO...")
    return YOLO(YOLO_MODEL_PATH)

# use yolo to get the class
def process_color(frame_path, yolo_model):
    try:
        original_img = cv2.imread(frame_path)
        if original_img is None:
            return "unknown", "#000000"

        # monochrome
        if is_monochrome(original_img):
            return "Night", "#000000"

        # enhance
        enhanced_img = adjust_gamma(original_img, gamma=1.2)

        # YOLO prediction
        results = yolo_model(enhanced_img, verbose=False)
        
        # parse YOLO probs
        yolo_probs_dict = {k: 0.0 for k in CLASSES if k != 'Night' and k != 'Other'}
        
        if results[0].probs is not None:
            for i, conf in enumerate(results[0].probs.data):
                # ensure class name matches casing 
                class_name = results[0].names[i].capitalize()
                if class_name in yolo_probs_dict:
                    yolo_probs_dict[class_name] = float(conf)
        
        # get winnee class
        top_class = max(yolo_probs_dict, key=yolo_probs_dict.get)
        confidence = yolo_probs_dict[top_class]

        # threshold check
        final_label = top_class
        if confidence < CONF_THRESH:
            final_label = 'Other'

        # hex extraction
        roi = extract_color_roi(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        hex_code = get_hex_color(cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

        return final_label, hex_code

    except Exception as e:
        print(f"[Color] Error processing {frame_path}: {e}")
        return "unknown", "#000000"

def color_worker():
    r = get_redis_connection()
    worker_id = os.environ.get('WORKER_ID', 'color_worker_yolo_1')
    
    try:
        yolo = load_model()
        print(f"[Color] YOLO model loaded. Worker started: {worker_id}")
    except Exception as e:
        print(f"[CRITICAL] Failed to load YOLO model: {e}")
        return

    while not shutdown_event.is_set():
        try:
            messages = r.xreadgroup(
                COLOR_GROUP, worker_id,
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
                    
                    print(f"[Color] Processing job: {job_id} ({vehicle_type})")
                    
                    if should_worker_process("color", vehicle_type):
                        try:
                            # PASS YOLO MODEL TO FUNCTION
                            color_name, hex_code = process_color(frame_path, yolo)
                            
                            result = f"{color_name}|{hex_code}"
                            
                            r.xadd(VEHICLE_RESULTS_STREAM, {
                                "job_id": job_id,
                                "vehicle_id": fields.get("vehicle_id"),
                                "worker": "color",
                                "result": result,
                                "status": "ok",
                                "frame_path": frame_path,  
                                "plate_path": plate_path 
                            })
                            print(f"[Color] Completed: {job_id} -> {color_name} ({hex_code})")
                            r.xack(VEHICLE_JOBS_STREAM, COLOR_GROUP, msg_id)
                        except Exception as e:
                            print(f"[Color] Failed for {job_id}: {e}")
                            r.xadd(VEHICLE_RESULTS_STREAM, {
                                "job_id": job_id,
                                "vehicle_id": fields.get("vehicle_id"),
                                "worker": "color",
                                "result": "unknown|#000000",
                                "status": "error",
                                "error": str(e)
                            })
                            r.xack(VEHICLE_JOBS_STREAM, COLOR_GROUP, msg_id)
                    else:
                        print(f"[Color] Skipping {vehicle_type} (cars only)")
                        r.xack(VEHICLE_JOBS_STREAM, COLOR_GROUP, msg_id)
        
        except Exception as e:
            print(f"[Color] Worker error: {e}")
            time.sleep(1)
    
    print("[Color] Shutdown complete.")

if __name__ == "__main__":
    color_worker()