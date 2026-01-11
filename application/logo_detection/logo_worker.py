import time
import signal
import sys
import threading
import cv2
import numpy as np
import os
from pathlib import Path
from db_redis.sentinel_redis_config import *

shutdown_event = threading.Event()

def handle_shutdown(signum, frame):
    print(f"\nReceived signal {signum}, shutting down Logo worker gracefully...")
    shutdown_event.set()

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# Setup storage path for cropped logos
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGOS_PATH = PROJECT_ROOT / "static" / "logos"
LOGOS_PATH.mkdir(parents=True, exist_ok=True)

# Logo Detection Model Configuration
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models" / "logo"

logo_cfg = MODEL_DIR / "yoloLogo.cfg"
logo_weights = MODEL_DIR / "yoloLogo.weights"
logo_names_file = MODEL_DIR / "yoloLogo.names"

# Load YOLO logo detector once at startup
print("[Logo] Loading YOLO logo detection model...")
try:
    logo_net = cv2.dnn.readNetFromDarknet(str(logo_cfg), str(logo_weights))
    with open(str(logo_names_file), "r") as f:
        logo_classes = [line.strip() for line in f.readlines()]
    
    # Get output layer names
    ln = logo_net.getLayerNames()
    try:
        ln = [ln[i - 1] for i in logo_net.getUnconnectedOutLayers()]
    except TypeError:
        ln = [ln[i[0] - 1] for i in logo_net.getUnconnectedOutLayers()]
    
    output_layers = ln
    print(f"[Logo] Model loaded successfully. {len(logo_classes)} logo classes available.")
except Exception as e:
    print(f"[Logo] ERROR: Failed to load model: {e}")
    logo_net = None
    logo_classes = []
    output_layers = []

def process_logo(frame_path, vehicle_id):
    """
    Detect and crop car logo from keyframe image.
    Returns: (make_name, logo_path)
    """
    if logo_net is None:
        print(f"[Logo] Model not loaded, returning default values")
        return "Unknown", None
    
    # Load the keyframe image
    image = cv2.imread(frame_path)
    if image is None:
        print(f"[Logo] Error: Could not load image from {frame_path}")
        return "Unknown", None
    
    (H, W) = image.shape[:2]
    
    # Create blob and run detection
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    logo_net.setInput(blob)
    layerOutputs = logo_net.forward(output_layers)
    
    boxes = []
    confidences = []
    classIDs = []
    
    # Process detections
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > 0.5:  # Confidence threshold
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    # Apply Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    
    # If logo detected, crop and save it
    if len(idxs) > 0:
        # Take the first (highest confidence) detection
        i = idxs.flatten()[0]
        (x, y, w, h) = boxes[i]
        
        # Ensure coordinates are within bounds
        x = max(0, x)
        y = max(0, y)
        x_end = min(W, x + w)
        y_end = min(H, y + h)
        
        # Crop the logo
        cropped_logo = image[y:y_end, x:x_end]
        
        # Get the logo class name
        logo_make = logo_classes[classIDs[i]]
        
        # Save cropped logo
        logo_filename = f"{vehicle_id}_logo.jpg"
        logo_path = LOGOS_PATH / logo_filename
        cv2.imwrite(str(logo_path), cropped_logo)
        
        print(f"[Logo] Detected {logo_make} (confidence: {confidences[i]:.2f})")
        print(f"[Logo] Saved cropped logo to: {logo_path}")
        
        return logo_make, str(logo_path)
    else:
        print(f"[Logo] No logo detected in {frame_path}")
        return "Unknown", None

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
                            # Process and get both make and logo_path
                            make, logo_path = process_logo(frame_path, vehicle_id)
                            
                            r.xadd(VEHICLE_RESULTS_STREAM, {
                                "job_id": job_id,
                                "vehicle_id": vehicle_id,
                                "worker": "logo",
                                "result": make,  # Car make/brand name
                                "logo_path": logo_path if logo_path else "N/A",  # Path to cropped logo
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
                    else:
                        print(f"[Logo] Skipping {vehicle_type} (not in scope)")
                        r.xack(VEHICLE_JOBS_STREAM, LOGO_GROUP, msg_id)
                        
        except Exception as e:
            print(f"[Logo] Worker error: {e}")
            time.sleep(1)
    
    print("[Logo] Shutdown complete.")

if __name__ == "__main__":
    logo_worker()
