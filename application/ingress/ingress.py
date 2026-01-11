import queue
import threading
import time
import cv2
import os
import numpy as np
import uuid
import datetime
from pathlib import Path
from ultralytics import YOLO
from db_redis.sentinel_redis_config import *
import pytz
from vidgear.gears import CamGear

IST = pytz.timezone('Asia/Kolkata')

# model = YOLO("models/classifier-yolov8n.pt")
model = YOLO("models/classifier-yolov8n_openvino_model/", task="detect")
plate_model = YOLO("models/license_plate_detector_openvino_model/", task="detect")

# new class ids
CLASS_ID_CAR = 0
CLASS_ID_MOTORCYCLE = 1
CLASS_ID_BUS = 2
CLASS_ID_TRUCK = 3
CLASS_ID_AUTO = 4

TRACK_CLASSES = [CLASS_ID_CAR, CLASS_ID_MOTORCYCLE, CLASS_ID_BUS, CLASS_ID_TRUCK, CLASS_ID_AUTO]

# get configuration from environment
LOCATION = os.getenv("LOCATION", "DEFAULT_LOCATION")
RTSP_URL = os.getenv("RTSP_STREAM")
IS_FILE = str(RTSP_URL).strip().lower().endswith(".mp4")
VISUAL_MODE_ENV = os.getenv("VISUAL_MODE", "0") == "1"
# VISUAL_MODE = VISUAL_MODE_ENV
VISUAL_MODE = True

print(f"Ingress started for location: {LOCATION}")
if VISUAL_MODE:
    print("Visual Debug Mode: ENABLED (Window will appear)")
 
if not RTSP_URL:
    print("Error: RTSP_STREAM not set in environment variables.")
    exit(1)

# set up storage paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
DIR_NAME = f"static" 
LOCATION_PATH = PROJECT_ROOT / DIR_NAME
# subfolders
KEYFRAMES_PATH = LOCATION_PATH / "keyframes"
PLATES_PATH = LOCATION_PATH / "plates"

# bufferless capture for rtsp ( mgiht ahve to swtich to vidgear )
class BufferlessCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        # force TCP for RTSP to prevent packet corruption
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        # set buffer size to minimum
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.q = queue.Queue()
        self.running = True
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    # we read frames as soon as they are available - keep only the most recent one
    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait() # discard old frame
                except queue.Empty:
                    pass
            self.q.put(frame) # put new frame

    def read(self):
        try:
            return self.q.get(timeout=1.0) # wait slightly for a frame
        except queue.Empty:
            return None

    def release(self):
        self.running = False
        self.t.join()
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()

#capture with vidgear
class VidgearCapture:
    def __init__(self, name):
        # logging=True helps debug connection issues
        # time_delay=0 prevents artificial delays
        options = {
                    "CAP_PROP_BUFFERSIZE": 0,           
                    "rtsp_transport": "tcp",       
                    "stimeout": "5000000",             
                    "max_delay": "500000",              
                    "fflags": "nobuffer"           
                } 
        self.stream = CamGear(source=name, logging=True, time_delay=0, **options).start()

    def read(self):
        # vidgear returns just the frame, or None if failed
        return self.stream.read()

    def release(self):
        self.stream.stop()
        
    def isOpened(self):
        # vidgear doesn't have isOpened, assume true if running
        return True

# mp4 prcoessing
class StandardCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)

    def read(self):
        # standard read - process every frame
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()
        
    def isOpened(self):
        return self.cap.isOpened()

def ensure_storage_structure():
    # ensure the directory structure exists"""    
    LOCATION_PATH.mkdir(exist_ok=True)
    
    KEYFRAMES_PATH.mkdir(parents=True, exist_ok=True)
    PLATES_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Storage structure initialized: {LOCATION_PATH}")

def save_keyframe_organized(vehicle_crop, vehicle_id):
    #save to static-LOCATION/keyframes/
    try:
        filename = f"{vehicle_id}.jpg"
        file_path = KEYFRAMES_PATH / filename
        
        success = cv2.imwrite(str(file_path), vehicle_crop)
        
        if success:
            # URL path matches folder name: static-LOCATION/keyframes/filename.jpg
            relative_path = f"{DIR_NAME}/keyframes/{filename}"
            return str(file_path), relative_path
        return None, None
    except Exception as e:
        print(f"Error saving keyframe: {e}")
        return None, None
    
def detect_and_save_plate(vehicle_crop, vehicle_id):
    #save to static-LOCATION/plates/
    if plate_model is None: return None, None
    
    try:
        results = plate_model(vehicle_crop, verbose=False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = vehicle_crop[y1:y2, x1:x2]
            
            filename = f"{vehicle_id}_plate.jpg"
            file_path = PLATES_PATH / filename
            
            if cv2.imwrite(str(file_path), plate_crop):
                # URL path matches folder name: static-LOCATION/plates/filename.jpg
                relative_path = f"{DIR_NAME}/plates/{filename}"
                print(f"  - Saved plate: {relative_path}")
                return str(file_path), relative_path
                
        return None, None
    except Exception as e:
        print(f"Plate error: {e}")
        return None, None

# initialize storage structure
ensure_storage_structure()

# connect to Redis
r = get_redis_connection()

# track saved vehicles to avoid duplicates
saved_ids = set()

# select engine based on source type
if IS_FILE:
    print(">> MODE: MP4 File detected. Using Standard Engine.")
    cap = StandardCapture(RTSP_URL)
else:
    print(">> MODE: RTSP Stream detected. Using Bufferless Engine.")
    cap = VidgearCapture(RTSP_URL)

# wait for the first frame to determine dimensions
print("Waiting for stream initialization...")
frame = None
while frame is None:
    frame = cap.read()
    time.sleep(0.1)

FRAME_HEIGHT, FRAME_WIDTH = frame.shape[:2]
print(f"Stream Active: {FRAME_WIDTH}x{FRAME_HEIGHT}")

# resolution
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# define keyframe trigger zone
ZONE_X1 = 0
ZONE_Y1 = 330
ZONE_X2 = 1000
ZONE_Y2 = 700
TRIGGER_ZONE = (ZONE_X1, ZONE_Y1, ZONE_X2, ZONE_Y2)

def publish_job(vehicle_type, organized_path, relative_path, track_id, vehicle_id, plate_path=None, plate_relative_path=None):
    """Publish job with organized file paths"""
    timestamp = datetime.datetime.now(IST)
    job_id = f"{vehicle_type}_{track_id}_{vehicle_id.split('_')[0]}"  
    
    payload = {
        "job_id": job_id,
        "vehicle_id": vehicle_id, 
        "vehicle_type": vehicle_type,
        "frame_path": organized_path,   
        "plate_path": plate_path if plate_path else "N/A",   
        "frame_url": relative_path,  
        "plate_url": plate_relative_path if plate_relative_path else "N/A",
        "timestamp": timestamp.isoformat(),
        "location": LOCATION
    }
    
    r.xadd(VEHICLE_JOBS_STREAM, payload)
    print(f"Published job: {job_id} (Vehicle ID: {vehicle_id}) @ {LOCATION}")
    print(f"  Keyframe stored: {relative_path}")

# main loop
frame_num = 0
print("Starting vehicle detection...")

try:
    while True:
        # get latest frame from the selected engine
        raw_frame = cap.read()
        
        # if frame is None (stream ended or connection lost)
        if raw_frame is None:
            if IS_FILE:
                print("End of video file.")
                break 
            else:
                pass
                continue

        # resize it 
        frame = cv2.resize(raw_frame, (TARGET_WIDTH, TARGET_HEIGHT))

        # YOLO Tracking (with openvino now)
        # ultralytics handles the OpenVINO backend automatically when passed the folder
        # persist=True is crucial for ID tracking
        results = model.track(frame, classes=TRACK_CLASSES, verbose=False, tracker="bytetrack.yaml", persist=True)
        
        # process detections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.int().cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for i, box in enumerate(boxes):
                track_id = track_ids[i]
                
                # check trigger zone logic
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = y2 
                
                # zone Check
                if (ZONE_X1 < cx < ZONE_X2) and (ZONE_Y1 < cy < ZONE_Y2):
                    if track_id not in saved_ids:
                        saved_ids.add(track_id)
                        
                        class_id = class_ids[i]
                        vehicle_type = model.names[class_id]
                        
                        # apply padding logic
                        if class_id == CLASS_ID_MOTORCYCLE:
                            h = y2 - y1
                            w = x2 - x1
                            y1 = max(0, y1 - int(h * 3.5))
                            x1 = max(0, x1 - int(w * 0.2))
                            x2 = min(FRAME_WIDTH, x2 + int(w * 0.2))

                        vehicle_crop = frame[y1:y2, x1:x2]
                        
                        if vehicle_crop.size > 0:
                            # ID generation
                            ts_str = datetime.datetime.now(IST).strftime("%Y%m%d_%H%M%S")
                            uid = uuid.uuid4().hex[:8]
                            vehicle_id = f"{uid}_{ts_str}_{vehicle_type}_{LOCATION}"

                            print(f"Captured {vehicle_type} (ID: {track_id})")
                            
                            # saving & publishing
                            org_path, rel_path = save_keyframe_organized(vehicle_crop, vehicle_id)
                            if org_path:
                                p_path, p_rel_path = detect_and_save_plate(vehicle_crop, vehicle_id)
                                publish_job(vehicle_type, org_path, rel_path, track_id, vehicle_id, p_path, p_rel_path)

        # visualization
        if VISUAL_MODE:
            # draw Zone
            cv2.rectangle(frame, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (0, 255, 0), 2)
            
            # draw Boxes
            if results[0].boxes is not None and results[0].boxes.id is not None:
                vis_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                vis_ids = results[0].boxes.id.int().cpu().numpy()
                for i, box in enumerate(vis_boxes):
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    cv2.putText(frame, str(vis_ids[i]), (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            display = cv2.resize(frame, (1280, 720))
            cv2.imshow("Sentinel Ingress", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("Stopping...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Ingress stopped.")