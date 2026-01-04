import time
import signal
import sys
import threading
import cv2
import numpy as np
import os
import joblib
from collections import Counter
from sklearn.cluster import KMeans
from ultralytics import YOLO
from db_redis.sentinel_redis_config import *

YOLO_MODEL_PATH = "models/colour-yolo.pt"
SVM_MODEL_PATH = "models/svm_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/encoder.pkl"

# weights
W_CLS = 0.65  # YOLO Weight
W_SVM = 0.35  # SVM Weight
CONF_THRESH = 0.55  # if final score is lower it goes to Other
BOOST_VAL = 0.05    # boost if both models agree
DIFF_VAL = 0.15

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
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype("float32") / 255.0
    mean_s = S.mean()
    std_s = S.std()
    return (mean_s < mean_thresh) and (std_s < std_thresh)

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

    kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
    kmeans.fit(pixels)
    
    counts = Counter(kmeans.labels_)
    center = kmeans.cluster_centers_[counts.most_common(1)[0][0]]
    
    rgb = tuple(center.astype(int))
    return "#{:02x}{:02x}{:02x}".format(*rgb)

# feature extacrtioin for svm
# we need to get teh 32 features which match the svm model
def extract_svm_features(image):
    h, w, _ = image.shape
    crop = image[int(h*0.50):int(h*0.75), int(w*0.35):int(w*0.65)]
    if crop.size == 0: crop = image
    
    # preprocess
    crop = cv2.GaussianBlur(crop, (5, 5), 0)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    s = cv2.multiply(s, 1.5) # boost saturation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)

    # features
    hist_h = cv2.normalize(cv2.calcHist([h], [0], None, [12], [0, 180]), None).flatten()
    hist_s = cv2.normalize(cv2.calcHist([s], [0], None, [4], [0, 256]), None).flatten()
    hist_v = cv2.normalize(cv2.calcHist([v], [0], None, [8], [0, 256]), None).flatten()
    
    mean_h, std_h = cv2.meanStdDev(h)
    mean_s, std_s = cv2.meanStdDev(s)
    mean_v, std_v = cv2.meanStdDev(v)
    
    p90_s = np.percentile(s, 90)
    p90_v = np.percentile(v, 90)

    features = np.concatenate([
        hist_h, hist_s, hist_v, 
        mean_h.flatten(), std_h.flatten(), 
        mean_s.flatten(), std_s.flatten(), 
        mean_v.flatten(), std_v.flatten(), 
        [p90_s, p90_v]
    ])
    return features.reshape(1, -1)

def load_all_models():
    print("[INFO] Loading YOLO...")
    yolo = YOLO(YOLO_MODEL_PATH)
    
    print("[INFO] Loading SVM Bundle...")
    svm = joblib.load(SVM_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    
    return yolo, svm, scaler, encoder

# here we do the hybrid logic with yolo and svm. if they agree, we boost the confidencem if they disagree adn we have a differenc ein 10 in the confidence, we give it to the more conf model. otherwise we give pref to yolo 
# returns: (color_name, hex_code)
def process_color(frame_path, yolo_model, svm_model, scaler, encoder):
    try:
        original_img = cv2.imread(frame_path)
        if original_img is None:
            return "unknown", "#000000"

        # monochrome check
        if is_monochrome(original_img):
            return "Night", "#000000"

        # enhance
        enhanced_img = adjust_gamma(original_img, gamma=1.2)

        # SVM prediction
        features = extract_svm_features(enhanced_img)
        features_scaled = scaler.transform(features)
        svm_probs_raw = svm_model.predict_proba(features_scaled)[0]
        
        svm_classes = encoder.classes_
        svm_probs_dict = {cls: prob for cls, prob in zip(svm_classes, svm_probs_raw)}
        svm_top_class = svm_classes[np.argmax(svm_probs_raw)]
        svm_score = svm_probs_raw.max()

        # YOLO prediction
        results = yolo_model(enhanced_img, verbose=False)
        yolo_probs_dict = {k: 0.0 for k in CLASSES if k != 'Night' and k != 'Other'}
        
        if results[0].probs is not None:
            for i, conf in enumerate(results[0].probs.data):
                class_name = results[0].names[i].capitalize()
                if class_name in yolo_probs_dict:
                    yolo_probs_dict[class_name] = float(conf)
        
        yolo_top_class = max(yolo_probs_dict, key=yolo_probs_dict.get)
        yolo_score = yolo_probs_dict[yolo_top_class]

        # hybrid logic
        final_label = "Other"
        best_conf = 0.0

        if svm_top_class == yolo_top_class:
            # agreement = boost
            raw_avg = (svm_score * W_SVM) + (yolo_score * W_CLS)
            best_conf = min(raw_avg + BOOST_VAL, 1.0)
            final_label = svm_top_class
        else:
            # no agree :(
            diff = abs(svm_score - yolo_score)
            if diff < DIFF_VAL:
                # close call -> YOLO
                final_label = yolo_top_class
                best_conf = yolo_score
            else:
                # clear winner -> trust highest
                if svm_score > yolo_score:
                    final_label = svm_top_class
                    best_conf = svm_score
                else:
                    final_label = yolo_top_class
                    best_conf = yolo_score

        # threshold check
        if best_conf < CONF_THRESH:
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
    worker_id = os.environ.get('WORKER_ID', 'color_worker_1')
    
    try:
        yolo, svm, scaler, encoder = load_all_models()
        print(f"[Color] Models loaded successfully. Worker started: {worker_id}")
    except Exception as e:
        print(f"[CRITICAL] Failed to load models: {e}")
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
                            # PASS LOADED MODELS TO FUNCTION
                            color_name, hex_code = process_color(frame_path, yolo, svm, scaler, encoder)
                            
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