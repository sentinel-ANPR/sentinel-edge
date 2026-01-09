import re
import time
import signal
import bcrypt
import threading
from rapidocr_onnxruntime import RapidOCR
import numpy as np
import cv2

from db_redis.sentinel_redis_config import *

shutdown_event = threading.Event()

def handle_shutdown(signum, frame):
    print(f"\nReceived signal {signum}, shutting down OCR worker gracefully...")
    shutdown_event.set()

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# initialzise RapidOCR
reader = RapidOCR()
print("RapidOCR reader initialized.")

def clean_and_sort_results(results):
    """Helper to sort boxes and clean text"""
    if not results:
        return None

    # sort logic -> first top to bottom, then left-to-tight
    def sort_key(res):
        box = res[0]
        x_left = box[0][0]
        y_top = box[0][1]
        line_bucket = int(y_top // 20) 
        return (line_bucket, x_left)

    sorted_results = sorted(results, key=sort_key)

    final_text_parts = []
    for res in sorted_results:
        text = res[1]
        clean_part = re.sub(r'[^A-Z0-9]', '', text.upper())

        # skip the bvlue hologram part
        if clean_part in ["IND", "IN", "ND"]:
            continue
        
        # strip "IND" prefix
        if clean_part.startswith("IND") and len(clean_part) > 3:
            clean_part = clean_part[3:]
        
        final_text_parts.append(clean_part)

    return "".join(final_text_parts)

# we gotta chakc if hte palte is greem so if >30% pixelsa re kinda green we use it
def is_green_plate(image):
    try:
        # convert BGR to HSV 
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # define range for green in HSV
        # opencv hue range is 0-179, green is kinda 60.
        # look for H: 35-90, and moderate saturation/value to avoid white/black detection
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([90, 255, 255])

        # create a mask of green pixels
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # calculate ratio of green pixels
        green_pixels = cv2.countNonZero(mask)
        total_pixels = image.shape[0] * image.shape[1]
        
        ratio = green_pixels / total_pixels
        
        if ratio > 0.3:
            return True
        return False
    except:
        return False
    
def process_ocr(frame_path, plate_path):
    if not plate_path or not os.path.exists(plate_path):
        print(f"OCR Error: Plate path '{plate_path}' is invalid or does not exist.")
        return "N/A"

    try:
        plate_image = cv2.imread(plate_path)
        
        if plate_image is None:
            print(f"OCR Error: Failed to read image from {plate_path}. Returning N/A.")
            return "N/A"
        
        # check if ev plate
        is_ev_plate = is_green_plate(plate_image)
        if is_ev_plate:
            print(f"OCR Info: Detected Green EV Plate: {os.path.basename(plate_path)}")

        # preprocess
        gray_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray_image)

        # resize
        scale_factor = 2.0
        width = int(enhanced_image.shape[1] * scale_factor)
        height = int(enhanced_image.shape[0] * scale_factor)
        resized_image = cv2.resize(enhanced_image, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened_image = cv2.filter2D(resized_image, -1, kernel)

        # denoise
        processed_image = cv2.bilateralFilter(sharpened_image, 11, 17, 17)

        # inverted iamge for ev
        inverted_image = cv2.bitwise_not(processed_image)

        # run both versions
        def run_standard():
            res, _ = reader(processed_image, use_det=True, use_cls=False)
            return clean_and_sort_results(res)

        def run_inverted():
            res, _ = reader(inverted_image, use_det=True, use_cls=False)
            return clean_and_sort_results(res)

        result = None

        if is_ev_plate:
            # priority 1 -> inverted
            result = run_inverted()
            if result and 4 <= len(result) <= 10:
                print(f"OCR Success: Found '{result}' (EV Priority)")
                return result
            
            # fallback to standard if inverted failed completely
            print("OCR Info: Inverted EV read failed, trying standard...")
            result = run_standard()
        else:
            # priority 1 -> standard
            result = run_standard()
            if result and 4 <= len(result) <= 10:
                print(f"OCR Success: Found '{result}' (Standard Priority)")
                return result
            
            # fallback to inverted
            print("OCR Info: Standard read failed, trying inverted...")
            result = run_inverted()

        # final validation
        if result and 4 <= len(result) <= 10:
            print(f"OCR Success: Found '{result}' (Fallback)")
            return result
        else:
            print(f"OCR Validation Failed: Final result '{result}' invalid.")
            return "N/A"

    except Exception as e:
        print(f"An unexpected error occurred during OCR process for {plate_path}: {e}")
        return "N/A"

def ocr_worker():
    r = get_redis_connection()
    worker_id = "ocr_worker_1"
    
    print(f"[OCR] Worker started: {worker_id}")
    
    while not shutdown_event.is_set():
        try:
            messages = r.xreadgroup(
                OCR_GROUP, worker_id, 
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
                    
                    print(f"[OCR] Processing job: {job_id} ({vehicle_type})")
                    
                    if should_worker_process("ocr", vehicle_type):
                        try:
                            result = process_ocr(frame_path, plate_path)
                            
                            # send result
                            r.xadd(VEHICLE_RESULTS_STREAM, {
                                "job_id": job_id,
                                "vehicle_id": fields.get("vehicle_id"),
                                "worker": "ocr",
                                "result": result,
                                "status": "ok",
                                "frame_path": frame_path, 
                                "plate_path": plate_path 
                            })
                            
                            # Log clearly
                            log_res = result if result != "N/A" else "N/A"
                            print(f"[OCR] Completed: {job_id} -> {log_res}")
                            
                            r.xack(VEHICLE_JOBS_STREAM, OCR_GROUP, msg_id)

                        except Exception as e:
                            print(f"[OCR] Failed for {job_id}: {e}")
                            r.xadd(VEHICLE_RESULTS_STREAM, {
                                "job_id": job_id,
                                "worker": "ocr",
                                "result": "N/A",
                                "status": "error",
                                "error": str(e),
                                "frame_path": frame_path,
                                "plate_path": plate_path
                            })
                            r.xack(VEHICLE_JOBS_STREAM, OCR_GROUP, msg_id)
                    else:
                        print(f"[OCR] Skipping {vehicle_type} (not in scope)")
                        r.xack(VEHICLE_JOBS_STREAM, OCR_GROUP, msg_id)
                        
        except Exception as e:
            print(f"[OCR] Worker error: {e}")
            time.sleep(1)
    
    print("[OCR] Shutdown complete.")

if __name__ == "__main__":
    ocr_worker()