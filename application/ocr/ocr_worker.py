import re
import time
import signal
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

def process_ocr(frame_path, plate_path):
    if not plate_path or not os.path.exists(plate_path):
        print(f"OCR Error: Plate path '{plate_path}' is invalid or does not exist.")
        return None

    try:
        plate_image = cv2.imread(plate_path)
        
        if not plate_path or not os.path.exists(plate_path):
            print(f"OCR Error: Plate path '{plate_path}' is invalid or does not exist.")
            return "N/A"
        
        # preprocessing
        gray_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

        #appyl clahe
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray_image)

        # tuning to resize
        scale_factor = 2.0
        width = int(enhanced_image.shape[1] * scale_factor)
        height = int(enhanced_image.shape[0] * scale_factor)
        resized_image = cv2.resize(enhanced_image, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Sharpening filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened_image = cv2.filter2D(resized_image, -1, kernel)

        processed_image = cv2.resize(enhanced_image, (width, height), interpolation=cv2.INTER_CUBIC)

        # mild denoise
        processed_image = cv2.bilateralFilter(processed_image, 11, 17, 17)

        # RapidOCR returns (result, elapse)
        results, _ = reader(processed_image, use_det=True, use_cls=False)

        if not results:
            print("OCR Info: RapidOCR found no text.")
            return "N/A"
        
        # sorting logic
        # RapidOCR result format-> [[top_left, top_right, bottom_right, bottom_left], text, score]
        # sort detected boxes by position:
        # priority 1: vertical position (Y) -> read top line first
        # priority 2: horizontal position (X) -> read left-to-right
        
        # group Y values into "lines" (20px tolerance)
        def sort_key(res):
            box = res[0]
            x_left = box[0][0]
            y_top = box[0][1]
            # integer divide Y by 20 so slightly misaligned text counts as "same line"
            line_bucket = int(y_top // 20) 
            return (line_bucket, x_left)

        sorted_results = sorted(results, key=sort_key)

        # 
        final_text_parts = []
        for res in sorted_results:
            text = res[1]
            score = res[2]
            
            clean_part = re.sub(r'[^A-Z0-9]', '', text.upper())

            # skip the hologram "IND" if detected as a separate block
            if clean_part in ["IND", "IN", "ND"]:
                continue
            
            # if "IND" is attached to text, strip
            if clean_part.startswith("IND") and len(clean_part) > 3:
                clean_part = clean_part[3:]
            
            final_text_parts.append(clean_part)

        # join ordered parts
        full_text = "".join(final_text_parts)

        # check length
        if 4 <= len(full_text) <= 10:
            print(f"OCR Success: Found '{full_text}' from {os.path.basename(plate_path)}")
            return full_text
        else:
            print(f"OCR Validation Failed: '{full_text}' failed length/format check. Returning N/A.")
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
                            r.xadd(VEHICLE_RESULTS_STREAM, {
                                "job_id": job_id,
                                "vehicle_id": fields.get("vehicle_id"),
                                "worker": "ocr",
                                "result": result,
                                "status": "ok",
                                "frame_path": frame_path, 
                                "plate_path": plate_path 
                            })
                            print(f"[OCR] Completed: {job_id} -> {result}")
                            r.xack(VEHICLE_JOBS_STREAM, OCR_GROUP, msg_id)
                        except Exception as e:
                            print(f"[OCR] Failed for {job_id}: {e}")
                            r.xadd(VEHICLE_RESULTS_STREAM, {
                                "job_id": job_id,
                                "worker": "ocr",
                                "result": "",
                                "status": "error",
                                "error": str(e),
                                "plate_path": plate_path
                            })
                    else:
                        print(f"[OCR] Skipping {vehicle_type} (not in scope)")
                        r.xack(VEHICLE_JOBS_STREAM, OCR_GROUP, msg_id)
                        
        except Exception as e:
            print(f"[OCR] Worker error: {e}")
            time.sleep(1)
    
    print("[OCR] Shutdown complete.")

if __name__ == "__main__":
    ocr_worker()