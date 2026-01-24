#!/usr/bin/env python3

import os
import re
import time
import signal
import threading
from typing import List, Dict, Any, Optional

import numpy as np
import cv2
import logging

from rapidocr_onnxruntime import RapidOCR

from db_redis.sentinel_redis_config import *

from processing import preprocess_plate, correct_plate_text, rank_plate_candidates

shutdown_event = threading.Event()

def handle_shutdown(signum, frame):
    print(f"\nReceived signal {signum}, shutting down OCR worker gracefully...")
    shutdown_event.set()

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# Initialize RapidOCR once (expensive model load)
try:
    reader = RapidOCR()
    print("RapidOCR reader initialized.")
except Exception as e:
    logging.exception("Failed to initialize RapidOCR: %s", e)
    raise

# Per-segment cutoff used in Calicut code (keep same or tune)
OCR_SEGMENT_THRESHOLD = 0.65

def clean_and_sort_results(results):
    """Helper to sort boxes and clean text (keeps original sentinel logic)."""
    if not results:
        return None

    # sort logic -> first top to bottom, then left-to-right
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

        # skip the blue hologram part
        if clean_part in ["IND", "IN", "ND"]:
            continue

        # strip "IND" prefix
        if clean_part.startswith("IND") and len(clean_part) > 3:
            clean_part = clean_part[3:]

        final_text_parts.append(clean_part)

    return "".join(final_text_parts)

def is_green_plate(image: np.ndarray) -> bool:
    """Detect green plates (EV) using HSV thresholding. Same heuristic as sentinel original."""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pixels = cv2.countNonZero(mask)
        total_pixels = image.shape[0] * image.shape[1]
        ratio = green_pixels / total_pixels if total_pixels > 0 else 0.0
        return ratio > 0.3
    except Exception:
        return False

def assemble_multiline_plate(detections: List[Dict[str, Any]], plate_height: int):
    """
    Assemble multi-line plate text from per-token detections.

    detections: list of dicts with keys:
        text (str), score (float), x_center (float), y_center (float), box
    plate_height: int - pixel height of the cropped plate image

    Returns:
        (assembled_text (str) or None, assembled_score (float) or None)
    """
    if not detections:
        return None, None

    detections = [d for d in detections if d.get("text")]
    if not detections:
        return None, None

    # Sort by y_center (top to bottom)
    detections = sorted(detections, key=lambda d: d["y_center"])

    # Threshold to separate rows: use fraction of plate height
    row_gap_thresh = max(8, plate_height * 0.20)

    rows = []
    current_row = [detections[0]]
    for det in detections[1:]:
        if abs(det["y_center"] - current_row[-1]["y_center"]) <= row_gap_thresh:
            current_row.append(det)
        else:
            rows.append(current_row)
            current_row = [det]
    rows.append(current_row)

    # If more than 2 rows found, merge tiny rows into neighbors
    if len(rows) > 2:
        merged = []
        for r in rows:
            if len(r) == 1 and merged:
                merged[-1].extend(r)
            else:
                merged.append(r)
        rows = merged

    # Sort tokens in each row left->right and join text
    assembled_rows = []
    row_scores = []
    for r in rows:
        r_sorted = sorted(r, key=lambda d: d["x_center"])
        texts = [re.sub(r'[^A-Z0-9]', '', d["text"].upper()) for d in r_sorted]
        assembled_rows.append("".join(texts))
        row_scores.append(sum(d.get("score", 0.8) for d in r_sorted) / max(1, len(r_sorted)))

    if len(assembled_rows) == 1:
        final_text = assembled_rows[0]
        final_score = row_scores[0]
    else:
        final_text = "".join(assembled_rows)  # top row first, bottom row next
        counts = [len(r) for r in rows]
        total_tokens = sum(counts) if sum(counts) > 0 else 1
        final_score = sum(s * c for s, c in zip(row_scores, counts)) / total_tokens

    return final_text, final_score

def run_rapid_on_variant(variant_img: np.ndarray):
    """
    Run RapidOCR reader on a single variant image.
    RapidOCR returns (result_list, img_info) where result_list items look like:
      [box_points, text, score]
    """
    try:
        # RapidOCR typically expects BGR images (OpenCV). If variant_img is RGB, convert.
        variant_for_reader = variant_img
        if len(variant_img.shape) == 3 and variant_img.shape[2] == 3:
            # Heuristic: preprocess_plate returns RGB; convert to BGR for RapidOCR
            try:
                variant_for_reader = cv2.cvtColor(variant_img, cv2.COLOR_RGB2BGR)
            except Exception:
                variant_for_reader = variant_img

        res, _ = reader(variant_for_reader, use_det=True, use_cls=False)
        return res or []
    except Exception as e:
        logging.exception("RapidOCR predict failed on variant: %s", e)
        return []

def process_ocr(frame_path: str, plate_path: str) -> str:
    """
    OCR using preprocessing variants + RapidOCR + Calicut postprocessing & ranker.
    Returns a single plate string or "N/A".
    """
    if not plate_path or not os.path.exists(plate_path):
        print(f"OCR Error: Plate path '{plate_path}' is invalid or does not exist.")
        return "N/A"

    try:
        plate_image = cv2.imread(plate_path)
        if plate_image is None:
            print(f"OCR Error: Failed to read image from {plate_path}. Returning N/A.")
            return "N/A"

        is_ev_plate = is_green_plate(plate_image)
        if is_ev_plate:
            print(f"OCR Info: Detected Green EV Plate: {os.path.basename(plate_path)}")

        # Primary path: use preprocessing and RapidOCR recognition
        try:
            variants = preprocess_plate(plate_image)  # dict {variant_name: rgb_image}
        except Exception as e:
            logging.exception("preprocess_plate failed: %s", e)
            variants = {}

        flat_candidates: List[Dict[str, Any]] = []

        for variant_name, variant_img in variants.items():
            if variant_img is None:
                continue

            ocr_res = run_rapid_on_variant(variant_img)
            if not ocr_res:
                continue

            # Build per-variant detections with centers
            variant_detections: List[Dict[str, Any]] = []
            for det in ocr_res:
                # det format expectation: [box_points, text, score?]
                box = det[0] if len(det) > 0 else None
                text = det[1] if len(det) > 1 else ""
                score = None
                if len(det) > 2:
                    try:
                        score = float(det[2])
                    except Exception:
                        score = None

                ocr_score = float(score) if score is not None else 0.8
                if ocr_score < OCR_SEGMENT_THRESHOLD:
                    continue

                x_center = 0.0
                y_center = 0.0
                if box and isinstance(box, (list, tuple)) and len(box) >= 4:
                    try:
                        xs = [float(p[0]) for p in box]
                        ys = [float(p[1]) for p in box]
                        x_center = sum(xs) / len(xs)
                        y_center = sum(ys) / len(ys)
                    except Exception:
                        x_center, y_center = 0.0, 0.0

                variant_detections.append({
                    "text": text,
                    "score": ocr_score,
                    "x_center": x_center,
                    "y_center": y_center,
                    "box": box,
                })

            # Assemble multi-line result per variant
            assembled_text, assembled_score = assemble_multiline_plate(variant_detections, plate_image.shape[0])
            if assembled_text:
                corrected = correct_plate_text(assembled_text, assembled_score if assembled_score is not None else 0.8)
                flat_candidates.append({
                    "variant": variant_name,
                    "raw_text": assembled_text,
                    "text": corrected,
                    "score": assembled_score if assembled_score is not None else 0.8,
                    "box": None,
                })

        # Rank candidates
        try:
            ranked = rank_plate_candidates(flat_candidates) if flat_candidates else []
        except Exception as e:
            logging.exception("rank_plate_candidates failed: %s", e)
            ranked = []

        best_entry: Optional[Dict[str, Any]] = ranked[0] if ranked else None
        best_plate_text = best_entry.get("plate") if best_entry else None

        # Validate
        if best_plate_text and 4 <= len(best_plate_text) <= 10:
            print(f"OCR Success (RapidOCR+ranker): Found '{best_plate_text}'")
            return best_plate_text

        # Fallback: original simpler RapidOCR pipeline kept for robustness
        # (grayscale->CLAHE->resize->sharpen->denoise and inverted variant)
        print("OCR Info: Ranked result invalid or not found. Trying fallback pipeline...")

        try:
            gray_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(gray_image)

            scale_factor = 2.0
            width = int(enhanced_image.shape[1] * scale_factor)
            height = int(enhanced_image.shape[0] * scale_factor)
            resized_image = cv2.resize(enhanced_image, (width, height), interpolation=cv2.INTER_CUBIC)

            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened_image = cv2.filter2D(resized_image, -1, kernel)
            processed_image = cv2.bilateralFilter(sharpened_image, 11, 17, 17)
            inverted_image = cv2.bitwise_not(processed_image)

            # RapidOCR expects BGR; processed_image is gray -> convert to BGR
            proc_bgr = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
            inv_bgr = cv2.cvtColor(inverted_image, cv2.COLOR_GRAY2BGR)

            def run_proc(img):
                res, _ = reader(img, use_det=True, use_cls=False)
                return clean_and_sort_results(res)

            result = None
            if is_green_plate(plate_image):
                result = run_proc(inv_bgr)
                if result and 4 <= len(result) <= 10:
                    print(f"OCR Success (fallback inverted EV): Found '{result}'")
                    return result
                result = run_proc(proc_bgr)
            else:
                result = run_proc(proc_bgr)
                if result and 4 <= len(result) <= 10:
                    print(f"OCR Success (fallback standard): Found '{result}'")
                    return result
                result = run_proc(inv_bgr)

            if result and 4 <= len(result) <= 10:
                print(f"OCR Success (fallback): Found '{result}'")
                return result
            else:
                print(f"OCR Validation Failed: Final fallback result '{result}' invalid.")
                return "N/A"

        except Exception as e:
            logging.exception("Fallback RapidOCR pipeline failed: %s", e)
            return "N/A"

    except Exception as e:
        logging.exception("An unexpected error occurred during OCR process for %s: %s", plate_path, e)
        return "N/A"

def ocr_worker():
    r = get_redis_connection()
    worker_id = "ocr_worker_rapid_1"

    print(f"[OCR] RapidOCR worker started: {worker_id}")

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

                            # send result (preserve exact sentinel fields)
                            r.xadd(VEHICLE_RESULTS_STREAM, {
                                "job_id": job_id,
                                "vehicle_id": fields.get("vehicle_id"),
                                "worker": "ocr",
                                "result": result,
                                "status": "ok",
                                "frame_path": frame_path,
                                "plate_path": plate_path
                            })

                            log_res = result if result != "N/A" else "N/A"
                            print(f"[OCR] Completed: {job_id} -> {log_res}")

                            r.xack(VEHICLE_JOBS_STREAM, OCR_GROUP, msg_id)

                        except Exception as e:
                            logging.exception("OCR worker failure for job %s: %s", job_id, e)
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
            logging.exception("OCR Worker error: %s", e)
            time.sleep(1)

    print("[OCR] Shutdown complete.")

if __name__ == "__main__":
    ocr_worker()