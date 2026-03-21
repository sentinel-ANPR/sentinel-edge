#!/usr/bin/env python3
"""
OCR worker adapted to use the same OCR engine and processing flow as
ItsThareesh/Calicut-Traffic-OCR (PaddleOCR + processing.py helpers).

This version adds a multiline plate assembly helper (`assemble_multiline_plate`)
to group per-token PaddleOCR detections into rows (top/bottom) so scooter
(two-line) plates are handled correctly. For each preprocessing variant we:
 - collect tokens with x_center/y_center
 - assemble rows into a single candidate string per variant
 - apply correct_plate_text and rank_plate_candidates as before
"""

import os
import re
import time
import signal
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import cv2
import logging

from db_redis.sentinel_redis_config import *
from plate_detection import detect_plate_crops
from model_config import resolve_model_path

from processing import preprocess_plate, correct_plate_text, rank_plate_candidates, is_valid_indian_plate

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    logging.exception("Failed to import YOLO for OCR fallback plate detection: %s", e)

try:
    from paddleocr import PaddleOCR
except Exception as e:
    PaddleOCR = None
    logging.exception("Failed to import PaddleOCR: %s", e)

shutdown_event = threading.Event()

def handle_shutdown(signum, frame):
    print(f"\nReceived signal {signum}, shutting down OCR worker gracefully...")
    shutdown_event.set()

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

if PaddleOCR is None:
    raise RuntimeError("PaddleOCR is required but not installed. Install paddleocr to proceed.")

ocr = PaddleOCR(
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True
)

# Optional: reduce PaddleOCR verbosity if it's noisy in your logs
logging.getLogger('ppocr').setLevel(logging.ERROR)
logging.getLogger('PaddleOCR').setLevel(logging.ERROR)
logging.getLogger('paddleocr').setLevel(logging.ERROR)

print("PaddleOCR initialized for OCR worker.")

OCR_SEGMENT_THRESHOLD = 0.65

PLATE_DETECTOR = None
if YOLO is not None:
    try:
        plate_model_path = Path(resolve_model_path("MODEL_PLATE_DETECTOR_PATH", "models/license_plate_detector.pt"))
        if plate_model_path.exists():
            PLATE_DETECTOR = YOLO(str(plate_model_path))
            print(f"OCR fallback plate detector loaded: {plate_model_path}")
        else:
            print(f"OCR fallback plate detector not found at: {plate_model_path}")
    except Exception as e:
        logging.exception("Failed to load fallback plate detector model: %s", e)

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

def clean_and_sort_results_for_fallback(results):
    """
    Keep the original sentinel cleaning/sorting helper (if you ever need an OCR fallback that
    uses the older RapidOCR-style output). This function is provided for parity but not used
    by the primary PaddleOCR path.
    """
    if not results:
        return None

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
        if clean_part in ["IND", "IN", "ND"]:
            continue
        if clean_part.startswith("IND") and len(clean_part) > 3:
            clean_part = clean_part[3:]
        final_text_parts.append(clean_part)

    return "".join(final_text_parts)

def run_paddle_on_variant(variant_img: np.ndarray) -> List[Dict[str, Any]]:
    """
    Run PaddleOCR.predict on a single variant image.
    usage: ocr.predict(ocr_input) -> list-like; take first element.
    We return the results list or an empty list on failure.
    """
    def _is_v2_line(item: Any) -> bool:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return False
        if not isinstance(item[1], (list, tuple)) or len(item[1]) < 1:
            return False
        return True

    def _normalize_results(raw_results: Any) -> List[Dict[str, Any]]:
        if not raw_results:
            return []

        if (
            isinstance(raw_results, list)
            and len(raw_results) > 0
            and isinstance(raw_results[0], dict)
            and ("rec_texts" in raw_results[0] or "dt_boxes" in raw_results[0])
        ):
            return raw_results

        lines = []
        if isinstance(raw_results, list) and len(raw_results) > 0:
            first = raw_results[0]
            if isinstance(first, list) and (len(first) == 0 or _is_v2_line(first[0])):
                lines = first
            elif _is_v2_line(first):
                lines = raw_results

        if not lines:
            return []

        rec_texts: List[str] = []
        rec_scores: List[float] = []
        dt_boxes: List[Any] = []

        for line in lines:
            try:
                box = line[0]
                text_score = line[1]
                text = str(text_score[0])
                score = float(text_score[1]) if len(text_score) > 1 else 0.8
                rec_texts.append(text)
                rec_scores.append(score)
                dt_boxes.append(box)
            except Exception:
                continue

        if not rec_texts:
            return []

        return [{"rec_texts": rec_texts, "rec_scores": rec_scores, "dt_boxes": dt_boxes}]

    try:
        if hasattr(ocr, "predict"):
            raw_results = ocr.predict(variant_img) or []
            return _normalize_results(raw_results)

        if hasattr(ocr, "ocr"):
            raw_results = ocr.ocr(variant_img, cls=False) or []
            return _normalize_results(raw_results)

        raise AttributeError("PaddleOCR instance has neither 'predict' nor 'ocr'")
    except Exception as e:
        logging.exception("PaddleOCR inference failed on variant: %s", e)
        return []

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


def build_variant_detections(
    ocr_result: Dict[str, Any],
    score_threshold: float = OCR_SEGMENT_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Convert PaddleOCR result dict into normalized token detections.
    """
    texts = ocr_result.get("rec_texts", []) or []
    scores = ocr_result.get("rec_scores", []) or []
    boxes = ocr_result.get("dt_boxes", []) or []

    detections: List[Dict[str, Any]] = []
    for i, (text, score) in enumerate(zip(texts, scores)):
        try:
            score_val = float(score)
        except Exception:
            score_val = 0.8

        if score_val < score_threshold:
            continue

        box = boxes[i] if boxes and i < len(boxes) else None
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

        detections.append({
            "text": text,
            "score": score_val,
            "x_center": x_center,
            "y_center": y_center,
            "box": box,
        })

    return detections


def collapse_text_from_ocr_result(
    ocr_result: Dict[str, Any],
    score_threshold: float = OCR_SEGMENT_THRESHOLD,
) -> str:
    """
    Build a simple alphanumeric candidate by concatenating high-confidence tokens.
    """
    texts = ocr_result.get("rec_texts", []) or []
    scores = ocr_result.get("rec_scores", []) or []

    parts: List[str] = []
    for t, s in zip(texts, scores):
        try:
            sval = float(s)
        except Exception:
            sval = 0.8
        if sval >= score_threshold:
            parts.append(re.sub(r'[^A-Z0-9]', '', str(t).upper()))

    return "".join(parts)


def build_fallback_images(plate_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare processed and inverted fallback images (both BGR).
    """
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

    proc_bgr = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    inv_bgr = cv2.cvtColor(inverted_image, cv2.COLOR_GRAY2BGR)
    return proc_bgr, inv_bgr


def is_valid_plate_candidate(text: Optional[str]) -> bool:
    if not text or text == "N/A":
        return False
    clean = re.sub(r'[^A-Z0-9]', '', str(text).upper())
    return is_valid_indian_plate(clean)


def process_plate_image(plate_image: np.ndarray, source_tag: str = "primary") -> str:
    if plate_image is None or plate_image.size == 0:
        return "N/A"

    try:
        is_ev_plate = is_green_plate(plate_image)
        if is_ev_plate:
            print(f"OCR Info: Detected Green EV Plate ({source_tag})")

        best_guess: Optional[str] = None

        try:
            variants = preprocess_plate(plate_image)
        except Exception as e:
            logging.exception("preprocess_plate failed (%s): %s", source_tag, e)
            return "N/A"

        flat_candidates: List[Dict[str, Any]] = []

        for variant_name, variant_img in variants.items():
            if variant_img is None:
                continue

            ocr_results = run_paddle_on_variant(variant_img)
            if not ocr_results:
                continue

            res = ocr_results[0] if isinstance(ocr_results, list) and len(ocr_results) > 0 else {}
            variant_detections = build_variant_detections(res, OCR_SEGMENT_THRESHOLD)

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

        try:
            ranked = rank_plate_candidates(flat_candidates) if flat_candidates else []
        except Exception as e:
            logging.exception("rank_plate_candidates failed (%s): %s", source_tag, e)
            ranked = []

        best_entry: Optional[Dict[str, Any]] = ranked[0] if ranked else None
        best_plate_text = best_entry.get("plate") if best_entry else None
        if best_plate_text:
            best_guess = best_plate_text

        if is_valid_plate_candidate(best_plate_text):
            print(f"OCR Success (PaddleOCR+ranker/{source_tag}): Found '{best_plate_text}'")
            return best_plate_text

        print(f"OCR Info: Ranked result invalid ({source_tag}). Attempting direct PaddleOCR fallback...")

        try:
            proc_bgr, inv_bgr = build_fallback_images(plate_image)

            proc_res = run_paddle_on_variant(proc_bgr)
            if proc_res:
                res = proc_res[0]
                candidate_text = collapse_text_from_ocr_result(res, OCR_SEGMENT_THRESHOLD)
                if candidate_text and not best_guess:
                    best_guess = candidate_text
                if is_valid_plate_candidate(candidate_text):
                    print(f"OCR Success (direct proc/{source_tag}): Found '{candidate_text}'")
                    return candidate_text

            inv_res = run_paddle_on_variant(inv_bgr)
            if inv_res:
                res = inv_res[0]
                candidate_text = collapse_text_from_ocr_result(res, OCR_SEGMENT_THRESHOLD)
                if candidate_text and not best_guess:
                    best_guess = candidate_text
                if is_valid_plate_candidate(candidate_text):
                    print(f"OCR Success (direct inverted/{source_tag}): Found '{candidate_text}'")
                    return candidate_text

        except Exception as e:
            logging.exception("Direct PaddleOCR fallback failed (%s): %s", source_tag, e)

        if best_guess:
            print(f"OCR Fallback ({source_tag}): No valid format candidate; using best match '{best_guess}'.")
            return best_guess

        print(f"OCR Validation Failed ({source_tag}): No valid plate candidate found.")
        return "N/A"

    except Exception as e:
        logging.exception("Unexpected OCR error on plate image (%s): %s", source_tag, e)
        return "N/A"

def process_ocr(frame_path: str, plate_path: str) -> str:
    """
    Primary OCR processing using preprocessing & PaddleOCR recognition.

    Returns:
      - recognized plate string (e.g., "KL11A1234") on success
      - "N/A" on failure / when no valid plate found
    """
    try:
        best_guess: Optional[str] = None

        if plate_path and os.path.exists(plate_path):
            plate_image = cv2.imread(plate_path)
            if plate_image is not None:
                primary_result = process_plate_image(plate_image, source_tag="primary")
                if is_valid_plate_candidate(primary_result):
                    return primary_result
                if primary_result and primary_result != "N/A":
                    best_guess = primary_result
            else:
                print(f"OCR Error: Failed to read image from {plate_path}.")
        else:
            print(f"OCR Info: Plate path '{plate_path}' is invalid or missing. Trying fallback from frame crop.")

        if not frame_path or not os.path.exists(frame_path):
            print(f"OCR Validation Failed: Frame path '{frame_path}' unavailable for fallback.")
            return "N/A"

        vehicle_crop = cv2.imread(frame_path)
        if vehicle_crop is None:
            print(f"OCR Validation Failed: Could not read frame crop from '{frame_path}'.")
            return "N/A"

        fallback_crops = detect_plate_crops(vehicle_crop, PLATE_DETECTOR, max_candidates=4)
        if not fallback_crops:
            print("OCR Validation Failed: No alternate plate candidates detected in frame crop.")
            return "N/A"

        print(f"OCR Info: Trying {len(fallback_crops)} alternate plate candidate(s) from frame crop...")
        for idx, candidate_crop in enumerate(fallback_crops, start=1):
            candidate_result = process_plate_image(candidate_crop, source_tag=f"fallback_{idx}")
            if is_valid_plate_candidate(candidate_result):
                print(f"OCR Success (alternate candidate {idx}): Found '{candidate_result}'")
                return candidate_result
            if candidate_result and candidate_result != "N/A" and not best_guess:
                best_guess = candidate_result

        if best_guess:
            print(f"OCR Fallback: No valid plate format from any crop; using best match '{best_guess}'.")
            return best_guess

        print("OCR Validation Failed: All alternate plate candidates rejected.")
        return "N/A"

    except Exception as e:
        logging.exception("An unexpected error occurred during OCR process for %s: %s", plate_path, e)
        return "N/A"


def ocr_worker():
    r = get_redis_connection()
    worker_id = "ocr_worker_paddle_1"

    print(f"[OCR] PaddleOCR worker started: {worker_id}")

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

                            # Log clearly
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