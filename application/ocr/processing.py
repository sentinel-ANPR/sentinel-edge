from collections import defaultdict
import math
import re
import cv2
from matplotlib import pyplot as plt
from plate_syntax_corrector import build_default_corrector


_SYNTAX_CORRECTOR = build_default_corrector()


def preprocess_plate(plate):
    """
    Returns multiple OCR-ready RGB variants
    """
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Edge-preserving denoise (BEFORE CLAHE to avoid filtering noise)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)  # Lighter filter (5px kernel)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Sharpen (unsharp mask)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
    sharpen = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    # Threshold variant
    thresh = cv2.adaptiveThreshold(
        sharpen, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    _, otsu = cv2.threshold(
        sharpen, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return  {
        "Sharpen": cv2.cvtColor(sharpen, cv2.COLOR_GRAY2RGB),
        "Adaptive Threshold": cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB),
        "Otsu Threshold": cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB),
    }

def is_valid_indian_plate(text: str) -> bool:
    """
    Check if text matches valid Indian license plate format.
    Handles standard, Bharat Series, EV (green), commercial, and taxi formats.
    """
    return _SYNTAX_CORRECTOR.is_valid_indian_plate(text)


def correct_plate_text(text: str, ocr_score: float) -> str:
    """
    Indian License Plate Post-Processing
    Formats:
        XX##X####      - Standard (e.g., KL11A2509)
        XX##XX####     - Standard (e.g., KL11AS2509)
        XX##EV####     - Electric Vehicle (green plates)
        XX##C####      - Commercial
        ##BH####XX     - Bharat Series
    """
    return _SYNTAX_CORRECTOR.correct(text, ocr_score)

VARIANT_WEIGHTS = {
    "Otsu Threshold": 1.0,
    "Adaptive Threshold": 0.9,
    "Sharpen": 0.7
}

def rank_plate_candidates(candidates):
    """
    Rank plate candidates using improved scoring:
    - Format validation bonus (huge boost)
    - sqrt(count) for diminishing returns on variant agreement
    - Consistency scoring (penalize wide score variance)
    - Length validation
    """
    grouped = defaultdict(list)

    # Group by corrected text
    for c in candidates:
        grouped[c["text"]].append(c)

    ranked = []

    for plate_text, items in grouped.items():
        count = len(items)
        scores = [i["score"] for i in items]
        avg_score = sum(scores) / count
        max_score = max(scores)
        min_score = min(scores)
        
        # Format validation bonus (critical for accuracy)
        format_valid = is_valid_indian_plate(plate_text)
        format_bonus = 5.0 if format_valid else 0.0
        syntax_bonus = _SYNTAX_CORRECTOR.syntax_score(plate_text)
        
        # Length penalty (Indian plates are 9-10 chars)
        length = len(plate_text)
        length_penalty = 0.0
        if length < 9 or length > 10:
            length_penalty = 2.0
        
        # Consistency bonus (narrow score range = reliable)
        consistency = 1.0 - (max_score - min_score)
        
        # Variant weighting
        variant_weight_sum = sum(
            VARIANT_WEIGHTS.get(i["variant"], 0.5) for i in items
        )
        
        # Improved scoring formula
        final_score = (
            format_bonus +                    # +5 if valid format (dominant)
            1.5 * syntax_bonus +             # Position-aware syntax confidence
            2.0 * math.sqrt(count) +          # Diminishing returns on count
            4.0 * avg_score +                 # Quality matters most
            2.0 * consistency +               # Reward consistent scores
            0.5 * variant_weight_sum -
            length_penalty
        )

        ranked.append({
            "plate": plate_text,
            "final_score": round(final_score, 4),
            "valid_format": format_valid,
            "count": count,
            "avg_ocr_score": round(avg_score, 3),
            "max_ocr_score": round(max_score, 3),
            "min_ocr_score": round(min_score, 3),
            "consistency": round(consistency, 3),
            "length": length,
            "sources": items
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked