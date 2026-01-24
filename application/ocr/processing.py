from collections import defaultdict
import math
import re
import cv2
from matplotlib import pyplot as plt


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

def show_preprocessing(stages):
    n = len(stages)
    plt.figure(figsize=(15, 3))

    for i, (name, img) in enumerate(stages.items()):
        plt.subplot(1, n, i + 1)
        plt.title(name)
        plt.axis("off")

        if len(img.shape) == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()


INDIAN_PLATE_FORMATS = [
    r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$',        # Standard: KL11AS2509
    r'^\d{2}BH\d{4}[A-Z]{2}$',                # Bharat Series: 22BH1234AB
    r'^[A-Z]{2}\d{2}[A-Z]\d{4}$',             # Old format: KL01A1234
    r'^[A-Z]{2}\d{2}EV\d{4}$',                # Electric Vehicle: KL11EV1234 (green plates)
    r'^[A-Z]{2}\d{2}C\d{4}$',                 # Commercial: KL11C1234
    r'^[A-Z]{2}\d{2}T\d{4}$',                 # Taxi: KL11T1234
]

def is_valid_indian_plate(text: str) -> bool:
    """
    Check if text matches valid Indian license plate format.
    Handles standard, Bharat Series, EV (green), commercial, and taxi formats.
    """
    if not text:
        return False
    
    # Remove common separators
    clean_text = text.replace(' ', '').replace('-', '').upper()
    
    return any(re.match(pattern, clean_text) for pattern in INDIAN_PLATE_FORMATS)


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
    if ocr_score >= 0.98:
        return text  # High confidence, skip correction

    if not text:
        return ""

    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    chars = list(text)

    if len(chars) < 6:
        return text

    aggressive = ocr_score >= 0.75

    digit_to_letter = {
        "0": "O", "1": "I", "2": "Z", "3": "E",
        "4": "A", "5": "S", "6": "G",
        "7": "T", "8": "B"
    }

    letter_to_digit = {
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "L": "1",
        "Z": "2",
        "E": "3",
        "A": "4",
        "S": "5",
        "G": "6",
        "T": "7",
        "B": "8"
    }

    # State code
    for i in range(2):
        if chars[i].isdigit():
            chars[i] = digit_to_letter.get(chars[i], chars[i])

    # RTO code
    for i in range(2, 4):
        if chars[i].isalpha():
            chars[i] = letter_to_digit.get(chars[i], chars[i])

    # Detect if this is an EV plate (green plate format: XX##EV####)
    is_ev_format = len(chars) >= 8 and chars[4:6] == ['E', 'V']
    
    if is_ev_format:
        # EV plates: XX##EV#### - series is always 'EV'
        if len(chars) >= 5 and chars[4].isdigit():
            chars[4] = 'E'  # Force E
        if len(chars) >= 6 and chars[5].isdigit():
            chars[5] = 'V'  # Force V
        series_end = 6
    else:
        # Standard format: Series (force at least 1 letter)
        if len(chars) >= 5 and chars[4].isdigit():
            chars[4] = digit_to_letter.get(chars[4], chars[4])

        # Optional second series letter
        if aggressive and len(chars) >= 6 and chars[5].isdigit():
            chars[5] = digit_to_letter.get(chars[5], chars[5])

        series_end = 5
        if len(chars) >= 6 and chars[5].isalpha():
            series_end = 6

    # Number part
    for i in range(series_end, len(chars)):
        if chars[i].isalpha():
            chars[i] = letter_to_digit.get(chars[i], chars[i])

    # Trim to max length
    chars = chars[:series_end + 4]

    return "".join(chars)

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