from datetime import datetime
import json
import cv2
from ultralytics import YOLO
from plate_preprocessing import PlatePreprocessor
from ocr_engine import PaddleOCREngine

# Load models
model = YOLO("./license_plate_detector.pt")
ocr = PaddleOCREngine()

# ✅ USE YOUR ACTUAL IMAGE FILENAME HERE
image_path = "/test/test_plate.jpg"  # ← CHANGE THIS TO YOUR IMAGE
img = cv2.imread(image_path)

if img is None:
    raise RuntimeError(f"❌ Image not found: {image_path}")

print(f"✅ Loaded image: {img.shape}")

# Run YOLO detection
results = model(img, conf=0.4)

output = {
    "image": image_path,
    "timestamp": datetime.now().isoformat(),
    "candidates": []
}

plate_count = 0
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate = img[y1:y2, x1:x2]
        if plate.size == 0:
            continue

        plate_count += 1
        print(f"\n🔍 Processing detected plate #{plate_count} (crop size: {plate.shape})")

        # DEBUG: Save cropped plate
        cv2.imwrite(f"debug_plate_{plate_count}.jpg", plate)

        # Preprocess & OCR
        variants = PlatePreprocessor.preprocess_plate(plate)
        PlatePreprocessor.show_preprocessing(variants)

        for variant_id, ocr_input in variants.items():
            ocr_result = ocr.predict(ocr_input)
            if not ocr_result or not ocr_result[0]:
                continue

            texts = ocr_result[0].get("rec_texts", [])
            scores = ocr_result[0].get("rec_scores", [])
            for raw_text, score in zip(texts, scores):
                if score < 0.5:
                    continue
                corrected = PlatePreprocessor.correct_plate_text(raw_text, score)
                output["candidates"].append({
                    "variant": variant_id,
                    "raw_text": raw_text,
                    "corrected_text": corrected,
                    "ocr_score": round(float(score), 3)
                })
                print(f"  → {variant_id}: '{raw_text}' → '{corrected}' (score: {score:.2f})")

# --- PRINT BEST RESULT ---
if output["candidates"]:
    best = max(output["candidates"], key=lambda x: x["ocr_score"])
    print(f"\n🎯 FINAL RESULT: License Plate = {best['corrected_text']}")
else:
    print("\n❌ No valid text detected on any plate.")

# Save full logs
image_path_fmt = image_path.replace('/', '_').replace('.', '_')
json_path = f"logs/ocr_candidates_{image_path_fmt}.json"
import os
os.makedirs("logs", exist_ok=True)
with open(json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n[INFO] Full results saved to {json_path}")