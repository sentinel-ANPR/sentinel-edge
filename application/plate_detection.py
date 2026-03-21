from typing import Any, List, Optional
import numpy as np

def detect_plate_crops(
    image: np.ndarray,
    plate_detector: Any,
    max_candidates: int = 4,
    min_width: int = 40,
    min_height: int = 12,
) -> List[np.ndarray]:
    if plate_detector is None or image is None or image.size == 0:
        return []

    try:
        results = plate_detector(image, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
        if results[0].boxes.conf is not None:
            confs = results[0].boxes.conf.cpu().numpy().tolist()
        else:
            confs = [0.5] * len(boxes_xyxy)

        order = sorted(range(len(boxes_xyxy)), key=lambda idx: confs[idx], reverse=True)

        h, w = image.shape[:2]
        crops: List[np.ndarray] = []

        for idx in order:
            if len(crops) >= max_candidates:
                break

            x1, y1, x2, y2 = boxes_xyxy[idx]

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue

            ch, cw = crop.shape[:2]
            if cw < min_width or ch < min_height:
                continue

            crops.append(crop)

        return crops
    except Exception:
        return []