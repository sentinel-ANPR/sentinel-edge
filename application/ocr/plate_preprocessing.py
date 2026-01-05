# plate_preprocessing.py

import re
import cv2
from matplotlib import pyplot as plt

class PlatePreprocessor:

    def preprocess_plate(plate):
        """
        Returns multiple OCR-ready RGB variants
        """
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Edge-preserving denoise
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # Sharpen (unsharp mask)
        blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
        sharpen = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

        # Morphology to connect characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        morph = cv2.morphologyEx(sharpen, cv2.MORPH_CLOSE, kernel)

        # Threshold variant
        thresh = cv2.adaptiveThreshold(
            morph, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        _, otsu = cv2.threshold(
            morph, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return  {
            "Sharpen": cv2.cvtColor(sharpen, cv2.COLOR_GRAY2RGB),
            "Morph Close": cv2.cvtColor(morph, cv2.COLOR_GRAY2RGB),
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


    def correct_plate_text(text: str, ocr_score: float) -> str:
        """
        Indian license plate post-processing
        """
        if not text:
            return ""

        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        aggressive = ocr_score >= 0.75

        letter_map = {
            "0": "O", "1": "I", "2": "Z",
            "5": "S", "6": "G", "8": "B"
        }

        digit_map = {
            "O": "0", "Q": "0", "D": "0",
            "I": "1", "Z": "2",
            "S": "5", "B": "8", "G": "6"
        }

        chars = list(text)

        # State code
        for i in range(min(2, len(chars))):
            if chars[i].isdigit():
                chars[i] = letter_map.get(chars[i], chars[i])

        # Number suffix
        for i in range(len(chars) - 4, len(chars)):
            if i >= 0 and chars[i].isalpha():
                chars[i] = digit_map.get(chars[i], chars[i])

        # Middle section
        if aggressive:
            for i in range(2, len(chars) - 4):
                if chars[i].isalpha():
                    chars[i] = digit_map.get(chars[i], chars[i])

        return "".join(chars)