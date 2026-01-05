
from paddleocr import PaddleOCR


class PaddleOCREngine:

    def __init__(self):
        self.ocr = PaddleOCR(
            lang="en",
            use_textline_orientation=True,
        )

    def predict(self, image):
        return self.ocr.predict(image)