import re
from typing import Dict, List, Optional, Tuple


class PlateSyntaxCorrector:
    VALID_STATE_CODES = {
        "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN",
        "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
        "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
        "SK", "TN", "TR", "TS", "UK", "UP", "WB",
    }

    PLATE_PATTERNS = {
        "standard_1": re.compile(r"^[A-Z]{2}\d{2}[A-Z]\d{4}$"),
        "standard_2": re.compile(r"^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$"),
        "ev": re.compile(r"^[A-Z]{2}\d{2}EV\d{4}$"),
        "commercial": re.compile(r"^[A-Z]{2}\d{2}C\d{4}$"),
        "taxi": re.compile(r"^[A-Z]{2}\d{2}T\d{4}$"),
        "bharat": re.compile(r"^\d{2}BH\d{4}[A-Z]{2}$"),
    }

    DIGIT_TO_LETTER = {
        "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A",
        "5": "S", "6": "G", "7": "T", "8": "B", "9": "G",
    }

    LETTER_TO_DIGIT = {
        "O": "0", "Q": "0", "D": "0", "U": "0",
        "I": "1", "L": "1", "T": "1", "|": "1",
        "Z": "2",
        "E": "3",
        "A": "4",
        "S": "5",
        "G": "6",
        "Y": "7",
        "B": "8",
    }

    def __init__(self, preferred_state: str = "KL"):
        self.preferred_state = (preferred_state or "KL").upper()

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def _to_letter(self, ch: str) -> str:
        if not ch:
            return ""
        if ch.isalpha():
            return ch
        return self.DIGIT_TO_LETTER.get(ch, ch)

    def _to_digit(self, ch: str) -> str:
        if not ch:
            return ""
        if ch.isdigit():
            return ch
        return self.LETTER_TO_DIGIT.get(ch, ch)

    def _plate_distance(self, src: str, dst: str) -> int:
        shared = min(len(src), len(dst))
        diffs = sum(1 for i in range(shared) if src[i] != dst[i])
        diffs += abs(len(src) - len(dst))
        return diffs

    def _state_cost(self, observed: str, candidate: str) -> float:
        score = 0.0
        for i in range(2):
            obs = observed[i] if i < len(observed) else ""
            tar = candidate[i]
            if obs == tar:
                continue
            if self._to_letter(obs) == tar:
                score += 0.25
            else:
                score += 1.0
        if candidate == self.preferred_state:
            score -= 0.15
        return score

    def _correct_state(self, observed: str) -> str:
        observed = self.normalize(observed)[:2]
        if len(observed) < 2:
            return self.preferred_state
        if observed in self.VALID_STATE_CODES:
            return observed

        best = None
        best_score = float("inf")
        for code in self.VALID_STATE_CODES:
            cost = self._state_cost(observed, code)
            if cost < best_score:
                best_score = cost
                best = code

        return best or self.preferred_state

    def _correct_standard(self, raw: str, series_len: int) -> Optional[str]:
        if len(raw) < 4 + series_len + 4:
            return None

        state = self._correct_state(raw[:2])
        rto_raw = raw[2:4]
        middle_raw = raw[4:-4]
        number_raw = raw[-4:]

        if len(middle_raw) < series_len:
            return None

        series_raw = middle_raw[:series_len]

        rto = "".join(self._to_digit(ch) for ch in rto_raw)
        series = "".join(self._to_letter(ch) for ch in series_raw)
        number = "".join(self._to_digit(ch) for ch in number_raw)

        if not (len(rto) == 2 and rto.isdigit()):
            return None
        if not (len(series) == series_len and series.isalpha()):
            return None
        if not (len(number) == 4 and number.isdigit()):
            return None

        return f"{state}{rto}{series}{number}"

    def _correct_marker_plate(self, raw: str, marker: str) -> Optional[str]:
        marker = marker.upper()
        marker_len = len(marker)
        if len(raw) < 4 + marker_len + 4:
            return None

        state = self._correct_state(raw[:2])
        rto_raw = raw[2:4]
        number_raw = raw[-4:]

        rto = "".join(self._to_digit(ch) for ch in rto_raw)
        number = "".join(self._to_digit(ch) for ch in number_raw)

        if not (len(rto) == 2 and rto.isdigit()):
            return None
        if not (len(number) == 4 and number.isdigit()):
            return None

        return f"{state}{rto}{marker}{number}"

    def _correct_bh(self, raw: str) -> Optional[str]:
        if len(raw) < 10:
            return None

        prefix_raw = raw[:2]
        number_raw = raw[4:8] if len(raw) >= 8 else ""
        suffix_raw = raw[-2:]

        prefix = "".join(self._to_digit(ch) for ch in prefix_raw)
        suffix = "".join(self._to_letter(ch) for ch in suffix_raw)
        number = "".join(self._to_digit(ch) for ch in number_raw)

        if not (len(prefix) == 2 and prefix.isdigit()):
            return None
        if not (len(number) == 4 and number.isdigit()):
            return None
        if not (len(suffix) == 2 and suffix.isalpha()):
            return None

        return f"{prefix}BH{number}{suffix}"

    def _generate_candidates(self, raw: str) -> List[str]:
        out = set()
        if not raw:
            return []

        out.add(raw)

        for series_len in (1, 2):
            cand = self._correct_standard(raw, series_len)
            if cand:
                out.add(cand)

        for marker in ("EV", "C", "T"):
            cand = self._correct_marker_plate(raw, marker)
            if cand:
                out.add(cand)

        bh = self._correct_bh(raw)
        if bh:
            out.add(bh)

        return list(out)

    def is_valid_indian_plate(self, text: str) -> bool:
        clean = self.normalize(text)
        if len(clean) < 8:
            return False
        return any(pattern.match(clean) for pattern in self.PLATE_PATTERNS.values())

    def syntax_score(self, text: str) -> float:
        clean = self.normalize(text)
        if not clean:
            return -5.0

        if self.PLATE_PATTERNS["bharat"].match(clean):
            return 4.0

        if len(clean) < 7:
            return -3.0

        score = 0.0

        if len(clean) >= 2:
            state = clean[:2]
            score += 2.5 if state in self.VALID_STATE_CODES else -1.0
            if state == self.preferred_state:
                score += 0.5

        if len(clean) >= 4:
            rto = clean[2:4]
            score += 1.0 if rto.isdigit() else -1.0

        if len(clean) >= 8:
            last4 = clean[-4:]
            score += 1.5 if last4.isdigit() else -2.0

        if any(pattern.match(clean) for pattern in self.PLATE_PATTERNS.values()):
            score += 4.0

        if len(clean) > 10:
            score -= 1.5
        if len(clean) < 8:
            score -= 2.0

        return score

    def correct(self, text: str, ocr_score: float) -> str:
        raw = self.normalize(text)
        if not raw:
            return ""

        if ocr_score >= 0.99 and self.is_valid_indian_plate(raw):
            return raw

        candidates = self._generate_candidates(raw)
        if not candidates:
            return raw

        best = raw
        best_score = -10_000.0

        for candidate in candidates:
            syntax = self.syntax_score(candidate)
            edits = self._plate_distance(raw, candidate)

            total = (
                4.0 * syntax +
                2.0 * max(0.0, min(1.0, float(ocr_score))) -
                0.85 * edits
            )

            if total > best_score:
                best_score = total
                best = candidate

        return best


def build_default_corrector() -> PlateSyntaxCorrector:
    return PlateSyntaxCorrector(preferred_state="KL")
