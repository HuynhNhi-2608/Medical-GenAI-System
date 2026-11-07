"""Phân tích báo cáo lâm sàng tiếng Việt.

Hai lớp xử lý:
1. Luật + từ điển Việt-Anh (có xử lý phủ định, "nghi ngờ"): quyết định bệnh nào được sinh ảnh.
2. Mô hình NER XLM-R (tùy chọn): phát hiện tên bệnh NGOÀI từ điển để báo "đã thấy nhưng chưa hỗ trợ".

Tên bệnh tiếng Anh trong từ điển phải trùng với nhãn bệnh lúc train LoRA (notebook 01).
"""
import json
import re
import unicodedata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

CLAUSE_SPLIT = re.compile(r"[,;.:\n()]| nhưng | kèm | trừ | ngoại trừ ")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text.lower())


def _spans(tags):
    """BIO -> [(loại, bắt đầu, kết thúc)]. I- không có B- đứng trước vẫn tính là bắt đầu mới."""
    out, cur = [], None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and (cur is None or cur[0] != t[2:])):
            if cur:
                out.append((cur[0], cur[1], i))
                cur = None
            if t != "O":
                cur = (t[2:], i)
    return out


class ReportParser:
    def __init__(self, models_dir=MODELS_DIR, use_ner=True):
        self.models_dir = Path(models_dir)
        with open(self.models_dir / "vi_en_dictionary.json", encoding="utf-8") as f:
            d = json.load(f)
        self.diseases = d["diseases"]
        self.modality_map = d["modality"]
        self.region_map = d["region"]
        self.negation_cues = d["negation_cues"]
        self.uncertain_cues = d["uncertain_cues"]
        with open(self.models_dir / "supported_diseases.json", encoding="utf-8") as f:
            s = json.load(f)
        self.supported = list(s["supported"])
        self.rare = list(s["rare"])

        self.ner_status = "tắt"
        self._ner_model = self._ner_tok = None
        if use_ner:
            self._load_ner()

    # ---------- NER (tùy chọn) ----------
    def _load_ner(self):
        ner_dir = self.models_dir / "ner_model"
        if not (ner_dir / "config.json").exists():
            self.ner_status = f"tắt (chưa thấy mô hình ở {ner_dir})"
            return
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._ner_tok = AutoTokenizer.from_pretrained(str(ner_dir))
            self._ner_model = AutoModelForTokenClassification.from_pretrained(str(ner_dir)).to(self._device).eval()
            self._id2label = {int(k): v for k, v in self._ner_model.config.id2label.items()}
            self.ner_status = f"bật ({self._device})"
        except Exception as e:  # không để lỗi NER làm sập cả hệ thống
            self._ner_model = self._ner_tok = None
            self.ner_status = f"tắt (lỗi nạp: {e})"

    @property
    def ner_enabled(self) -> bool:
        return self._ner_model is not None

    def extract_entities(self, text: str):
        if not self.ner_enabled:
            return []
        import torch
        words = TOKEN_RE.findall(text.lower())
        if not words:
            return []
        enc = self._ner_tok(words, is_split_into_words=True, truncation=True, max_length=192, return_tensors="pt")
        wid = enc.word_ids()
        with torch.no_grad():
            logits = self._ner_model(**{k: v.to(self._device) for k, v in enc.items()}).logits[0].cpu()
        tags, prev = ["O"] * len(words), None
        for pos, w in enumerate(wid):
            if w is not None and w != prev:
                tags[w] = self._id2label[int(logits[pos].argmax())]
            prev = w
        return [{"loai": typ, "text": " ".join(words[a:b])} for typ, a, b in _spans(tags)]

    # ---------- Luật + từ điển ----------
    def _has_cue(self, clause, cues):
        return any(re.search(rf"(?<!\w){re.escape(c)}(?!\w)", clause) for c in cues)

    def is_negated(self, text, pos):
        return self._has_cue(CLAUSE_SPLIT.split(text[:pos])[-1], self.negation_cues)

    def is_uncertain(self, text, pos):
        return self._has_cue(CLAUSE_SPLIT.split(text[:pos])[-1], self.uncertain_cues)

    def find_diseases(self, report_vi):
        """Mọi lần xuất hiện của thuật ngữ trong từ điển (kể cả bị phủ định), ưu tiên cụm dài."""
        text, hits, used = _norm(report_vi), [], []
        for term in sorted(self.diseases, key=len, reverse=True):
            for m in re.finditer(rf"(?<!\w){re.escape(_norm(term))}(?!\w)", text):
                s, e = m.span()
                if any(s < ue and e > us for us, ue in used):
                    continue
                used.append((s, e))
                hits.append({"term": term, "en": self.diseases[term], "start": s,
                             "negated": self.is_negated(text, s), "uncertain": self.is_uncertain(text, s)})
        return sorted(hits, key=lambda h: h["start"])

    @staticmethod
    def _detect_first(text, mapping, default=""):
        for pattern, value in mapping.items():
            if re.search(pattern, text):
                return value
        return default

    # ---------- Hàm chính ----------
    def analyze(self, report_vi: str, use_ner: bool = True) -> dict:
        text = _norm(report_vi)
        hits = self.find_diseases(report_vi)
        present = [h for h in hits if not h["negated"]]
        uniq = lambda seq: list(dict.fromkeys(seq))
        en_present = uniq(h["en"] for h in present)

        ner_only, ner_entities = [], []
        if use_ner and self.ner_enabled:
            ner_entities = self.extract_entities(report_vi)
            terms = [_norm(h["term"]) for h in hits]
            for e in ner_entities:
                t = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", e["text"])).strip()
                if e["loai"] != "ten_benh" or len(t) < 3:
                    continue
                if any(x in t or t in x for x in terms):
                    continue
                pat = r"\s*".join(re.escape(w) for w in t.split())
                m = re.search(pat, text)
                if m and self.is_negated(text, m.start()):
                    continue
                if t not in ner_only:
                    ner_only.append(t)

        return {
            "modality": self._detect_first(text, self.modality_map, "medical image"),
            "region": self._detect_first(text, self.region_map, ""),
            "supported": [d for d in en_present if d in self.supported],
            "unsupported": [d for d in en_present if d not in self.supported],
            "uncertain": uniq(h["en"] for h in present if h["uncertain"]),
            "excluded_negated": uniq(h["en"] for h in hits if h["negated"]),
            "ner_only": ner_only,
            "ner_used": bool(use_ner and self.ner_enabled),
            "hits": hits,
        }
