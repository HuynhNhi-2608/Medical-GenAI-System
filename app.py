"""Ứng dụng web: báo cáo lâm sàng tiếng Việt -> phân tích -> ảnh minh họa -> lưu SQL Server.

Chạy:  uvicorn app:app --port 8000      rồi mở http://127.0.0.1:8000
Biến môi trường (.env hoặc terminal):
    ENABLE_NER=1        nạp mô hình NER (mặc định bật nếu có thư mục models/ner_model)
    ENABLE_GENERATOR=1  nạp SD + LoRA (đặt 0 để thử giao diện và SQL Server trước)
"""
import csv
import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

from AI_Module.Generator import ImageGenerator, assess_reliability  # noqa: E402
from AI_Module.nlp_parser import ReportParser  # noqa: E402
from database import db_connection as db  # noqa: E402

state = {"parser": None, "generator": None, "db_ok": False, "db_error": ""}
gpu_lock = threading.Lock()  # chỉ cho một yêu cầu dùng GPU tại một thời điểm


def _flag(name, default="1"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


@asynccontextmanager
async def lifespan(app):
    try:
        db.init_db()
        state["db_ok"] = True
    except Exception as e:  # app vẫn chạy được khi SQL Server chưa bật
        state["db_error"] = str(e)[:300]
        print("[CẢNH BÁO] Không kết nối được cơ sở dữ liệu:", e)
    state["parser"] = ReportParser(use_ner=_flag("ENABLE_NER"))
    print("[NER]", state["parser"].ner_status)
    if _flag("ENABLE_GENERATOR"):
        gen = ImageGenerator()
        try:
            gen.load()
        except Exception as e:
            gen.status = f"lỗi nạp: {e}"
            print("[CẢNH BÁO] Không nạp được mô hình sinh ảnh:", e)
        state["generator"] = gen
        print("[SD+LoRA]", gen.status)
    yield


app = FastAPI(title="Medical GenAI System", lifespan=lifespan)
(BASE_DIR / "static" / "generated_images").mkdir(parents=True, exist_ok=True)
class NoCacheStatic(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app.mount("/static", NoCacheStatic(directory=BASE_DIR / "static"), name="static")

class ReportIn(BaseModel):
    report: str = Field(..., min_length=3, max_length=5000)
    mode: str = Field("combined", pattern="^(combined|per_disease)$")
    seed: int = Field(1234, ge=0, le=2**31 - 1)


class RatingIn(BaseModel):
    generation_id: int
    rater: str = Field("", max_length=100)
    disease_score: Optional[int] = Field(None, ge=1, le=5)
    modality_ok: Optional[bool] = None
    realism_score: Optional[int] = Field(None, ge=1, le=5)
    comment: str = Field("", max_length=2000)


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")

@app.get("/compare")
def compare_page():
    return FileResponse(BASE_DIR / "static" / "compare.html")


@app.get("/evaluation")
def evaluation_page():
    return FileResponse(BASE_DIR / "static" / "evaluation.html")

@app.get("/rate")
def rate_page():
    return FileResponse(BASE_DIR / "static" / "rate.html")


@app.get("/stats")
def stats_page():
    return FileResponse(BASE_DIR / "static" / "stats.html")

@app.get("/parse")
def parse_page():
    return FileResponse(BASE_DIR / "static" / "parse.html")


@app.get("/cases")
def cases_page():
    return FileResponse(BASE_DIR / "static" / "cases.html")

@app.get("/about")
def about_page():
    return FileResponse(BASE_DIR / "static" / "about.html")


@app.get("/dictionary")
def dictionary_page():
    return FileResponse(BASE_DIR / "static" / "dictionary.html")


@app.get("/export")
def export_page():
    return FileResponse(BASE_DIR / "static" / "export.html")


@app.get("/api/health")
def health():
    p, g = state["parser"], state["generator"]
    return {"database": "ok" if state["db_ok"] else f"lỗi: {state['db_error']}",
            "ner": p.ner_status if p else "chưa khởi tạo",
            "generator": g.status if g else "tắt (ENABLE_GENERATOR=0)",
            "supported": p.supported if p else []}


@app.post("/api/analyze")
def analyze(body: ReportIn):
    return state["parser"].analyze(body.report)


@app.post("/api/generate")
def generate(body: ReportIn):
    parser, gen = state["parser"], state["generator"]
    analysis = parser.analyze(body.report)
    reliability = assess_reliability(analysis["modality"], analysis["region"], gen.cfg if gen else {})
    out = {"analysis": analysis, "reliability": reliability, "images": [], "db_saved": False, "notes": []}

    if gen is None or not gen.ready:
        out["notes"].append("Mô hình sinh ảnh chưa sẵn sàng, chỉ trả về kết quả phân tích.")
        return out
    if not analysis["supported"]:
        out["notes"].append("Không có bệnh nào trong danh sách được hỗ trợ sinh ảnh, nên không vẽ ảnh.")
        return out

    with gpu_lock:
        images = gen.generate(analysis, mode=body.mode, seed=body.seed)

    meta = {"model_id": gen.cfg["model_id"], "lora_name": gen.lora_name, "steps": gen.cfg["steps"],
            "guidance": gen.cfg["guidance"], "reliability": reliability["level"]}
    gen_ids = [None] * len(images)
    if state["db_ok"]:
        try:
            report_id, gen_ids = db.save_result(body.report, analysis, images, meta)
            out["report_id"], out["db_saved"] = report_id, True
        except Exception as e:
            out["notes"].append(f"Chưa lưu được vào cơ sở dữ liệu: {str(e)[:150]}")
    else:
        out["notes"].append("Cơ sở dữ liệu chưa kết nối, kết quả không được lưu.")

    for im, gid in zip(images, gen_ids):
        out["images"].append({**im, "generation_id": gid, "url": f"/static/generated_images/{im['file']}"})
    return out


@app.get("/api/history")
def history(limit: int = 20):
    if not state["db_ok"]:
        raise HTTPException(503, "Cơ sở dữ liệu chưa kết nối")
    rows = db.list_history(max(1, min(limit, 100)))
    for r in rows:
        r["url"] = f"/static/generated_images/{r['image_file']}"
    return rows


@app.post("/api/rating")
def rating(body: RatingIn):
    if not state["db_ok"]:
        raise HTTPException(503, "Cơ sở dữ liệu chưa kết nối")
    try:
        rid = db.add_rating(body.generation_id, body.rater, body.disease_score, body.modality_ok,
                            body.realism_score, body.comment)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"rating_id": rid}


@app.get("/api/ratings/summary")
def ratings_summary():
    if not state["db_ok"]:
        raise HTTPException(503, "Cơ sở dữ liệu chưa kết nối")
    return db.rating_summary()

@app.get("/api/ratings/queue")
def ratings_queue(rater: str = "", limit: int = 10):
    if not state["db_ok"]:
        raise HTTPException(503, "Cơ sở dữ liệu chưa kết nối")
    data = db.list_unrated(rater.strip()[:100], max(1, min(limit, 50)))
    for it in data["items"]:
        it["url"] = f"/static/generated_images/{it['image_file']}"
    return data


@app.get("/api/stats")
def api_stats():
    if not state["db_ok"]:
        raise HTTPException(503, "Cơ sở dữ liệu chưa kết nối")
    return db.stats()

def _find_cue(text, pos, cues):
    """Từ khóa phủ định/không chắc chắn nằm cùng mệnh đề với thuật ngữ bệnh (để giải thích cho người dùng)."""
    import re
    from AI_Module.nlp_parser import CLAUSE_SPLIT
    clause = CLAUSE_SPLIT.split(text[:pos])[-1]
    for c in cues:
        if re.search(rf"(?<!\w){re.escape(c)}(?!\w)", clause):
            return c
    return None


@app.post("/api/parse")
def parse_preview(body: ReportIn):
    """Phân tích báo cáo kèm thông tin để tô màu văn bản và xem trước prompt (không cần GPU)."""
    import unicodedata
    parser = state["parser"]
    a = parser.analyze(body.report)
    norm = unicodedata.normalize("NFC", body.report.lower())
    nfc = unicodedata.normalize("NFC", body.report)
    shown = nfc if len(nfc) == len(norm) else norm   # giữ nguyên chữ hoa/thường nếu vị trí vẫn khớp
    for h in a["hits"]:
        h["end"] = h["start"] + len(unicodedata.normalize("NFC", h["term"].lower()))
        h["cue"] = None
        if h["negated"]:
            h["cue"] = _find_cue(norm, h["start"], parser.negation_cues)
        elif h["uncertain"]:
            h["cue"] = _find_cue(norm, h["start"], parser.uncertain_cues)
        h["supported"] = h["en"] in parser.supported
    if state["generator"] is not None:
        builder = state["generator"]
    else:
        if "builder" not in state:
            state["builder"] = ImageGenerator()
        builder = state["builder"]
    return {"analysis": a, "text": shown,
            "reliability": assess_reliability(a["modality"], a["region"], builder.cfg),
            "prompts": {m: builder.build_prompts(a, m) for m in ("combined", "per_disease")},
            "negative_prompt": builder.cfg.get("negative_prompt", "")}


@app.get("/api/reports")
def reports_list(q: str = "", limit: int = 20, offset: int = 0):
    if not state["db_ok"]:
        raise HTTPException(503, "Cơ sở dữ liệu chưa kết nối")
    data = db.list_reports(q.strip()[:100], max(1, min(limit, 100)), max(0, offset))
    for it in data["items"]:
        it["url"] = f"/static/generated_images/{it['image_file']}" if it["image_file"] else None
    return data


@app.get("/api/reports/{report_id}")
def report_detail(report_id: int):
    if not state["db_ok"]:
        raise HTTPException(503, "Cơ sở dữ liệu chưa kết nối")
    data = db.get_report(report_id)
    if data is None:
        raise HTTPException(404, "Không có ca này")
    for g in data["generations"]:
        g["url"] = f"/static/generated_images/{g['image_file']}"
    return data

def _builder():
    """Lấy bộ dựng prompt/cấu hình mà không cần nạp mô hình sinh ảnh."""
    if state["generator"] is not None:
        return state["generator"]
    if "builder" not in state:
        state["builder"] = ImageGenerator()
    return state["builder"]


def _pretty_pattern(pattern):
    """Biến biểu thức chính quy trong từ điển thành chữ dễ đọc (x[- ]?quang -> x-quang)."""
    out = []
    for part in pattern.split("|"):
        part = part.replace("\\b", "").replace("[- ]?", "-")
        if part not in out:
            out.append(part)
    return out


@app.get("/api/dictionary")
def dictionary_info():
    p = state["parser"]
    cfg = _builder().cfg
    groups = {}
    for term, en in p.diseases.items():
        groups.setdefault(en, []).append(term)
    diseases = []
    for en, terms in groups.items():
        status = "supported" if en in p.supported else "rare" if en in p.rare else "other"
        diseases.append({"en": en, "terms": sorted(terms), "status": status})
    order = {"supported": 0, "rare": 1, "other": 2}
    diseases.sort(key=lambda d: (order[d["status"]], d["en"]))
    return {"diseases": diseases, "n_terms": len(p.diseases),
            "modality": [{"name": k, "words": _pretty_pattern(v)} for v, k in p.modality_map.items()],
            "region": [{"name": k, "words": _pretty_pattern(v)} for v, k in p.region_map.items()],
            "negation_cues": p.negation_cues, "uncertain_cues": p.uncertain_cues,
            "weak_modalities": cfg.get("weak_modalities", []), "weak_regions": cfg.get("weak_regions", []),
            "max_per_image": cfg.get("max_diseases_per_image", 3)}


def _csv_response(rows, columns, filename):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns)
    w.writeheader()
    w.writerows(rows)
    # "\ufeff" (BOM) giúp Excel mở đúng tiếng Việt
    return Response("\ufeff" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/export/ratings.csv")
def export_ratings_csv():
    if not state["db_ok"]:
        raise HTTPException(503, "Cơ sở dữ liệu chưa kết nối")
    rows = db.export_ratings()
    cols = ["rating_id", "created_at", "rater", "disease_score", "modality_ok", "realism_score", "comment",
            "generation_id", "report_id", "modality", "region", "diseases", "prompt", "reliability",
            "lora_name", "seed", "steps", "guidance", "image_file"]
    return _csv_response(rows, cols, "diem_cham.csv")


@app.get("/api/export/generations.csv")
def export_generations_csv():
    if not state["db_ok"]:
        raise HTTPException(503, "Cơ sở dữ liệu chưa kết nối")
    rows = db.export_generations()
    cols = ["generation_id", "report_id", "created_at", "report", "modality", "region", "diseases", "prompt",
            "reliability", "model_id", "lora_name", "seed", "steps", "guidance", "n_ratings", "image_file"]
    return _csv_response(rows, cols, "anh_da_sinh.csv")