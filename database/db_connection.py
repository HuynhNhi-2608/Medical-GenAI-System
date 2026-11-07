"""Kết nối SQL Server (qua pyodbc + SQLAlchemy) và định nghĩa các bảng.

Đọc cấu hình từ file .env:
    DB_SERVER=.            (máy hiện tại, instance mặc định)  hoặc  .\\SQLEXPRESS
    DB_NAME=MedicalGenAIDB
    DB_USER=...            (để trống nếu dùng Windows Authentication)
    DB_PASS=...
Có thể đặt DB_URL=sqlite:///file.db để chạy thử không cần SQL Server.
Chỉ dùng DỮ LIỆU GIẢ LẬP, không nhập thông tin bệnh nhân thật.
"""
import os
from contextlib import contextmanager
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer, Unicode, UnicodeText,
                        cast, create_engine, func)
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

load_dotenv()
Base = declarative_base()


# ---------------- Các bảng ----------------
class Report(Base):
    """Báo cáo lâm sàng người dùng nhập."""
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    report_text = Column(UnicodeText, nullable=False)
    modality = Column(Unicode(50))
    region = Column(Unicode(50))
    ner_used = Column(Boolean, default=False)
    findings = relationship("Finding", back_populates="report", cascade="all, delete-orphan")
    generations = relationship("Generation", back_populates="report", cascade="all, delete-orphan")


class Finding(Base):
    """Từng bệnh hệ thống tìm thấy và trạng thái của nó."""
    __tablename__ = "findings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False)
    disease = Column(Unicode(200), nullable=False)
    # supported | unsupported | negated | uncertain | ner_only
    status = Column(Unicode(30), nullable=False)
    source = Column(Unicode(20), default="rule")  # rule | ner
    report = relationship("Report", back_populates="findings")


class Generation(Base):
    """Mỗi ảnh được sinh ra, kèm đủ thông tin để tái lập."""
    __tablename__ = "generations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    prompt = Column(Unicode(500), nullable=False)
    diseases = Column(Unicode(500))
    seed = Column(Integer)
    image_file = Column(Unicode(260), nullable=False)
    model_id = Column(Unicode(200))
    lora_name = Column(Unicode(100))
    steps = Column(Integer)
    guidance = Column(Float)
    reliability = Column(Unicode(20))
    report = relationship("Report", back_populates="generations")
    ratings = relationship("Rating", back_populates="generation", cascade="all, delete-orphan")


class Rating(Base):
    """Đánh giá của chuyên gia/người dùng cho từng ảnh (dùng cho phần thực nghiệm)."""
    __tablename__ = "ratings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    generation_id = Column(Integer, ForeignKey("generations.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    rater = Column(Unicode(100))
    disease_score = Column(Integer)   # 1-5: ảnh có thể hiện đúng bệnh không
    modality_ok = Column(Boolean)     # đúng loại ảnh (CT/X-quang/...) không
    realism_score = Column(Integer)   # 1-5: độ giống ảnh thật
    comment = Column(UnicodeText)
    generation = relationship("Generation", back_populates="ratings")


def _pick_driver() -> str:
    import pyodbc
    names = [d for d in pyodbc.drivers() if "SQL Server" in d]
    if not names:
        raise RuntimeError("Chưa cài ODBC Driver for SQL Server. Tải 'ODBC Driver 17/18 for SQL Server' từ Microsoft.")
    for pref in ("ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server"):
        if pref in names:
            return pref
    return names[-1]


def _conn_str(database):
    server = os.getenv("DB_SERVER", ".")
    user, pwd = os.getenv("DB_USER", ""), os.getenv("DB_PASS", "")
    auth = f"UID={user};PWD={pwd};" if user else "Trusted_Connection=yes;"
    return f"DRIVER={{{_pick_driver()}}};SERVER={server};DATABASE={database};{auth}TrustServerCertificate=yes;"


def build_url():
    if os.getenv("DB_URL"):
        return os.environ["DB_URL"]
    name = os.getenv("DB_NAME", "MedicalGenAIDB")
    return URL.create("mssql+pyodbc", query={"odbc_connect": _conn_str(name)})


def ensure_database():
    if os.getenv("DB_URL"):
        return
    import re
    import pyodbc
    name = os.getenv("DB_NAME", "MedicalGenAIDB")
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError("DB_NAME chỉ được chứa chữ cái, số và dấu gạch dưới")
    conn = pyodbc.connect(_conn_str("master"), autocommit=True, timeout=10)
    try:
        conn.cursor().execute(f"IF DB_ID(N'{name}') IS NULL CREATE DATABASE [{name}]")
    finally:
        conn.close()


_engine = None
_Session = None


def get_engine():
    global _engine, _Session
    if _engine is None:
        _engine = create_engine(build_url(), pool_pre_ping=True, future=True)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def init_db():
    ensure_database()
    Base.metadata.create_all(get_engine())


@contextmanager
def session_scope():
    get_engine()
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


# ---------------- Hàm tiện ích cho app ----------------
def save_result(report_text, analysis, generations, meta):
    """Lưu báo cáo, các bệnh tìm thấy và ảnh đã sinh. Trả về (report_id, [generation_id])."""
    with session_scope() as s:
        rep = Report(report_text=report_text, modality=analysis["modality"], region=analysis["region"],
                     ner_used=analysis.get("ner_used", False))
        for d in analysis["supported"]:
            rep.findings.append(Finding(disease=d, status="supported"))
        for d in analysis["unsupported"]:
            rep.findings.append(Finding(disease=d, status="unsupported"))
        for d in analysis["excluded_negated"]:
            rep.findings.append(Finding(disease=d, status="negated"))
        for d in analysis["uncertain"]:
            rep.findings.append(Finding(disease=d, status="uncertain"))
        for d in analysis["ner_only"]:
            rep.findings.append(Finding(disease=d, status="ner_only", source="ner"))
        gens = []
        for g in generations:
            gens.append(Generation(prompt=g["prompt"], diseases=", ".join(g["diseases"]), seed=g["seed"],
                                   image_file=g["file"], model_id=meta.get("model_id"), lora_name=meta.get("lora_name"),
                                   steps=meta.get("steps"), guidance=meta.get("guidance"),
                                   reliability=meta.get("reliability")))
        rep.generations.extend(gens)
        s.add(rep)
        s.flush()
        return rep.id, [g.id for g in gens]


def list_history(limit=20):
    with session_scope() as s:
        rows = s.query(Generation).order_by(Generation.id.desc()).limit(limit).all()
        return [{"id": g.id, "created_at": g.created_at.isoformat(timespec="seconds"), "prompt": g.prompt,
                 "image_file": g.image_file, "reliability": g.reliability,
                 "report": g.report.report_text if g.report else "",
                 "n_ratings": len(g.ratings)} for g in rows]


def add_rating(generation_id, rater, disease_score, modality_ok, realism_score, comment):
    with session_scope() as s:
        if s.get(Generation, generation_id) is None:
            raise ValueError("Không có ảnh với id này")
        r = Rating(generation_id=generation_id, rater=rater, disease_score=disease_score,
                   modality_ok=modality_ok, realism_score=realism_score, comment=comment)
        s.add(r)
        s.flush()
        return r.id


def rating_summary():
    with session_scope() as s:
        n = s.query(func.count(Rating.id)).scalar() or 0
        # SQL Server: AVG của số nguyên bị làm tròn xuống, nên ép sang FLOAT trước khi tính trung bình
        avg = lambda col: s.query(func.avg(cast(col, Float))).scalar()
        # SQL Server không hiểu "IS 1", phải so sánh bằng "= 1"
        ok = s.query(func.count(Rating.id)).filter(Rating.modality_ok == True).scalar() or 0  # noqa: E712
        d_avg, r_avg = avg(Rating.disease_score), avg(Rating.realism_score)
        return {"n_ratings": n,
                "avg_disease_score": round(float(d_avg), 2) if n and d_avg is not None else None,
                "avg_realism_score": round(float(r_avg), 2) if n and r_avg is not None else None,
                "modality_ok_rate": round(ok / n, 3) if n else None}


def list_unrated(rater="", limit=10):
    """Ảnh người chấm này chưa đánh giá (không truyền rater thì lấy ảnh chưa có đánh giá nào)."""
    with session_scope() as s:
        q = s.query(Generation)
        if rater:
            q = q.filter(~Generation.ratings.any(Rating.rater == rater))
        else:
            q = q.filter(~Generation.ratings.any())
        total = q.count()
        rows = q.order_by(Generation.id.asc()).limit(limit).all()
        n_done = 0
        if rater:
            n_done = s.query(func.count(Rating.id)).filter(Rating.rater == rater).scalar() or 0
        items = [{"id": g.id, "prompt": g.prompt, "diseases": g.diseases or "", "image_file": g.image_file,
                  "reliability": g.reliability, "report": g.report.report_text if g.report else "",
                  "modality": g.report.modality if g.report else None,
                  "region": g.report.region if g.report else None} for g in rows]
        return {"total_unrated": total, "n_done_by_rater": n_done, "items": items}


def stats():
    """Số liệu tổng hợp cho trang thống kê."""
    with session_scope() as s:
        def counts(q):
            return [{"label": (k if k not in (None, "") else "không rõ"), "n": int(n)} for k, n in q.all()]

        def dist(col):
            got = {k: int(n) for k, n in s.query(col, func.count(Rating.id)).filter(col.isnot(None)).group_by(col).all()}
            return [{"label": str(i), "n": got.get(i, 0)} for i in range(1, 6)]

        def by_group(rows):
            return [{"label": (k if k not in (None, "") else "không rõ"), "n": int(n),
                     "avg_disease": round(float(d), 2) if d is not None else None,
                     "avg_realism": round(float(r), 2) if r is not None else None} for k, n, d, r in rows]

        avg_d, avg_r = func.avg(cast(Rating.disease_score, Float)), func.avg(cast(Rating.realism_score, Float))
        out = {
            "totals": {"reports": s.query(func.count(Report.id)).scalar() or 0,
                       "generations": s.query(func.count(Generation.id)).scalar() or 0,
                       "ratings": s.query(func.count(Rating.id)).scalar() or 0,
                       "raters": s.query(func.count(func.distinct(Rating.rater))).scalar() or 0},
            "status": counts(s.query(Finding.status, func.count(Finding.id)).group_by(Finding.status)
                             .order_by(func.count(Finding.id).desc())),
            "supported_diseases": counts(s.query(Finding.disease, func.count(Finding.id))
                                         .filter(Finding.status == "supported").group_by(Finding.disease)
                                         .order_by(func.count(Finding.id).desc()).limit(15)),
            "modality": counts(s.query(Report.modality, func.count(Report.id)).group_by(Report.modality)
                               .order_by(func.count(Report.id).desc())),
            "region": counts(s.query(Report.region, func.count(Report.id)).group_by(Report.region)
                             .order_by(func.count(Report.id).desc())),
            "reliability": counts(s.query(Generation.reliability, func.count(Generation.id))
                                  .group_by(Generation.reliability).order_by(func.count(Generation.id).desc())),
            "score_dist": {"disease": dist(Rating.disease_score), "realism": dist(Rating.realism_score)},
            "ratings_by_reliability": by_group(
                s.query(Generation.reliability, func.count(Rating.id), avg_d, avg_r)
                 .select_from(Rating).join(Generation, Rating.generation_id == Generation.id)
                 .group_by(Generation.reliability).all()),
            "ratings_by_modality": by_group(
                s.query(Report.modality, func.count(Rating.id), avg_d, avg_r)
                 .select_from(Rating).join(Generation, Rating.generation_id == Generation.id)
                 .join(Report, Generation.report_id == Report.id).group_by(Report.modality).all()),
        }
    out["rating_summary"] = rating_summary()
    return out

def list_reports(q="", limit=20, offset=0):
    """Danh sách báo cáo (mới nhất trước), có thể tìm theo nội dung."""
    with session_scope() as s:
        query = s.query(Report)
        if q:
            query = query.filter(Report.report_text.contains(q, autoescape=True))
        total = query.count()
        rows = query.order_by(Report.id.desc()).offset(offset).limit(limit).all()
        items = [{"id": r.id, "created_at": r.created_at.isoformat(timespec="seconds"),
                  "report": r.report_text, "modality": r.modality, "region": r.region,
                  "supported": [f.disease for f in r.findings if f.status == "supported"],
                  "n_negated": sum(1 for f in r.findings if f.status == "negated"),
                  "n_generations": len(r.generations),
                  "n_ratings": sum(len(g.ratings) for g in r.generations),
                  "image_file": r.generations[0].image_file if r.generations else None} for r in rows]
        return {"total": total, "items": items}


def get_report(report_id):
    """Toàn bộ thông tin một ca: báo cáo, các bệnh tìm thấy, ảnh đã sinh và điểm chấm."""
    with session_scope() as s:
        r = s.get(Report, report_id)
        if r is None:
            return None
        gens = []
        for g in sorted(r.generations, key=lambda x: x.id):
            ratings = [{"rater": x.rater or "", "disease_score": x.disease_score, "modality_ok": x.modality_ok,
                        "realism_score": x.realism_score, "comment": x.comment or "",
                        "created_at": x.created_at.isoformat(timespec="seconds")} for x in g.ratings]
            gens.append({"id": g.id, "created_at": g.created_at.isoformat(timespec="seconds"), "prompt": g.prompt,
                         "diseases": g.diseases or "", "seed": g.seed, "image_file": g.image_file,
                         "model_id": g.model_id, "lora_name": g.lora_name, "steps": g.steps,
                         "guidance": g.guidance, "reliability": g.reliability, "ratings": ratings})
        return {"id": r.id, "created_at": r.created_at.isoformat(timespec="seconds"), "report": r.report_text,
                "modality": r.modality, "region": r.region, "ner_used": bool(r.ner_used),
                "findings": [{"disease": f.disease, "status": f.status, "source": f.source} for f in r.findings],
                "generations": gens}


def export_ratings():
    """Toàn bộ điểm chấm kèm thông tin ảnh/báo cáo (để xuất CSV)."""
    with session_scope() as s:
        rows = (s.query(Rating, Generation, Report)
                 .join(Generation, Rating.generation_id == Generation.id)
                 .join(Report, Generation.report_id == Report.id)
                 .order_by(Rating.id).all())
        return [{"rating_id": r.id, "created_at": r.created_at.isoformat(timespec="seconds"), "rater": r.rater or "",
                 "disease_score": r.disease_score, "modality_ok": "" if r.modality_ok is None else int(bool(r.modality_ok)),
                 "realism_score": r.realism_score, "comment": r.comment or "",
                 "generation_id": g.id, "report_id": p.id, "modality": p.modality or "", "region": p.region or "",
                 "diseases": g.diseases or "", "prompt": g.prompt, "reliability": g.reliability or "",
                 "lora_name": g.lora_name or "", "seed": g.seed, "steps": g.steps, "guidance": g.guidance,
                 "image_file": g.image_file} for r, g, p in rows]


def export_generations():
    """Mọi ảnh đã sinh kèm báo cáo gốc (để xuất CSV)."""
    with session_scope() as s:
        rows = (s.query(Generation, Report).join(Report, Generation.report_id == Report.id)
                 .order_by(Generation.id).all())
        return [{"generation_id": g.id, "report_id": p.id, "created_at": g.created_at.isoformat(timespec="seconds"),
                 "report": p.report_text, "modality": p.modality or "", "region": p.region or "",
                 "diseases": g.diseases or "", "prompt": g.prompt, "reliability": g.reliability or "",
                 "model_id": g.model_id or "", "lora_name": g.lora_name or "", "seed": g.seed, "steps": g.steps,
                 "guidance": g.guidance, "n_ratings": len(g.ratings), "image_file": g.image_file} for g, p in rows]