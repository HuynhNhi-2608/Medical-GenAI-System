"""Sinh ảnh minh họa bằng Stable Diffusion 1.5 + LoRA (huấn luyện ở notebook 02).

Ảnh luôn được đóng dấu "AI-generated" ở đáy. Đây chỉ là ảnh MINH HỌA, không phải ảnh chẩn đoán.
"""
import json
import os
import time
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_OUT_DIR = BASE_DIR / "static" / "generated_images"
LABEL_TEXT = "AI-generated illustration - not a real medical image"


def load_config(models_dir=MODELS_DIR) -> dict:
    with open(Path(models_dir) / "pipeline_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    # cho phép ghi đè bằng biến môi trường (hữu ích khi thử nghiệm)
    cfg["model_id"] = os.getenv("SD_MODEL_ID", cfg["model_id"])
    cfg["resolution"] = int(os.getenv("GEN_RESOLUTION", cfg["resolution"]))
    cfg["steps"] = int(os.getenv("GEN_STEPS", cfg["steps"]))
    return cfg


def assess_reliability(modality: str, region: str, cfg: dict) -> dict:
    """Cảnh báo độ tin cậy: loại ảnh/vùng có ít dữ liệu train thì ảnh sinh ra kém tin cậy hơn."""
    reasons = []
    if modality in cfg.get("weak_modalities", []):
        reasons.append(f"loại ảnh '{modality}' có ít dữ liệu huấn luyện")
    if region in cfg.get("weak_regions", []):
        reasons.append(f"vùng '{region}' có ít dữ liệu huấn luyện")
    if reasons:
        return {"level": "low", "message": "Độ tin cậy thấp: " + "; ".join(reasons) + ". Ảnh có thể sai loại hoặc sai giải phẫu."}
    return {"level": "normal", "message": "Ảnh chỉ mang tính minh họa, chưa được chuyên gia y khoa kiểm chứng."}


class ImageGenerator:
    def __init__(self, models_dir=MODELS_DIR, out_dir=DEFAULT_OUT_DIR):
        self.models_dir = Path(models_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = load_config(self.models_dir)
        self.pipe = None
        self.device = "cpu"
        self.lora_name = "chưa nạp"
        self.status = "chưa nạp"

    # ---------- Nạp mô hình ----------
    def load(self):
        import torch
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

        lora_dir = self.models_dir / "lora_weights"
        weight = lora_dir / "pytorch_lora_weights.safetensors"
        if not weight.exists():
            raise FileNotFoundError(f"Không thấy {weight}. Hãy chép file LoRA từ Drive vào thư mục models/lora_weights.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg["model_id"], torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
        ).to(self.device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        pipe.load_lora_weights(str(lora_dir), weight_name=weight.name)
        if self.device == "cuda" and torch.cuda.get_device_properties(0).total_memory < 7 * 1024**3:
            pipe.enable_attention_slicing()  # tiết kiệm VRAM cho GPU 6 GB
        self.pipe = pipe
        self.lora_name = "lora_weights"
        self.status = f"sẵn sàng ({self.device})"
        if self.device == "cpu":
            self.status += " - rất chậm, nên dùng GPU"

    @property
    def ready(self) -> bool:
        return self.pipe is not None

    # ---------- Ghép prompt ----------
    def _n_tokens(self, prompt: str) -> int:
        return len(self.pipe.tokenizer(prompt).input_ids)

    def build_prompts(self, analysis: dict, mode: str = "combined"):
        """Prompt đúng định dạng lúc train: '<loại ảnh> of <vùng>: bệnh 1, bệnh 2'."""
        modality, region = analysis["modality"], analysis["region"]
        prefix = f"{modality} of {region}" if region else modality
        dis = list(analysis["supported"])
        if not dis:
            return []
        k = self.cfg["max_diseases_per_image"]
        groups = [[d] for d in dis] if mode == "per_disease" else [dis[i:i + k] for i in range(0, len(dis), k)]
        jobs = []
        for g in groups:
            g = list(g)
            while len(g) > 1 and self.ready and self._n_tokens(f"{prefix}: " + ", ".join(g)) > self.cfg["max_tokens"]:
                g.pop()
            jobs.append({"prompt": f"{prefix}: " + ", ".join(g), "diseases": g})
        return jobs

    # ---------- Sinh ảnh ----------
    @staticmethod
    def _stamp(img):
        from PIL import Image, ImageDraw
        w, h = img.size
        out = Image.new("RGB", (w, h + 22), (20, 20, 20))
        out.paste(img, (0, 0))
        ImageDraw.Draw(out).text((6, h + 5), LABEL_TEXT, fill=(235, 235, 235))
        return out

    def generate(self, analysis: dict, mode: str = "combined", seed: int = 1234):
        import torch
        if not self.ready:
            raise RuntimeError("Mô hình sinh ảnh chưa được nạp.")
        results = []
        for j, job in enumerate(self.build_prompts(analysis, mode)):
            s = seed + j
            gen = torch.Generator(device=self.device).manual_seed(s)
            t0 = time.time()
            img = self.pipe(job["prompt"], negative_prompt=self.cfg["negative_prompt"],
                            num_inference_steps=self.cfg["steps"], guidance_scale=self.cfg["guidance"],
                            height=self.cfg["resolution"], width=self.cfg["resolution"], generator=gen).images[0]
            fname = f"{uuid.uuid4().hex}.png"
            self._stamp(img).save(self.out_dir / fname)
            results.append({"prompt": job["prompt"], "diseases": job["diseases"], "seed": s,
                            "file": fname, "seconds": round(time.time() - t0, 1)})
        return results
