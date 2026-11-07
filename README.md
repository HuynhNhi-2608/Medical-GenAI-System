# Hệ thống hỗ trợ chẩn đoán y tế tích hợp GenAI sinh ảnh minh họa

Nhập báo cáo lâm sàng tiếng Việt → nhận diện bệnh (từ điển + quy tắc, NER tùy chọn) → dựng prompt tiếng Anh → Stable Diffusion 1.5 + LoRA sinh ảnh minh họa có đóng dấu chìm → lưu SQL Server → chuyên gia chấm điểm.

> **Lưu ý:** ảnh do AI tạo chỉ mang tính minh họa, không dùng để chẩn đoán hay điều trị. Chỉ dùng dữ liệu giả lập, không nhập thông tin bệnh nhân thật.

## Các trang

| Đường dẫn | Chức năng |
|---|---|
| `/about` | Giới thiệu hệ thống |
| `/` | Phân tích báo cáo và sinh ảnh |
| `/parse` | Thử phân tích (không cần GPU) |
| `/cases` | Lịch sử ca đã lưu |
| `/compare` | So sánh SD gốc, LoRA v1, LoRA v2 |
| `/evaluation` | Kết quả đánh giá (FID, KID, CLIP, độ chính xác loại ảnh) |
| `/rate` | Chuyên gia chấm điểm ảnh |
| `/stats` | Thống kê điểm chấm |
| `/dictionary` | Từ điển bệnh và thuật ngữ |
| `/export` | Xuất CSV, in PDF từng ca |

## Cài đặt (Windows)

1. Cài Python 3.11, SQL Server (bản Express) và ODBC Driver for SQL Server.
2. Mở terminal tại thư mục dự án:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

3. Mở `.env` và điền thông tin SQL Server (mặc định dùng Windows Authentication, để trống `DB_USER` và `DB_PASS`).
4. Chạy:

```powershell
uvicorn app:app --port 8000
```

5. Mở http://127.0.0.1:8000

Muốn xem giao diện mà chưa có GPU: đặt `ENABLE_GENERATOR=0` trong `.env`. Hầu hết các trang vẫn hoạt động.

## Mô hình

- **LoRA** (12 MB) đã có sẵn trong `models/lora_weights/`.
- **Mô hình NER** (1 GB, không bắt buộc) quá nặng cho GitHub: [tải tại đây](https://drive.google.com/file/d/19c5LvadHWETIdoDv3N3YQ1KT8zl2NH0s/view?usp=drive_link). Giải nén vào `models/ner_model/`. Nếu thiếu, hệ thống vẫn chạy bằng từ điển và quy tắc.
- Mô hình nền Stable Diffusion 1.5 tự tải về khi chạy lần đầu.

## Giới hạn

Ảnh chưa được chuyên gia y khoa kiểm chứng; các loại ảnh ít dữ liệu (siêu âm, MRI, chụp mạch) và vùng cột sống, chậu, cơ xương khớp cho kết quả kém hơn. Chi tiết xem trang `/evaluation`.