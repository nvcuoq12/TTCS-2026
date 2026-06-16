# CineRec — Hệ thống Gợi ý Phim trên MovieLens ml-100k

Đồ án môn **(TTCS)** — B23DCCN870  
Triển khai ba mô hình Collaborative Filtering từ đầu (from scratch) và xây dựng demo web thời gian thực.

---

## Tổng quan

| Thông tin | Chi tiết |
|-----------|----------|
| Dataset | MovieLens ml-100k — 943 users · 1.682 phim · 100.000 ratings |
| Mô hình | User-based CF · Item-based CF · Matrix Factorization (Funk SVD) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML / CSS / JavaScript (dark theme) |
| Python | 3.10+ |

### Kết quả thực nghiệm (Temporal Split 80/20)

| Mô hình | MAE | RMSE | Precision@10 | Recall@10 |
|---------|-----|------|--------------|-----------|
| Baseline (Global Mean) | ~0.945 | ~1.126 | — | — |
| User-based CF (K=30) | 0.7746 | 0.9863 | 0.0047 | 0.0034 |
| Item-based CF (K=30) | 0.7996 | 1.0211 | 0.0152 | 0.0092 |
| **Matrix Factorization** | **0.7706** | **0.9758** | **0.0518** | **0.0484** |

---

## Cấu trúc dự án

```
TTCS-2026/
├── Documents/
│   ├── FinalReport/          # Báo cáo cuối kỳ
│   ├── MidtermReport/        # Báo cáo giữa kỳ
│   └── WeeklyReports/        # Báo cáo tuần
│
└── SourceCode/
    ├── BTL_Template/         # Template LaTeX báo cáo
    │   ├── BTL.tex           # File LaTeX chính
    │   ├── Chuong/           # Các chương nội dung
    │   ├── Hinhve/           # Hình ảnh minh họa
    │   └── BTL.pdf           # Báo cáo PDF (output)
    │
    └── recsys_ml100k/        # Source code hệ thống
        ├── data/
        │   ├── u.data        # Raw ratings (tab-separated)
        │   ├── u.item        # Raw movie info
        │   ├── ratings.csv   # Processed ratings
        │   └── movies.csv    # Processed movie info
        ├── models.py         # UserBasedCF / ItemBasedCF / MatrixFactorization
        ├── app.py            # FastAPI backend (4 endpoints)
        ├── run.py            # Script khởi động nhanh
        ├── demo_console.py   # Demo giao diện terminal
        └── static/
            └── index.html    # Giao diện web
```

---

## Cài đặt và chạy

### Yêu cầu
- Python 3.10+
- RAM tối thiểu 4 GB (khuyến nghị 8 GB)

### 1. Clone và cài dependencies

```bash
git clone <repo-url>
cd TTCS-2026/SourceCode/recsys_ml100k

pip install fastapi uvicorn pandas numpy scikit-learn
```

### 2. Chạy web demo (khuyến nghị)

```bash
python run.py
```

Trình duyệt tự động mở tại `http://localhost:8000`.  
Lần đầu khởi động mất **45–90 giây** để huấn luyện 3 mô hình.

### 3. Chạy demo terminal (không cần server)

```bash
python demo_console.py
```

---

## API Endpoints

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `GET /` | GET | Giao diện web (index.html) |
| `GET /api/stats` | GET | Thống kê dataset và metrics mô hình |
| `GET /api/recommend/{user_id}` | GET | Gợi ý phim cho user |
| `GET /api/health` | GET | Kiểm tra trạng thái server |

**Parameters của `/api/recommend/{user_id}`:**
- `method`: `ubcf` \| `ibcf` \| `mf` (mặc định: `mf`)
- `n`: số phim gợi ý (mặc định: `10`)

**Ví dụ:**
```
GET /api/recommend/1?method=mf&n=10
```

---

## Các mô hình

### User-based Collaborative Filtering
- Tính cosine similarity mean-centered giữa các users (vectorized)
- Ma trận tương đồng: 943 × 943
- Dự đoán bằng K láng giềng gần nhất (K=30)

### Item-based Collaborative Filtering
- Adjusted cosine similarity (trừ user mean trước khi tính cosine)
- Ma trận tương đồng: 1.615 × 1.615
- Dự đoán bằng K items tương tự nhất (K=30)

### Matrix Factorization (Funk SVD)
- Triển khai SGD từ đầu với bias terms (bu, bi, μ)
- Siêu tham số tối ưu: K=20, α=0.005, λ=0.02, 30 epochs
- Cải thiện RMSE 13.3% so với baseline

---

## Biên dịch báo cáo LaTeX

```bash
cd SourceCode/BTL_Template

pdflatex BTL.tex
bibtex BTL
pdflatex BTL.tex
pdflatex BTL.tex
```

Yêu cầu: MiKTeX hoặc TeX Live với các gói `titlesec`, `biblatex`, `tikz`, `pgfplots`, `listings`, `glossaries`.

---

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Ngôn ngữ | Python 3.10 |
| Thuật toán | NumPy (vectorized) |
| Web framework | FastAPI 0.110 |
| ASGI server | Uvicorn 0.27 |
| Data processing | Pandas 2.0 |
| Evaluation | Scikit-learn (MAE/RMSE) |
| Báo cáo | LaTeX (MiKTeX) |

---

## Tác giả

**B23DCCN870** — Học viện Công nghệ Bưu chính Viễn thông (PTIT)  
Năm học 2025–2026
