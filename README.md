# Spatio-Temporal Anatomy of Urban Traffic Bottlenecks

**Giải phẫu Không gian – Thời gian các Nút thắt Giao thông Đô thị: Tín hiệu Cảnh báo sớm và Thử nghiệm Điều tiết chủ động qua Mạng Nơ-ron Đồ thị**

Dự án nghiên cứu cuối kỳ môn *Data Visualization & Analytics* (bài báo chuẩn IEEE $\le 3.000$ từ) trên hai mạng lưới cảm biến cao tốc Caltrans PeMS (`PEMS08` làm tập nghiên cứu chính và `PEMS04` làm tập kiểm chứng độc lập).

---

## 1. Lộ trình 7 Bước & Báo cáo

| Bước | Notebook (`notebooks/`) | Báo cáo (`docs/`) | Trạng thái | Tóm tắt đầu ra chính |
| :---: | :--- | :--- | :---: | :--- |
| **01** | [`01_Overview.ipynb`](./notebooks/01_Overview.ipynb) | [`01_Overview_Report.md`](./docs/01_Overview_Report.md) | Hoàn thành | Kiểm toán 4 tập PeMS; chọn `PEMS08` (Primary) và `PEMS04` (Validation). |
| **02** | [`02_Data_Cleaning_and_Interpolation.ipynb`](./notebooks/02_Data_Cleaning_and_Interpolation.ipynb) | [`02_Data_Cleaning_Report.md`](./docs/02_Data_Cleaning_Report.md) | Hoàn thành | Bộ 4 tiêu chí tách lỗi cảm biến; nội suy Hybrid (`L* = 18` bước, `Profile + Spatial Mean`). |
| **03** | [`03_EDA_and_Bottleneck_Analysis.ipynb`](./notebooks/03_EDA_and_Bottleneck_Analysis.ipynb) | [`03_EDA_and_Bottleneck_Report.md`](./docs/03_EDA_and_Bottleneck_Report.md) | Hoàn thành | Tiêu chí tắc thực `(Speed < 40 & Occ ≥ 0,15)`; Top 20 điểm nghẽn (`71,2%` tắc `PEMS08`); đồ thị tĩnh `.gexf`. |
| **04** | [`04_Pre_Congestion_Signal_Analysis.ipynb`](./notebooks/04_Pre_Congestion_Signal_Analysis.ipynb) | [`04_Pre_Congestion_Signal_Report.md`](./docs/04_Pre_Congestion_Signal_Report.md) | Hoàn thành | Quỹ đạo pha (`n = 3.136`); chốt `H* = 8` (`40p`); kiểm chứng `Std40p` (`5,32x`); `Self + 2 chiều` (`+9,36% PR-AUC`, `p = 2,26e-14`); tối ưu Pareto `Δt* = 15p`. |
| **05** | `05_Modeling_and_Explainability.ipynb` | `05_Modeling_Report.md` | Đang triển khai | Đối chiếu khách quan các kiến trúc mô hình dự báo (`TensorFlow / Keras`) và giải thích mô hình. |
| **06** | `06_Simulation_and_Cross_Validation.ipynb` | `06_Simulation_Report.md` *(dự kiến)* | Kế hoạch | Giả lập điều tiết lưu lượng đầu nguồn (`RQ2`) và kiểm chứng chéo trên `PEMS04`. |
| **07** | `07_Research_Synthesis_and_Conclusions.ipynb` | `07_Synthesis_Report.md` *(dự kiến)* | Kế hoạch | Tổng hợp bằng chứng định lượng kết luận trọn vẹn `RQ1` và `RQ2`. |

---

## 2. Bộ Tham số Định lượng Đã Chốt (Từ Bước 01 – 04)

* **Tiêu chí tắc nghẽn thực:** `(Speed < 40 mph) & (Occupancy ≥ 0,15)`, giữ nguyên chuỗi 24h liên tục (`99,56%` tắc thực nằm trong `05–23h`, độ vững Top 20 đạt `20/20`).
* **Cửa sổ quá khứ đầu vào (`H*`):** **`8` bước (`40 phút`)** chốt thuần thống kê qua `PACF` (`95% CI = 0,0147` cắt đồng thời cả `Speed` và `Occupancy` tại `k = 9`).
* **Cấu trúc đồ thị đầu vào:** **`Self` + Đồ thị 2 chiều đa nhánh** (`277` cạnh có hướng duy nhất trên `PEMS08` mở rộng hai chiều + `self-loop`; `Self + 2 chiều (Mean + Max)` đạt `Val PR-AUC = 0,3383`, tăng `+9,36%` so với `Self only` với `LRT p = 2,26e-14`).
* **Tầm dự báo tối ưu (`Δt*`):** **Mốc trọng tâm `15 phút`** (`d = 0,8067`, vùng cân bằng `10–20 phút`; dải quét `{5, 10, 15, 20, 30}` phút).

---

## 3. Cấu trúc Thư mục & Môi trường

```text
gat-traffic-warning/
├── datasets/              # Dữ liệu thô (data/) và dữ liệu sạch (cleaned/) — không đẩy lên Git
├── docs/                  # Tài liệu đề tài, quy chuẩn (rules.md, notebook-template.md) & báo cáo 01–04
├── notebooks/             # Chuỗi Jupyter Notebook thực nghiệm từ 01 đến 07
├── outputs/               # Biểu đồ 300 DPI (figures/), bảng điểm nghẽn (.csv) & đồ thị Gephi (.gexf)
└── src/                   # Mã nguồn tái sử dụng (data.py, ...)
```

* **Môi trường khuyến nghị:** Python `3.12` (`.venv`), `TensorFlow / Keras`, `NumPy`, `Pandas`, `SciPy`, `Scikit-learn`, `NetworkX`, `Matplotlib`, `Seaborn`, `SHAP`, cố định `SEED = 42`.
* **Quy chuẩn vận hành dự án:** Xem chi tiết tại [`docs/rules.md`](./docs/rules.md) và [`docs/notebook-template.md`](./docs/notebook-template.md).
