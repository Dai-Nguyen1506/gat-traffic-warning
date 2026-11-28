# Physics-Informed Pre-Congestion Signals for Freeway Bottleneck Forecasting with Graph Attention

**Tín hiệu Tiền tắc nghẽn Hướng Vật lý cho Bài toán Dự báo Điểm nghẽn Cao tốc bằng Mạng Chú ý Đồ thị**

Dự án nghiên cứu cuối kỳ môn *Data Visualization & Analytics* (bài báo chuẩn IEEE $\le 3.000$ từ) trên hai mạng lưới cảm biến cao tốc Caltrans PeMS (`PEMS08` làm tập nghiên cứu chính và `PEMS04` giữ độc lập hoàn toàn cho bước kiểm chứng chéo cuối cùng `NB05`). Chi tiết định hướng xem tại [`docs/plan.md`](./docs/plan.md) và tổng quan 22 tài liệu tham khảo chuẩn IEEE tại [`docs/literature_review.md`](./docs/literature_review.md).

---

## 1. Lộ trình 5 Notebook Gọn nhẹ & Báo cáo

| Bước | Notebook (`notebooks/`) | Báo cáo (`docs/`) | Trạng thái | Mục tiêu chính |
| :---: | :--- | :--- | :---: | :--- |
| **01** | [`01_Overview.ipynb`](./notebooks/01_Overview.ipynb) | [`01_Overview_Report.md`](./docs/01_Overview_Report.md) | Hoàn thành | Kiểm toán 4 tập PeMS; chọn `PEMS08` (Primary) và `PEMS04` (Held-out Validation). |
| **02** | [`02_Data_Cleaning_and_Interpolation.ipynb`](./notebooks/02_Data_Cleaning_and_Interpolation.ipynb) | [`02_Data_Cleaning_Report.md`](./docs/02_Data_Cleaning_Report.md) | Hoàn thành | Chẩn đoán 4 chỉ số $S_1 \dots S_4$ & hồi quy cặp lân cận thuần **Chen et al. (2003)** (`0` lỗi tồn đọng). |
| **03** | [`03_EDA_and_Bottleneck_Analysis.ipynb`](./notebooks/03_EDA_and_Bottleneck_Analysis.ipynb) | [`03_EDA_and_Bottleneck_Report.md`](./docs/03_EDA_and_Bottleneck_Report.md) | Hoàn thành | **RQ1 (thuần `PEMS08`):** Khám phá đa biến & tương quan `3 × 3`, công thức tự tính ngưỡng (FHWA `<60%` FFS + Biểu đồ cơ bản) + loại trừ `Flow`, kiểm chứng `7` chỉ báo tiền tắc + `4` chu kỳ `STID` (`+22,51%` đến `+31,66% PR-AUC`), đối chiếu đồ thị tĩnh (`p = 0,3966`) vs. động `In + Out` (`+1,88%` đến `+5,48%`, khớp `PDFormer`). |
| **04** | `04_Modeling_and_Feature_Contribution.ipynb` | `04_Modeling_Report.md` *(kế hoạch)* | Tiếp theo | **RQ2 (`PyTorch`, `PEMS08`):** Thí nghiệm đối chứng `Base` vs. `+ 7 chỉ báo tiền tắc nghẽn` trên `Persistence`, `GRU`, `ST-GAT`; phân tích đóng góp từng nhóm đặc trưng & chú ý cạnh `GAT`. |
| **05** | `05_Cross_Map_Validation_PEMS04.ipynb` | `05_Cross_Map_Validation_Report.md` *(kế hoạch)* | Kế hoạch | **Kiểm chứng độc lập (`PEMS04`):** Tự hiệu chỉnh ngưỡng từ Biểu đồ cơ bản `PEMS04` và xác nhận lại ích lợi của dấu hiệu tiền tắc nghẽn trên bản đồ mới. |

---

## 2. Bộ Công thức & Đặc trưng Đã Chốt (Từ Bước 01 – 03 / RQ1 trên `PEMS08`)

* **Tiền xử lý chuẩn:** Thuần **Chen et al. (TRR 2003)** (`src/cleaning.py`) — đạt `0` lỗi tồn đọng trên `PEMS08` & `PEMS04`.
* **Công thức tự tính ngưỡng từ Biểu đồ cơ bản (`src/bottlenecks.py` — không chép cứng hằng số):**
  * $v_{\text{ff}} = \text{median}(v \mid o < 0{,}05)$, $v_{\text{cong}} = 0{,}60\,v_{\text{ff}}$ (**FHWA 23 CFR Part 490**), $v_{\text{sync}} = 0{,}80\,v_{\text{ff}}$, và $o_{\text{sat}}$ là điểm sớm nhất trên nhánh đi lên đạt $90\%$ đỉnh công suất $Q_{\max}$.
  * **Tắc thực ($C = 1$):** $(v < v_{\text{cong}}) \land (o \ge o_{\text{sat}})$; **Pha chuyển tiếp (*Synchronized*):** chưa tắc nhưng $(v < v_{\text{sync}}) \lor (o \ge o_{\text{sat}})$.
* **Bộ `14` đặc trưng cục bộ (`H = 12` = `60 phút`):** `3` biến gốc (`Flow, Occ, Speed`) + `7` chỉ báo tiền tắc nghẽn nhân quả (Dao động khử dốc `occ_rstd, spd_rstd, flow_rstd`, Liên kết đa biến `occ_spd_corr, flow_occ_decouple` — *Chattopadhyay & Gupta, 2026*, Độ dốc có dấu `occ_slope, spd_slope`) + `4` chu kỳ thời gian `STID` (`tod_sin, tod_cos, dow_sin, dow_cos` — *Shao et al., 2022*).

---

## 3. Cấu trúc Thư mục & Môi trường

```text
gat-traffic-warning/
├── datasets/              # Dữ liệu thô (data/) và dữ liệu sạch (cleaned/) — không đẩy lên Git
├── docs/                  # Kế hoạch (plan.md), tổng quan văn học (literature_review.md), quy chuẩn & báo cáo
├── notebooks/             # 5 Notebook thực nghiệm (01–05)
├── outputs/               # Biểu đồ 300 DPI (figures/), bảng điểm tắc PEMS08 (.csv) & đồ thị (.gexf)
└── src/                   # Mã nguồn tái sử dụng (cleaning.py, bottlenecks.py, plots_nb03.py, ...)
```

* **Môi trường khuyến nghị:** Python `3.12` (`.venv`), **PyTorch** (`torch 2.15.0+cpu`), `NumPy`, `Pandas`, `SciPy`, `Scikit-learn`, `NetworkX`, `Matplotlib`, `Seaborn`, cố định `SEED = 42`.
* **Quy chuẩn vận hành dự án:** Xem chi tiết tại [`docs/rules.md`](./docs/rules.md) và [`docs/notebook-template.md`](./docs/notebook-template.md).
