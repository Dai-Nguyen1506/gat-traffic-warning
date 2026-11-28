# Kế hoạch Nghiên cứu & Nhật ký Thay đổi (`docs/plan.md`)

Tài liệu này chốt toàn bộ khung đề tài tối giản, sắc nét sau khi rà soát văn học PeMS ([`literature_review.md`](./literature_review.md)) và thống nhất lại phạm vi nghiên cứu.

* **Ràng buộc cốt lõi:** Bài báo cuối kỳ chuẩn IEEE **tối đa 3.000 từ**, thuyết trình **10 phút**, trọng số trực quan hóa cao (`30%`), sử dụng `PEMS08` làm tập nghiên cứu chính và giữ `PEMS04` hoàn toàn độc lập cho bước kiểm chứng chéo cuối cùng (`NB05`) nhằm tăng độ uy tín cho bài báo.
* **Ngăn xếp kỹ thuật (ĐÃ CHỐT):** **PyTorch** (`Python 3.12`, `.venv`, cố định `SEED = 42`), kết hợp `NumPy`, `Pandas`, `SciPy`, `Scikit-learn`, `NetworkX`, `Matplotlib`, `Seaborn`.

---

## 1. Bản Chốt Đề tài & 2 Câu hỏi Nghiên cứu (CHỐT CHÍNH THỨC — v7)

* **Tiêu đề (EN):** *Physics-Informed Pre-Congestion Signals for Freeway Bottleneck Forecasting with Graph Attention*
* **Tiêu đề (VI):** *Tín hiệu Tiền tắc nghẽn Hướng Vật lý cho Bài toán Dự báo Điểm nghẽn Cao tốc bằng Mạng Chú ý Đồ thị*

### Tầng Tiền xử lý Bắt buộc (Pre-RQ — ĐÃ HOÀN THÀNH)
* **Làm sạch dữ liệu thuần theo Chen et al. (*Transportation Research Record*, 2003)** (`src/cleaning.py`, `NB02`): Chẩn đoán 4 chỉ số ngày ($S_1 \dots S_4$) tách bạch đường vắng ban đêm khỏi Lỗi Loại A (`0.0`) và Loại B (đóng băng dương), phục hồi bằng hồi quy tuyến tính cặp trạm lân cận ($K = 5$, trọng số $R^2$) → đạt `0` lỗi tồn đọng trên cả `PEMS08` và `PEMS04`.

---

### RQ1 (Chính): Xác nhận Tắc nghẽn Hướng Dữ liệu & Trích xuất Dấu hiệu Tiền tắc nghẽn (`NB03` — thuần `PEMS08`)
* **EN:** *How can freeway congestion states and pre-congestion volatility signals be identified directly from multi-variable sensor data?*
* **VI:** *Làm thế nào xác định trạng thái tắc nghẽn và trích xuất các dấu hiệu dao động tiền tắc nghẽn trực tiếp từ dữ liệu cảm biến đa biến?*
* **Nội dung cốt lõi (tối giản — đã bỏ Tầng 2, bỏ Pareto; giữ chứng minh loại trừ `Flow` & `Betweenness`):**
  1. **Công thức tự tính ngưỡng tắc nghẽn từ Biểu đồ cơ bản của chính dữ liệu** (kế thừa nguyên lý HCM / Choe 2002 / Chen 2004 / Cassidy & Bertini 1999, **không chép cứng hằng số từ bài báo**):
     * Vận tốc dòng tự do: $v_{\text{ff}} = \text{median}(v \mid o < 0{,}05)$.
     * Ngưỡng tắc nghẽn: $v_{\text{cong}} = 0{,}60\,v_{\text{ff}}$; ngưỡng bắt đầu pha chuyển tiếp (*Synchronized Flow*): $v_{\text{sync}} = 0{,}80\,v_{\text{ff}}$.
     * Ngưỡng chiếm dụng bão hòa $o_{\text{sat}}$: điểm sớm nhất trên nhánh đi lên của đường cong `Flow(Occ)` đạt $90\%$ đỉnh công suất $Q_{\max}$ tại $o_{\text{crit}}$.
     * Phân chia tự động 3 pha Kerner: **Thông thoáng (Pha 0)**, **Pha chuyển tiếp Synchronized (Pha 1)**, và **Tắc thực (Pha 2: $v < v_{\text{cong}} \land o \ge o_{\text{sat}}$)**.
  2. **Bộ `7` chỉ báo tiền tắc nghẽn nhân quả khử xu hướng, liên kết đa biến & gắn dấu (`H = 12` bước = `60 phút`):**
     * **Dao động khử dốc (*Kerner, 2004*):** `occ_rstd`, `spd_rstd`, `flow_rstd`.
     * **Liên kết Đa biến (*Multivariate CSD — Chattopadhyay & Gupta, RSOS 2026*):** `occ_spd_corr` (`-corr(Occ, Speed)`) và `flow_occ_decouple` (`-corr(Flow, Occ)`).
     * **Độ dốc có dấu (*California / McMaster* & phần xu hướng từ bước khử dốc):** `occ_slope`, `spd_slope` (phân tách rạch ròi pha tụt tốc `< 0` với pha hồi phục thoát tắc `> 0`, chống báo động giả).
  3. **Đối chiếu đồ thị tĩnh vs. đồ thị động (`NB03` Phần 4):** Kiểm định loại trừ `Betweenness` tĩnh (**Li et al., 2015**, `p = 0,3966`) và chứng minh tương tác động qua lân cận có hướng `In + Out` (**DCRNN, 2018**) nâng `Val PR-AUC` thêm `+1,35%` (`5p`) đến `+5,53%` (`30p`).

---

### RQ2 (Chính): Đánh giá Đóng góp của Dấu hiệu Tiền tắc nghẽn trên Mô hình Dự báo Đồ thị & GAT (`NB04` trên `PEMS08`, kiểm chứng độc lập trên `PEMS04` ở `NB05`)
* **EN:** *Does augmenting graph-based forecasting models (including GAT) with pre-congestion signals improve future congestion prediction, and how much does each signal contribute?*
* **VI:** *Việc bổ sung các dấu hiệu tiền tắc nghẽn có giúp các mô hình dự báo đồ thị (gồm GAT) dự đoán điểm tắc nghẽn tương lai tốt hơn không, và từng dấu hiệu đóng góp ra sao?*
* **Thiết kế thí nghiệm đối chứng gọn nhẹ & chặt chẽ (`PyTorch`, chia thời gian chuẩn `60 / 20 / 20`, cửa sổ `12 → 12` bước):**
  1. **So sánh trực diện hai tập đặc trưng trên cùng bộ mô hình gọn nhẹ:**
     * **(A) Đặc trưng thô (`Base`):** `[Flow, Occupancy, Speed]` + chu kỳ thời gian `(sin/cos)`.
     * **(B) Bổ sung dấu hiệu tiền tắc nghẽn (`+ Pre-Congestion Signals`):** `Base` + `7` chỉ báo từ RQ1 (`occ_rstd, spd_rstd, flow_rstd, occ_spd_corr, flow_occ_decouple, occ_slope, spd_slope`).
  2. **3 mô hình gọn nhẹ chạy nhanh trên CPU:** `Persistence` (đường cơ sở giữ nguyên), `Temporal GRU` (chỉ chuỗi thời gian, không đồ thị), và **`ST-GAT` (`GRU + GATv2` gọn nhẹ có lưu trọng số chú ý cạnh)**.
  3. **Phân tích đóng góp từng dấu hiệu:** Chạy **Ablation từng nhóm tín hiệu** (`+rstd` / `+coupling` / `+slope` / `All 7`), **Permutation Feature Importance**, và trực quan hóa **trọng số chú ý cạnh của GAT** quanh các nút tắc nghẽn.
  4. **Kiểm chứng tổng quát sang `PEMS04` (`NB05`):** Áp nguyên si công thức tự tính ngưỡng từ dữ liệu `PEMS04` và lặp lại thí nghiệm đối chứng `Base` vs. `+ Signals` để khẳng định tính tổng quát của kết luận nghiên cứu.

---

## 2. Lộ trình 5 Notebook Gọn nhẹ & Nhất quán

| Bước | Notebook (`notebooks/`) | Báo cáo (`docs/`) | Trạng thái | Mục tiêu duy nhất |
| :---: | :--- | :--- | :---: | :--- |
| **01** | [`01_Overview.ipynb`](../notebooks/01_Overview.ipynb) | [`01_Overview_Report.md`](./01_Overview_Report.md) | Hoàn thành | Tổng quan đề tài (v7: `RQ1–RQ2`) & kiểm toán 4 tập dữ liệu gốc PeMS. |
| **02** | [`02_Data_Cleaning_and_Interpolation.ipynb`](../notebooks/02_Data_Cleaning_and_Interpolation.ipynb) | [`02_Data_Cleaning_Report.md`](./02_Data_Cleaning_Report.md) | Hoàn thành | Làm sạch thuần **Chen et al. (2003)** (`0` lỗi trên `PEMS08` & `PEMS04`). |
| **03** | [`03_EDA_and_Bottleneck_Analysis.ipynb`](../notebooks/03_EDA_and_Bottleneck_Analysis.ipynb) | [`03_EDA_and_Bottleneck_Report.md`](./03_EDA_and_Bottleneck_Report.md) | Hoàn thành | **RQ1 (thuần `PEMS08`):** Khám phá đa biến & tương quan `3 × 3` → Ngưỡng tự tính (FHWA `<60%` FFS + Biểu đồ cơ bản) & loại trừ `Flow` → Chuyển pha & `7` chỉ báo trọn chu kỳ (`H = 12` = `60p`) → Đối chiếu đồ thị tĩnh (`Betweenness`) vs. động (`In + Out`). |
| **04** | `04_Modeling_and_Feature_Contribution.ipynb` | `04_Modeling_Report.md` *(kế hoạch)* | Tiếp theo | **RQ2 (`PyTorch`, `PEMS08`):** Đối chứng `Base` vs. `+ Signals` trên `Persistence / GRU / ST-GAT`, phân tích đóng góp từng đặc trưng & trọng số chú ý `GAT`. |
| **05** | `05_Cross_Map_Validation_PEMS04.ipynb` | `05_Cross_Map_Validation_Report.md` *(kế hoạch)* | Kế hoạch | **Kiểm chứng độc lập trên `PEMS04`:** Tự hiệu chỉnh ngưỡng từ Biểu đồ cơ bản `PEMS04` + xác nhận lại ích lợi của dấu hiệu tiền tắc nghẽn trên bản đồ mới. |

---

## 3. Nhật ký Thay đổi (Change Log)

| Ngày | Phiên bản | Chi tiết cập nhật | Trạng thái |
| :--- | :---: | :--- | :---: |
| 2026-11-28 | v1–v4 | Rà soát 35 bài báo PeMS, chuyển sang PyTorch, cập nhật `rules.md` & `requirements.txt`. | Hoàn thành |
| 2026-11-28 | v5 | **Hoàn tất Bước 2:** Đơn giản hóa làm sạch thuần **Chen et al. (2003)** (`0` lỗi trên `PEMS08` & `PEMS04`). | Hoàn thành |
| 2026-11-28 | v6 | Phát hiện lỗi áp cứng `Occ ≥ 0,15` sang `PEMS04` và nhiễu pha hồi phục của `Std` thô → tách `PEMS04` khỏi bước khám phá, nâng cấp sang dao động khử dốc + độ dốc có dấu. | Hoàn thành |
| 2026-11-28 | **v7** | **CHỐT CHÍNH THỨC ĐỀ TÀI TỐI GIẢN (5 Notebook):** Đổi RQ2 sang đối chứng ích lợi của dấu hiệu tiền tắc nghẽn trên mô hình gọn & `GAT`; bỏ RQ3, bỏ phân loại Tầng 2, bỏ Pareto; giữ chứng minh định lượng loại trừ ngưỡng `Flow` (*Capacity Drop*) và `Betweenness` tĩnh (*Li 2015*); chuyển công thức ngưỡng RQ1 sang **tự tính 100% từ Biểu đồ cơ bản của dữ liệu**; giữ `PEMS04` độc lập làm **Notebook 05** kiểm chứng chéo cuối cùng. | Đã chốt |
| 2026-11-28 | **v7.1** | **HOÀN TẤT & ĐẨY GIT CHẶNG 1 (`NB01 – NB03`, trọn vẹn RQ1):** Đồng bộ `NB01–NB02` sang v7; chốt `NB03` (`117` dòng code, tách `src/plots_nb03.py`, chuẩn FHWA `<60%` FFS + Biểu đồ cơ bản, bộ `7` chỉ báo đối xứng với Nhóm 2 liên kết đa biến `occ_spd_corr` & `flow_occ_decouple` nâng `+21,15%` đến `+31,33% PR-AUC`, đối chiếu đồ thị tĩnh vs. động `In + Out` nâng `+1,35%` đến `+5,53%`); tinh gọn `docs/literature_review.md` theo đúng các bài báo sử dụng; **sẵn sàng bắt đầu `NB04` (`RQ2` bằng PyTorch)**. | Hoàn thành (`NB01–03`) |
