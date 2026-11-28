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
  1. **Công thức tự tính ngưỡng tắc nghẽn từ Biểu đồ cơ bản của chính dữ liệu** (kết hợp quy chuẩn liên bang **FHWA 23 CFR Part 490** và nguyên lý **HCM / Choe 2002 / Chen 2004 / Cassidy & Bertini 1999**, **không chép cứng hằng số từ bài báo**):
     * Vận tốc dòng tự do: $v_{\text{ff}} = \text{median}(v \mid o < 0{,}05)$ (`PEMS08`: `67,10 mph`).
     * Ngưỡng tắc nghẽn (**FHWA `< 60%` FFS**): $v_{\text{cong}} = 0{,}60\,v_{\text{ff}}$ (`40,26 mph`, tự khớp sát `~40 mph` của **Chen et al., 2004**); ngưỡng bắt đầu pha chuyển tiếp (*Synchronized Flow*): $v_{\text{sync}} = 0{,}80\,v_{\text{ff}}$ (`53,68 mph`).
     * Ngưỡng chiếm dụng bão hòa $o_{\text{sat}}$: điểm sớm nhất trên nhánh đi lên của đường cong `Flow(Occ)` đạt $90\%$ đỉnh công suất $Q_{\max}$ tại $o_{\text{crit}}$ (`PEMS08`: `o_sat = 0,1325`, `o_crit = 0,1675`, khớp dải `11%–15%` của **Choe 2002 / HCM**).
     * Phân chia tự động 3 pha Kerner: **Thông thoáng (Pha 0)**, **Pha chuyển tiếp Synchronized (Pha 1)**, và **Tắc thực (Pha 2: $v < v_{\text{cong}} \land o \ge o_{\text{sat}}$)**.
  2. **Trọn bộ `14` đặc trưng cục bộ (`3` biến gốc + `4` nhóm đặc trưng bổ sung trên cửa sổ nhân quả cố định `H = 12` bước = `60 phút`):**
     * **Nhóm 1 — Dao động khử dốc (`3` biến — *Kerner, 2004*):** `occ_rstd`, `spd_rstd`, `flow_rstd`.
     * **Nhóm 2 — Liên kết Đa biến (`2` cặp — *Multivariate CSD, Chattopadhyay & Gupta, RSOS 2026*):** `occ_spd_corr` (`-corr(Occ, Speed)`) và `flow_occ_decouple` (`-corr(Flow, Occ)`).
     * **Nhóm 3 — Độ dốc có dấu (`2` biến — *California / McMaster* & phần xu hướng từ bước khử dốc):** `occ_slope`, `spd_slope` (phân tách rạch ròi pha tụt tốc vào tắc với pha hồi phục thoát tắc, chống báo động giả).
     * **Nhóm 4 — Chu kỳ Thời gian (`4` biến — *STID, Shao et al., CIKM 2022*):** `tod_sin`, `tod_cos`, `dow_sin`, `dow_cos` (mã hóa tuần hoàn giờ trong ngày & thứ trong tuần, hoàn thiện đủ **`14` đặc trưng**, nâng tổng `Val PR-AUC` từ **`+22,51%` đến `+31,66%`** so với chỉ `3` biến thô).
  3. **Đối chiếu đồ thị tĩnh vs. đồ thị động (`NB03` Phần 4):** Kiểm định loại trừ `Betweenness` tĩnh (**Li et al., *PNAS* 2015**, `p = 0,3966`) và chứng minh tương tác động qua lân cận có hướng `In + Out` (**DCRNN, Li et al., 2018**; khớp cơ chế độ trễ lan truyền không gian của **PDFormer, Jiang et al., *AAAI* 2023**) nâng tiếp `Val PR-AUC` trên nền `14` đặc trưng cục bộ thêm **`+1,88%` (`5p`) đến `+5,48%` (`30p`)**.

---

### RQ2 (Chính): Đánh giá Đóng góp của Dấu hiệu Tiền tắc nghẽn trên Mô hình `ST-GAT (LSTM → GATv2)` (`NB04` trên `PEMS08`, kiểm chứng độc lập trên `PEMS04` ở `NB05`)
* **EN:** *Does augmenting a standard graph-attention traffic forecasting model (`ST-GAT` with `GATv2`) with physics-informed pre-congestion signals reduce forecasting error (`MAE`), and which signals contribute most?*
* **VI:** *Việc bổ sung các dấu hiệu tiền tắc nghẽn vào kiến trúc dự báo chú ý đồ thị tiêu chuẩn (`ST-GAT` với `GATv2`) có giúp giảm sai số dự báo (`MAE`) hay không, và từng đặc trưng/nhóm đặc trưng đóng góp ra sao?*
* **Thiết kế thí nghiệm đối chứng tối giản & chặt chẽ (`PyTorch`, chia thời gian chuẩn `60 / 20 / 20`, cửa sổ `12 → 12` bước theo *LibCity, 2021* & *BasicTS, 2025*):**
  1. **Sử dụng DUY NHẤT `1` kiến trúc chuẩn kế thừa bài báo — đóng băng hoàn toàn kiến trúc để tách riêng tác dụng của đặc trưng:**
     * Kế thừa khung **`ST-GAT`** của **Zhang et al. (*IEEE Access*, 2019)** *(bài báo gốc đã chứng minh vượt trội `HA / ARIMA / VAR`, `FC-LSTM` cục bộ và `STGCN` tĩnh — không cần chạy lại các baseline ngây thơ hay RNN cục bộ)*.
     * Nén chuỗi quá khứ `H = 12` bước (`60 phút`) tại từng trạm bằng **`LSTM` (`hidden = 32`)**, sau đó truyền tin không gian `2` bước trên đồ thị có hướng 2 chiều (`Self / In / Out`) bằng **`2-layer Multi-Head GATv2` (`4` heads)** (**Brody et al., *ICLR* 2022**), cuối cùng chiếu qua lớp tuyến tính dự báo hồi quy phần dư `12` bước tương lai (`5p → 60p`) cho cả `3` biến `[Flow, Occupancy, Speed]` với hàm mất mát thuần **`MAE` (`L1Loss`)**.
     * Mô hình quy nạp 100% (*inductive*, chia sẻ trọng số — **GraphSAGE, Hamilton et al., 2017**), chuyển trực tiếp sang `PEMS04` (`307` trạm) ở `NB05` không đổi kiến trúc.
  2. **So sánh đối chứng `6` cấu hình tập đặc trưng đầu vào (chỉ đổi `in_channels`, giữ nguyên mọi siêu tham số & `SEED = 42`):**
     * **`Raw (3)`:** `3` biến cảm biến thô `[Flow, Occupancy, Speed]` *(baseline nguyên bản ST-GAT 2019)*.
     * **`Base (7)`:** `Raw (3)` + **Nhóm 4 `STID` (`4`)** `[tod_sin, tod_cos, dow_sin, dow_cos]` (**Shao et al., 2022** — *baseline chuẩn BasicTS 2025, đồng thời kiểm chứng câu hỏi mở từ `NB03` về vai trò phi tuyến của Nhóm 4*).
     * **`Base + Nhóm 1 (10)`** (`+Dao động khử dốc`: `occ_rstd, spd_rstd, flow_rstd`), **`Base + Nhóm 2 (9)`** (`+Liên kết đa biến`: `occ_spd_corr, flow_occ_decouple`), **`Base + Nhóm 3 (9)`** (`+Độ dốc có dấu`: `occ_slope, spd_slope`).
     * **`Full (14)`:** `Base (7)` + đầy đủ **`7` chỉ báo tiền tắc nghẽn** từ RQ1.
  3. **Trọng tâm đánh giá `MAE` + Trích xuất đóng góp + Đánh giá phụ tắc nghẽn không rò rỉ:**
     * **Đánh giá chính:** **`MAE`** (kèm `RMSE` tham chiếu) cho `Speed`, `Occupancy`, `Flow` tại các mốc `15p / 30p / 60p` và trung bình `5–60p` trên cả **toàn mạng** lẫn **vùng tắc nghẽn / các trạm điểm nghẽn**.
     * **Trích xuất đóng góp:** **Permutation Feature Importance (`ΔMAE`)** trên đủ `14` đặc trưng và `4` nhóm đặc trưng ngay trên tập `Test`, kết hợp trực quan hóa **trọng số chú ý cạnh động $\alpha_{ij}$ của `GATv2`** (`Self / In / Out`) quanh các điểm nghẽn trọng yếu.
     * **Đánh giá phụ nhỏ (chống rò rỉ 100%):** Hiệu chuẩn ngưỡng tắc nghẽn `(v_cong, o_sat)` **chỉ trên tập `Train (60%)`** rồi áp dụng lên kết quả dự báo $(\hat{\text{Speed}}, \hat{\text{Occ}})$ của tập `Test (20%)` để đối chiếu gọn `F1 / Recall` đánh dấu điểm tắc nghẽn.
  4. **Kiểm chứng tổng quát sang `PEMS04` (`NB05`):** Hiệu chuẩn ngưỡng thuần `Train (60%)` của `PEMS04` ($o_{\text{sat}} = 0{,}1025$) và lặp lại **đầy đủ, đối xứng `100%` cả `6` cấu hình đặc trưng** trên cùng kiến trúc đóng băng `ST-GAT (LSTM → 2-layer GATv2)` (kèm chuyển giao *Zero-Shot* ở `0` epoch từ `NB04`, đối chiếu `Test PFI` xuyên thành phố và chú ý cạnh `GATv2`) để khẳng định tính tổng quát một cách công bằng tuyệt đối (*ceteris paribus*).

---

## 2. Lộ trình 5 Notebook Gọn nhẹ & Nhất quán

| Bước | Notebook (`notebooks/`) | Báo cáo (`docs/`) | Trạng thái | Mục tiêu duy nhất |
| :---: | :--- | :--- | :---: | :--- |
| **01** | [`01_Overview.ipynb`](../notebooks/01_Overview.ipynb) | [`01_Overview_Report.md`](./01_Overview_Report.md) | Hoàn thành | Tổng quan đề tài (v7: `RQ1–RQ2`) & kiểm toán 4 tập dữ liệu gốc PeMS. |
| **02** | [`02_Data_Cleaning_and_Interpolation.ipynb`](../notebooks/02_Data_Cleaning_and_Interpolation.ipynb) | [`02_Data_Cleaning_Report.md`](./02_Data_Cleaning_Report.md) | Hoàn thành | Làm sạch thuần **Chen et al. (2003)** (`0` lỗi trên `PEMS08` & `PEMS04`). |
| **03** | [`03_EDA_and_Bottleneck_Analysis.ipynb`](../notebooks/03_EDA_and_Bottleneck_Analysis.ipynb) | [`03_EDA_and_Bottleneck_Report.md`](./03_EDA_and_Bottleneck_Report.md) | Hoàn thành | **RQ1 (thuần `PEMS08`):** Khám phá đa biến & tương quan `3 × 3` → Ngưỡng tự tính (FHWA `<60%` FFS + Biểu đồ cơ bản) & loại trừ `Flow` → Chuyển pha, `7` chỉ báo tiền tắc nghẽn + Nhóm 4 `STID` (`14` đặc trưng, `H = 12` = `60p`) → Đối chiếu đồ thị tĩnh (`Betweenness`) vs. động (`In + Out`, khớp `PDFormer`). |
| **04** | [`04_Modeling_and_Feature_Contribution.ipynb`](../notebooks/04_Modeling_and_Feature_Contribution.ipynb) | [`04_Modeling_Report.md`](./04_Modeling_Report.md) | Hoàn thành | **RQ2 (`PyTorch`, `PEMS08`):** Đóng băng kiến trúc `ST-GAT (LSTM → 2-layer GATv2)` đối chứng `6` cấu hình đặc trưng (`Raw 3` → `Base 7` → `+G1/+G2/+G3` → `Full 14`) theo trọng tâm **`MAE`** (`Std-MAE -8,19%` vs Raw, `-2,17%` vs Base), `Permutation Feature Importance` (`14/14` dương), chú ý cạnh `GATv2` (`Out` tại `N22`: `23,8% → 39,6%`) & đánh giá phụ tắc nghẽn (ngưỡng thuần `Train 60%`). |
| **05** | [`05_Cross_Map_Validation_PEMS04.ipynb`](../notebooks/05_Cross_Map_Validation_PEMS04.ipynb) | [`05_Cross_Map_Validation_Report.md`](./05_Cross_Map_Validation_Report.md) | Hoàn thành | **Kiểm chứng đối xứng (`PEMS04`) & Kết luận:** Tự hiệu chuẩn ngưỡng thuần `Train` ($o_{\text{sat}} = 0{,}1025$), lặp lại trọn vẹn `6` cấu hình đặc trưng trên cùng kiến trúc `ST-GAT` (`Std-MAE -6,25%` vs Raw, `-1,38%` vs Base) kèm *Zero-Shot* (`0` epoch), xác nhận nhất quán `PFI` xuyên 2 thành phố (`ρ = 0,9956`), chú ý `GATv2` (`Out +22,5%`) & Kết luận toàn dự án. |

---

## 3. Nhật ký Thay đổi (Change Log)

| Ngày | Phiên bản | Chi tiết cập nhật | Trạng thái |
| :--- | :---: | :--- | :---: |
| 2026-11-28 | v1–v4 | Rà soát văn học PeMS, chuyển sang PyTorch, cập nhật `rules.md` & `requirements.txt`. | Hoàn thành |
| 2026-11-28 | v5 | **Hoàn tất Bước 2:** Đơn giản hóa làm sạch thuần **Chen et al. (2003)** (`0` lỗi trên `PEMS08` & `PEMS04`). | Hoàn thành |
| 2026-11-28 | v6 | Phát hiện lỗi áp cứng `Occ ≥ 0,15` sang `PEMS04` và nhiễu pha hồi phục của `Std` thô → tách `PEMS04` khỏi bước khám phá, nâng cấp sang dao động khử dốc + độ dốc có dấu. | Hoàn thành |
| 2026-11-28 | **v7** | **CHỐT CHÍNH THỨC ĐỀ TÀI TỐI GIẢN (5 Notebook):** Đổi RQ2 sang đối chứng ích lợi của dấu hiệu tiền tắc nghẽn trên mô hình gọn & `GAT`; bỏ RQ3, bỏ phân loại Tầng 2, bỏ Pareto; giữ chứng minh định lượng loại trừ ngưỡng `Flow` (*Capacity Drop*) và `Betweenness` tĩnh (*Li 2015*); chuyển công thức ngưỡng RQ1 sang **tự tính 100% từ Biểu đồ cơ bản của dữ liệu**; giữ `PEMS04` độc lập làm **Notebook 05** kiểm chứng chéo cuối cùng. | Đã chốt |
| 2026-11-28 | **v7.1** | **HOÀN TẤT CHẶNG 1 (`NB01 – NB03`, trọn vẹn RQ1):** Đồng bộ `NB01–NB02` sang v7; chốt `NB03` gọn nhẹ (tách `src/plots_nb03.py`, chuẩn FHWA `<60%` FFS + Biểu đồ cơ bản, `7` chỉ báo tiền tắc nghẽn đối xứng + Nhóm 4 `4` chu kỳ `STID` có biểu đồ trực quan riêng nâng `+22,51%` đến `+31,66% PR-AUC`, đối chiếu đồ thị tĩnh `p = 0,3966` vs. động `In + Out` nâng `+1,88%` đến `+5,48%` khớp `PDFormer`); tinh gọn `docs/literature_review.md` còn **`22` tài liệu tham khảo chuẩn IEEE** gắn thẳng vào `NB01–NB05`. | Hoàn thành (`NB01–03`) |
| 2026-11-28 | **v7.2** | **HOÀN TẤT CHẶNG 2 (`NB04 – NB05`, trọn vẹn RQ2 & Kết luận):** Xây dựng mô-đun `PyTorch` thuần (`src/data.py`, `models.py`, `train.py`, `explain.py`, `plots_nb04.py`, `plots_nb05.py`), đóng băng kiến trúc quy nạp `ST-GAT (LSTM → 2-layer GATv2)` đối chứng đối xứng **`6` cấu hình đặc trưng** trên cả `PEMS08` (`Std-MAE -8,19%` vs Raw / `-2,17%` vs Base) và `PEMS04` (`-6,25%` vs Raw / `-1,38%` vs Base, kèm *Zero-Shot* `0` epoch), xác nhận nhất quán `PFI` xuyên 2 thành phố (`Spearman ρ = 0,9956`) và hoàn thiện Kết luận toàn dự án. | Hoàn thành (`NB01–05`) |
