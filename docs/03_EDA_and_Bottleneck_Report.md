# Báo cáo Bước 03: Khám phá Dữ liệu, Xác định Trạng thái Tắc nghẽn và Phân tích Dấu hiệu Tiền tắc nghẽn trên `PEMS08`

* **Tệp nguồn:** [`03_EDA_and_Bottleneck_Analysis.ipynb`](../notebooks/03_EDA_and_Bottleneck_Analysis.ipynb), mô-đun tính toán [`src/bottlenecks.py`](../src/bottlenecks.py) và mô-đun biểu đồ [`src/plots_nb03.py`](../src/plots_nb03.py)
* **Dữ liệu đầu vào:** `datasets/cleaned/PEMS08_clean.npz` (`17.856 × 170 × 3`, `62` ngày từ `01/07` đến `31/08/2016`, `277` cạnh có hướng duy nhất, `0` lỗi tồn đọng từ Bước 02). Tập `PEMS04` giữ độc lập hoàn toàn cho bước kiểm chứng chéo cuối cùng (`NB05`).
* **Tệp đầu ra bàn giao:** [`PEMS08_bottlenecks.csv`](../outputs/PEMS08_bottlenecks.csv), [`PEMS08_network.gexf`](../outputs/PEMS08_network.gexf) cùng `5` biểu đồ `300 DPI` tại `outputs/figures/nb03_*.png`.

---

## 1. Phân tích Khám phá Đa biến và Cấu trúc Tương quan trên `PEMS08`

| Biến đo lường | Đơn vị | Trung bình | Độ lệch chuẩn | Nhỏ nhất | Trung vị | Lớn nhất |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Lưu lượng (Flow)`** | `xe / 5 phút` | 230,8092 | 146,1292 | 0,0000 | 215,0000 | 1.147,0000 |
| **`Độ chiếm dụng (Occupancy)`** | `tỷ lệ [0, 1]` | 0,0651 | 0,0459 | 0,0000 | 0,0601 | 0,8955 |
| **`Vận tốc (Speed)`** | `mph` | 63,7633 | 6,6499 | 3,0000 | 64,9000 | 82,3000 |

* **Ma trận tương quan Pearson `3 × 3` trên toàn tập (`3.035.520` điểm đo):**

| Biến | `Flow` | `Occupancy` | `Speed` |
| :--- | :---: | :---: | :---: |
| **`Flow`** | `1,0000` | `+0,6735` | `-0,2968` |
| **`Occupancy`** | `+0,6735` | `1,0000` | **`-0,7531`** |
| **`Speed`** | `-0,2968` | **`-0,7531`** | `1,0000` |

* **Nhịp điệu cao điểm sáng–chiều:** Trên `10` trạm chịu tải nặng nhất (`[22, 167, 13, 14, 4, 71, 128, 65, 100, 117]`), chuỗi 14 ngày của đủ `3` biến và nhiệt đồ Thứ × Giờ thể hiện rõ **cao điểm sáng T2–T6 (`07–09h`, TB `48,66 mph`)**, **cao điểm chiều T2–T6 (`14–18h`, TB `45,45 mph`, thấp nhất Thứ Sáu lúc `15h` xuống `37,55 mph`)**, và cuối tuần thông thoáng (**`59,75 mph`**).
* **Biểu đồ minh họa:** [`nb03_pems08_overview.png`](../outputs/figures/nb03_pems08_overview.png) *(Chuỗi 14 ngày đủ `3` biến Vận tốc – Độ chiếm dụng – Lưu lượng TB + Nhiệt đồ Thứ × Giờ + Ma trận Heatmap tương quan `3 × 3`)*.

---

## 2. Xác định Ngưỡng Tắc nghẽn từ Biểu đồ Cơ bản và Lý giải Loại trừ Lưu lượng

Áp dụng quy chuẩn liên bang Hoa Kỳ **FHWA (23 CFR Part 490 / NPMRDS)** kết hợp nguyên lý Biểu đồ cơ bản (**Chen et al., 2004**; **Choe et al., 2002**; **Cassidy & Bertini, 1999**), Phần 2 xác định ranh giới nhị phân **`Thông thoáng` vs. `Tắc thực`** trực tiếp từ dữ liệu `PEMS08`:

| Tham số tự tính trên `PEMS08` | Giá trị tính ra | Cơ sở & Đối chiếu lý thuyết |
| :--- | :---: | :--- |
| **Vận tốc tự do ($v_{\text{ff}} = \text{median}(v \mid o < 0{,}05)$)** | **`67,10 mph`** | Tốc độ dòng tự do đặc trưng của mạng |
| **Ngưỡng tắc nghẽn ($v_{\text{cong}} = 0{,}60\,v_{\text{ff}}$)** | **`40,26 mph`** | Quy chuẩn **FHWA (`< 60%` FFS)**; tự khớp sát `~40 mph` (Chen et al., 2004) |
| **Đỉnh công suất Biểu đồ cơ bản ($o_{\text{crit}}$ / $Q_{\max}$)** | **`0,1675` (`438,6 xe/5p`)** | Điểm bắt đầu hiệu ứng *Capacity Drop* (Cassidy & Bertini, 1999) |
| **Ngưỡng chiếm dụng tối thiểu ($o_{\text{sat}}$ tại `90% Q_max` nhánh lên)** | **`0,1325` (`13,25%`)** | Ước lượng hướng dữ liệu khớp dải `11%–15%` (Choe et al., 2002 / HCM) |

* **Điều kiện Tắc thực:** $C_i(t) = \mathbb{I}[v_i(t) < 40{,}26 \land o_i(t) \ge 0{,}1325]$ xác nhận **`59.003` điểm (`1,94%`, lọc sạch `4.752` điểm chạy chậm khi đường vắng, `99,51%` trong `05–23h`)**.
* **Lý giải loại trừ ngưỡng `Flow`:** Khi tắc cứng quá tải (`Occ ≥ 0,30`, `Speed TB = 23,7 mph`), `Flow` tụt ngược **`-13,7%`** so với vùng đỉnh (`433,6 → 374,2 xe/5p`, trung vị `367,0`, thấp hơn cả lúc chạy thoáng `61,6 mph` có trung vị `400,0`). Trong dải `Flow ∈ [300, 450]` (`672.898` điểm), có tới **`90,7%` đang thông thoáng** và chỉ `3,4%` bị tắc (`Flow AUC = 0,7586` so với `Occ AUC = 0,9781`).
* **Biểu đồ minh họa:** [`nb03_fundamental_and_workday.png`](../outputs/figures/nb03_fundamental_and_workday.png)

---

## 3. Phân tích Chuyển pha và Đánh giá Hệ chỉ báo Tiền tắc nghẽn

Từ điều kiện tắc thực ở Phần 2, Phần 3 xác lập **Pha chuyển tiếp tiền tắc nghẽn** (*Synchronized Flow* — **Kerner, 2004**) với $v_{\text{sync}} = 0{,}80\,v_{\text{ff}} = \mathbf{53{,}68\text{ mph}}$ (`152.114` điểm = `5,01%`) và cố định cửa sổ nhân quả chuẩn **`H = 12` bước (`60 phút`)** đồng bộ với ma trận đầu vào `NB04` để trích xuất đủ **`7` chỉ báo mạnh** thuộc `3` nhóm:

1. **Nhóm 1 — Dao động khử dốc (*Kerner, 2004*):** `occ_rstd`, `spd_rstd`, `flow_rstd`. Tại `t = 0`, `occ_rstd` và `spd_rstd` tăng vọt **`2,29× – 2,37×`** so với mốc `-60p`, còn `flow_rstd` chỉ tăng `1,12×`.
2. **Nhóm 2 — Liên kết Đa biến (*Multivariate CSD* — Chattopadhyay & Gupta, *RSOS* 2026):** Kế thừa kết luận của bài báo rằng các chỉ báo đa biến vượt trội chỉ báo đơn biến, sử dụng 2 cặp tương quan chéo nhất quán theo Biểu đồ cơ bản: **`occ_spd_corr` (`-corr(Occ, Speed)`)** tăng từ **`+0,552 → +0,905`** (`AUC = 0,8490`, `≥55 mph: 0,7304`) và **`flow_occ_decouple` (`-corr(Flow, Occ)`)** bứt phá từ **`-0,648 → +0,135`** (`AUC = 0,8217`, `≥55 mph: 0,6439`).
3. **Nhóm 3 — Độ dốc có dấu (*California / McMaster* & phần xu hướng từ bước khử dốc):** `occ_slope` và `spd_slope`. Trên trọn chu kỳ **`-60p → +90p`** của **`n = 3.199` ca mới sạch** (`48,4%` trong `6.611` ca trên `170` trạm), khi bước sang pha hồi phục (`t = +60p`) dù `occ_rstd` còn cao gấp **`2,41×`**, cặp độ dốc lập tức **đảo dấu đối xứng** (`spd_slope = +0,85 mph/5p`, `occ_slope = -0,0042/5p`) để chặn báo động giả.
4. **Nhóm 4 — Chu kỳ Thời gian (*STID* — Shao et al., *CIKM* 2022):** `tod_sin, tod_cos, dow_sin, dow_cos` (`4` biến) mã hóa khung giờ cao điểm và ngày thường/cuối tuần — bổ sung tiếp ý nghĩa thống kê rất rõ ở tầm xa (`p = 1,56e-19` ở `30p`), hoàn thiện cơ sở định lượng cho trọn bộ **`14` đặc trưng cục bộ**.

### Đánh giá định lượng toàn mạng (`Δt = 15p` đơn biến & `LRT` từng bước trên chia `60 / 20 / 20`)

* **Sức mạnh đơn biến (`n = 2.928.641` điểm thông sạch 60p):** Toàn bộ `7` chỉ báo đều đạt mạnh và đồng đều (`AUC = 0,78–0,93`, $p < 10^{-300}$): `occ_rstd` **`0,9318`** (`RR 36,42×`, `≥55 mph: 0,8613`), `spd_rstd` **`0,9044`** (`25,67×`, `0,7983`), **`occ_spd_corr` `0,8490`** (`13,54×`, `0,7304`), **`flow_occ_decouple` `0,8217`** (`10,49×`, `0,6439`), `spd_slope` `0,8113`, `occ_slope` `0,8101`, `flow_rstd` `0,7784`.
* **Giá trị cộng thêm từng bước `LRT` (`60 / 20 / 20`):**

| Tầm cảnh báo (`Δt`) | `Raw 3 biến` | `+7 Chỉ báo (10)` | **`+4 Chu kỳ STID (Chốt 14)`** | Tổng tăng `PR-AUC` (%) | `p` *(+7 Chỉ báo)* | `p` *(+4 STID)* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`5 phút`** | 0,1888 | 0,2480 | **0,2486** | **`+31,66%`** | `3,20e-53` | `7,32e-06` |
| **`15 phút`** | 0,1483 | 0,1797 | **0,1817** | **`+22,51%`** | `7,68e-129` | `6,28e-05` |
| **`30 phút`** | 0,1168 | 0,1437 | **0,1448** | **`+23,99%`** | `1,04e-224` | `1,56e-19` |

* **Biểu đồ minh họa:** [`nb03_precongestion_trajectories.png`](../outputs/figures/nb03_precongestion_trajectories.png) · [`nb03_precongestion_validation.png`](../outputs/figures/nb03_precongestion_validation.png)

---

## 4. Phân tích Cấu trúc Đồ thị: Đối chiếu Hình học Tĩnh và Tương tác Động Lân cận

* **Mặt 1 — Hình học đồ thị TĨNH (`Li et al., PNAS 2015`) không nhận biết được điểm tắc (`47 / 170` trạm tắc `≥ 1%`):**

| Chỉ số đồ thị tĩnh (`PEMS08`, `N = 170`) | `Spearman ρ` | `p-value` | Kết luận kiểm định giả thuyết |
| :--- | :---: | :---: | :--- |
| **`Weighted Betweenness Centrality`** | **`0,0654`** | **`0,3966`** | **Chấp nhận $H_0$ ($p > 0,05$): hình học tĩnh không nhận biết được điểm tắc** |
| `Degree` (Tổng bậc vào + ra) | `0,1859` | `0,0152` | Tương quan rất yếu |

  *(Phản ví dụ trực tiếp: `N9` tắc `18,1%` có `Betweenness = 0,0000`, `N65` tắc `15,7%` có `Betweenness = 0,0047`).*

* **Mặt 2 — Tương tác ĐỘNG trên cạnh có hướng (`Lân cận Đầu vào In` & `Đầu ra Out` — DCRNN, `Li et al., 2018`; PDFormer, `Jiang et al., AAAI 2023`):**

| Tầm cảnh báo (`Δt`) | Cục bộ (`14 đặc trưng`) | `+ Lân cận In` (Đầu vào) | `+ Lân cận Out` (Đầu ra) | **`+ Cả 2 chiều (In + Out)`** | Tăng `PR-AUC` (%) | `LRT χ²` (`df = 6`) | `p` *(LRT)* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`5 phút`** | 0,2486 | 0,2520 | 0,2506 | **0,2533** | **`+1,88%`** | **33,0** | **`1,03e-05`** |
| **`15 phút`** | 0,1817 | 0,1849 | 0,1859 | **0,1907** | **`+4,96%`** | **197,4** | **`6,81e-40`** |
| **`30 phút`** | 0,1448 | 0,1469 | 0,1519 | **0,1528** | **`+5,48%`** | **409,2** | **`2,98e-85`** |

  *Mức tăng bứt phá ở tầm xa `15–30 phút` (với chiều `Out` sóng tắc dội ngược từ hạ nguồn nổi trội) khớp trực tiếp với cơ chế độ trễ lan truyền không gian của **PDFormer (2023)** — là căn cứ đưa trực tiếp đồ thị có hướng 2 chiều vào mô hình chú ý đồ thị ở `NB04` (không cần tạo biến hàng xóm thủ công).*
* **Biểu đồ minh họa:** [`nb03_spatial_and_betweenness.png`](../outputs/figures/nb03_spatial_and_betweenness.png)

---

## 5. Tổng kết Câu hỏi Nghiên cứu 1 và Thiết lập Đầu vào cho Mô hình hóa

* **Đầu ra chốt trên `PEMS08`:** Ngưỡng tự tính `v_cong = 40,26 mph`, `o_sat = 0,1325`, `v_sync = 53,68 mph`; ma trận đầu vào cố định `H = 12` bước (`60 phút`) × `14` đặc trưng chuẩn hóa (`3` biến gốc + `7` chỉ báo tiền tắc nghẽn nhân quả + `4` chu kỳ `sin/cos`) cùng đồ thị có hướng `outputs/PEMS08_network.gexf` và `outputs/PEMS08_bottlenecks.csv` bàn giao sang `NB04` (`RQ2`).
