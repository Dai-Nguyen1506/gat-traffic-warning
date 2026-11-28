# Báo cáo Bước 04: Mô hình hóa `ST-GAT (LSTM → GATv2)` và Đánh giá Đóng góp Các Nhóm Đặc trưng trên `PEMS08`

* **Tệp nguồn:** [`04_Modeling_and_Feature_Contribution.ipynb`](../notebooks/04_Modeling_and_Feature_Contribution.ipynb) · Mã nguồn `PyTorch`: [`src/data.py`](../src/data.py), [`src/models.py`](../src/models.py), [`src/train.py`](../src/train.py), [`src/explain.py`](../src/explain.py), [`src/plots_nb04.py`](../src/plots_nb04.py).
* **Dữ liệu đầu vào:** `datasets/cleaned/PEMS08_clean.npz` (`17.856 × 170 × 3`, `62` ngày, `0` lỗi tồn đọng) cùng đồ thị có hướng 2 chiều `718` cạnh (`170 Self + 277 In thượng nguồn + 271 Out hạ nguồn`). Tập `PEMS04` giữ độc lập hoàn toàn cho Bước 05 (`NB05`).
* **Tệp đầu ra bàn giao:** `6` checkpoint PyTorch (`outputs/models/stgatv2_pems08_*.pt`), `3` bảng định lượng ([`nb04_ablation_summary.csv`](../outputs/nb04_ablation_summary.csv), [`nb04_horizon_results.csv`](../outputs/nb04_horizon_results.csv), [`nb04_permutation_importance.csv`](../outputs/nb04_permutation_importance.csv)) và `4` biểu đồ `300 DPI` tại `outputs/figures/nb04_*.png`.

---

## 1. Thiết lập Chống rò rỉ `60 / 20 / 20` và Ngưỡng Hiệu chuẩn Thuần `Train`

Áp dụng chuẩn chia thời gian **`60 / 20 / 20`** và cửa sổ **`12 → 12` bước** (`60p` quá khứ → `60p` tương lai — **LibCity, 2021**; **BasicTS, 2025**), toàn bộ `StandardScaler`, tập **`43 / 170` trạm điểm nghẽn** (`tần suất tắc >= 1%`, Top 5: `[22, 13, 14, 4, 9]`), và **ngưỡng điều kiện tắc nghẽn** đều được hiệu chuẩn **duy nhất trên tập `Train (60%)`** trước khi áp lên `Validation` và `Test`:

* **Ngưỡng tắc nghẽn thuần `Train (60%)`:** $v_{\text{ff}} = 67{,}10\text{ mph}$, $v_{\text{cong}} = \mathbf{40{,}26\text{ mph}}$ (trùng khớp tuyệt đối với `NB03`), và $o_{\text{sat}} = \mathbf{0{,}1375}$ (nằm gọn trong dải `11%–15%` của **Choe et al., 2002 / HCM**).
* **Cách ly biên nhân quả:** Dịch vào `22` bước (`2H − 2`) tại đầu mỗi phân đoạn để `7` chỉ báo tiền tắc nghẽn nhân quả không bao giờ chạm dữ liệu phân đoạn trước.

| Phân đoạn (`60 / 20 / 20`) | Khoảng chỉ số thời gian | Số ngày (`~`) | Số mẫu cửa sổ (`12 → 12`) | Tỷ lệ Tắc thực (`%`, ngưỡng Train) |
| :--- | :---: | :---: | :---: | :---: |
| **`Train (60%)`** | `[0, 10712]` | 37,2 | 10.679 | 1,81% |
| **`Validation (20%)`** | `[10713, 14283]` | 12,4 | 3.537 | 1,97% |
| **`Test (20%)`** | `[14284, 17855]` | 12,4 | 3.538 | 2,24% |

---

## 2. Đối chứng `6` Cấu hình Tập Đặc trưng trên Kiến trúc Cố định `ST-GAT (LSTM → GATv2)`

Kế thừa khung **`ST-GAT`** (**Zhang et al., *IEEE Access* 2019** — đã chứng minh vượt `HA / ARIMA / VAR`, `FC-LSTM` cục bộ và `STGCN` tĩnh), kiến trúc duy nhất được đóng băng `100%` qua mọi cấu hình đầu vào: **`LSTM` (`hidden = 32`) nén chuỗi quá khứ `H = 12` bước trước → `2-layer Multi-Head GATv2` (`4` heads, phân loại cạnh `Self / In / Out` — Brody et al., *ICLR* 2022) → Đầu ra hồi quy phần dư `12` bước (`L1Loss / MAE`, `AdamW lr = 5e-3`, `20` epochs, `SEED = 42`)**:

| Cấu hình đầu vào | Số kênh | Tham số | `Val L1` | `Test Std-MAE` | `MAE Speed` (`mph`) | `RMSE Speed` | `MAE Occ` | `MAE Flow` (`xe/5p`) | `MAE Speed Điểm nghẽn` | So với `Raw` | So với `Base` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Raw (3)`** *(ST-GAT gốc)* | 3 | 11.524 | 0,1806 | 0,1771 | 1,469 | 3,432 | 0,00777 | 19,21 | 2,681 | `+0,00%` | `+6,56%` |
| **`Base (7)`** *(`+Nhóm 4 STID`)* | 7 | 12.036 | 0,1693 | 0,1662 | 1,397 | 3,327 | 0,00727 | 17,66 | 2,589 | `-6,15%` | `+0,00%` |
| **`Base + Nhóm 1 (10)`** *(Dao động)* | 10 | 12.420 | 0,1675 | 0,1652 | 1,393 | 3,298 | 0,00722 | 17,51 | 2,572 | `-6,72%` | `-0,60%` |
| **`Base + Nhóm 2 (9)`** *(Liên kết)* | 9 | 12.292 | 0,1682 | 0,1656 | 1,400 | 3,330 | 0,00723 | 17,48 | 2,591 | `-6,49%` | `-0,36%` |
| **`Base + Nhóm 3 (9)`** *(Độ dốc)* | 9 | 12.292 | 0,1685 | 0,1659 | 1,395 | 3,326 | 0,00726 | 17,63 | 2,589 | `-6,32%` | `-0,18%` |
| **`Full (14)`** *(Gộp cả 4 nhóm)* | **14** | 12.932 | **0,1647** | **0,1626** | **1,371** | **3,304** | **0,00712** | **17,20** | **2,550** | **`-8,19%`** | **`-2,17%`** |

* **Phát hiện chính:**
  1. **Nhóm 4 `STID` bứt phá trên mô hình phi tuyến:** Giải đáp trọn vẹn câu hỏi mở từ `NB03` — khi chuyển từ hồi quy tuyến tính sang `ST-GAT` phi tuyến, `Raw (3) → Base (7)` giảm mạnh `Std-MAE` **`-6,15%`** (`MAE Flow -8,07%`, `MAE Occ -6,44%`, `MAE Speed -4,90%`).
  2. **Thứ hạng độc lập của `3` nhóm tín hiệu trùng khớp `100%` với `NB03`:** **Nhóm 1 Dao động (`0,1652`) > Nhóm 2 Liên kết (`0,1656`) > Nhóm 3 Độ dốc (`0,1659`)** trên cả `Val L1` lẫn `Test Std-MAE`.
  3. **Hiệu ứng cộng hưởng khi gộp đủ `14` đặc trưng (`Full 14`):** Mức giảm `Std-MAE` của `Full (14)` so với `Base (7)` đạt **`-0,0036` (`-2,17%`)**, gần **gấp đôi** tổng mức giảm của `3` nhóm chạy lẻ cộng lại (`0,0010 + 0,0006 + 0,0003 = 0,0019`), và giảm **`-8,19%`** so với `Raw (3)`.
* **Biểu đồ minh họa:** [`nb04_feature_ablation_mae.png`](../outputs/figures/nb04_feature_ablation_mae.png)

---

## 3. Độ nhạy Theo Tầm Dự báo (`15p / 30p / 60p`) & Đánh giá Phụ Đánh dấu Tắc nghẽn

| Tầm dự báo | Cấu hình | `Std-MAE` | `MAE Speed` (`mph`) | `MAE Occ` | `MAE Flow` (`xe/5p`) | `MAE Speed Điểm nghẽn` | `F1 Tắc` *(phụ)* | `Precision` *(phụ)* | `Recall` *(phụ)* |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`15 phút`** | `Raw (3)` | 0,1467 | 1,173 | 0,00655 | 16,55 | 2,050 | 0,7835 | 0,8382 | 0,7356 |
| **`15 phút`** | `Base (7)` | 0,1428 | 1,152 | 0,00637 | 15,90 | 2,035 | 0,7835 | 0,8360 | 0,7373 |
| **`15 phút`** | **`Full (14)`** | **0,1409** | **1,141** | **0,00627** | **15,62** | **2,019** | 0,7786 | **0,8473** | 0,7202 |
| **`30 phút`** | `Raw (3)` | 0,1771 | 1,494 | 0,00771 | 18,83 | 2,723 | 0,6799 | 0,7688 | 0,6095 |
| **`30 phút`** | `Base (7)` | 0,1676 | 1,428 | 0,00728 | 17,55 | 2,648 | 0,6806 | 0,7857 | 0,6004 |
| **`30 phút`** | **`Full (14)`** | **0,1639** | **1,401** | **0,00714** | **17,02** | **2,613** | 0,6785 | **0,8038** | 0,5871 |
| **`60 phút`** | `Raw (3)` | 0,2255 | 1,911 | 0,00975 | 24,01 | 3,605 | 0,5280 | 0,6763 | 0,4331 |
| **`60 phút`** | `Base (7)` | 0,2035 | 1,775 | 0,00872 | 20,81 | 3,406 | 0,5381 | 0,7248 | 0,4279 |
| **`60 phút`** | **`Full (14)`** | **0,1980** | **1,729** | **0,00849** | **20,17** | **3,334** | **0,5511** | **0,7427** | **0,4381** |

* **Lợi thế mở rộng ở tầm xa (`60 phút`):** `Full (14)` giảm `MAE Speed` `-9,52%` vs `Raw` (`-2,59%` vs `Base`), giảm `MAE Flow` `-15,99%` vs `Raw` (`-3,08%` vs `Base`), đồng thời ở đánh giá phụ đánh dấu tắc nghẽn (ngưỡng thuần `Train`) nâng đồng thời cả `F1` (`0,5511`), `Precision` (`0,7427`) và `Recall` (`0,4381`). Tính chung `Avg 5–60p`, `F1` tăng `0,6866 → 0,6911 → 0,6923` và `Precision` tăng `0,7812 → 0,8015 → 0,8166`.
* **Biểu đồ minh họa:** [`nb04_horizon_and_congestion_check.png`](../outputs/figures/nb04_horizon_and_congestion_check.png)

---

## 4. Trích xuất Đóng góp Đặc trưng (`Permutation Feature Importance`) & Chú ý Cạnh `GATv2`

### 4.1. Xếp hạng Đóng góp theo Nhóm và Từng Đặc trưng (`Test PFI`, `5` lần lặp trên `ST-GAT Full 14`)

| Nhóm đặc trưng | Số biến | `ΔStd-MAE` | `±SD` | `ΔMAE Speed` (`mph`) | `ΔMAE Occ` | `ΔMAE Flow` | Đặc trưng dẫn đầu nhóm (`ΔStd-MAE`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Raw (3 biến gốc)` *(Mốc neo)* | 3 | `+0,57183` | `0,00406` | `+2,8916` | `+0,02852` | `+92,13` | `Flow (+0,2446)`, `Occupancy (+0,2293)`, `Speed (+0,1477)` |
| **Nhóm 4 — `STID` (Chu kỳ thời gian)** | 4 | **`+0,06865`** | `0,00181` | `+0,2910` | `+0,00343` | `+12,31` | **`tod_cos (+0,0567)`**, **`tod_sin (+0,0342)`**, `dow_sin (+0,0055)` |
| **Nhóm 3 — Độ dốc có dấu** | 2 | **`+0,02478`** | `0,00068` | `+0,1200` | `+0,00111` | `+4,51` | **`occ_slope (+0,0222)`**, **`spd_slope (+0,0067)`** |
| **Nhóm 1 — Dao động khử dốc** | 3 | **`+0,00514`** | `0,00011` | `+0,0377` | `+0,00023` | `+0,65` | **`occ_rstd (+0,0042)`**, `flow_rstd (+0,0023)`, `spd_rstd (+0,0018)` |
| **Nhóm 2 — Liên kết đa biến** | 2 | **`+0,00423`** | `0,00007` | `+0,0321` | `+0,00019` | `+0,51` | **`flow_occ_decouple (+0,0032)`**, `occ_spd_corr (+0,0013)` |

* **Điểm sâu sắc về cơ chế cộng hưởng:** Toàn bộ **`14 / 14` đặc trưng** đều có `ΔStd-MAE > 0` (không có biến thừa). Khi gộp đủ `14` biến, cặp **độ dốc có dấu (`occ_slope`, `spd_slope`)** trở thành bộ tín hiệu tiền tắc nghẽn được mô hình khai thác mạnh nhất để định hướng xu thế tăng/giảm, kết hợp với mức bất ổn mật độ (`occ_rstd`) và tín hiệu đứt gãy Biểu đồ cơ bản (`flow_occ_decouple`).
* **Biểu đồ minh họa:** [`nb04_permutation_importance.png`](../outputs/figures/nb04_permutation_importance.png)

### 4.2. Phân bổ Trọng số Chú ý Động `GATv2` ($\alpha_{ij}$) theo Hướng Cạnh

| Hướng cạnh `GATv2` (TB `40` trạm điểm nghẽn trên `Test`) | Bình thường ($\sum \alpha_{ij}$) | Tiền tắc nghẽn ($\sum \alpha_{ij}$) | Dịch chuyển ($\Delta \alpha$) |
| :--- | :---: | :---: | :---: |
| **`Self` (Bản thân)** | `0,3385` | **`0,4296`** | **`+0,0911`** |
| **`In` (Thượng nguồn đổ vào)** | `0,4214` | `0,3193` | `-0,1021` |
| **`Out` (Hạ nguồn dội ngược)** | `0,2401` | **`0,2511`** | **`+0,0110`** |

* Riêng tại **điểm nghẽn số 1 `N22`**, toàn bộ **`4 / 4` cạnh hạ nguồn `Out`** đều tăng mạnh chú ý trước khi tắc (`N149 → N22: +0,0588`, `N30 → N22: +0,0496`, `N52 → N22: +0,0402`, `N4 → N22: +0,0098`), đẩy tổng chú ý chiều `Out` từ **`0,2376` (`23,8%`) → `0,3961` (`39,6%`, tăng `+66,7%` tương đối)** — nhất quán hoàn toàn với cơ chế lan truyền ngược dòng (**DCRNN, 2018**; **PDFormer, 2023**).
* **Biểu đồ minh họa:** [`nb04_gatv2_attention_and_case.png`](../outputs/figures/nb04_gatv2_attention_and_case.png)

---

## 5. Kiểm thử Tính Quy nạp & Bàn giao sang Bước 05 (`PEMS04`)

* Nạp trọng số `stgatv2_pems08_full14.pt` và hoán đổi sang đồ thị **`PEMS04` (`307` trạm, `987` cạnh = `307 Self + 340 In + 340 Out`)** cho đầu ra hợp lệ `output = (2, 307, 12, 3)`, `attention = (2, 987)` — sẵn sàng cho kiểm chứng độc lập trên `PEMS04` ở `NB05`.
* **Tiếp theo:** [`05_Cross_Map_Validation_PEMS04.ipynb`](../notebooks/05_Cross_Map_Validation_PEMS04.ipynb) · Báo cáo: [`05_Cross_Map_Validation_Report.md`](./05_Cross_Map_Validation_Report.md).
