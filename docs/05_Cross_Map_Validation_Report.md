# Báo cáo Bước 05: Kiểm chứng Đối xứng Toàn diện trên `PEMS04` và Kết luận Toàn Dự án

* **Tệp nguồn:** [`05_Cross_Map_Validation_PEMS04.ipynb`](../notebooks/05_Cross_Map_Validation_PEMS04.ipynb) · Mã nguồn `PyTorch` & biểu đồ: [`src/data.py`](../src/data.py), [`src/models.py`](../src/models.py), [`src/train.py`](../src/train.py), [`src/explain.py`](../src/explain.py), [`src/plots_nb05.py`](../src/plots_nb05.py).
* **Dữ liệu đầu vào:** `datasets/cleaned/PEMS04_clean.npz` (`16.992 × 307 × 3`, `59` ngày tại khu vực Vịnh San Francisco, `0` lỗi tồn đọng từ Bước 02; `987` cạnh có hướng 2 chiều = `307 Self + 340 In + 340 Out`), hoàn toàn độc lập với quá trình khám phá (`NB03`) và chọn đặc trưng (`NB04`) trên `PEMS08`.
* **Nguyên tắc công bằng tuyệt đối (*ceteris paribus*):** Toàn bộ **`6` cấu hình đặc trưng** (`Raw 3`, `Base 7`, `+Nhóm 1`, `+Nhóm 2`, `+Nhóm 3`, `Full 14`) đều được huấn luyện lại độc lập trên `PEMS04` bằng **cùng một kiến trúc đóng băng `ST-GAT (LSTM → 2-layer Multi-Head GATv2)`**, cùng tiền xử lý **Chen et al. (2003)**, cùng chuẩn chia **`60 / 20 / 20` (`12 → 12`)** (**LibCity, 2021**; **BasicTS, 2025**) và cùng `SEED = 42` — không so chéo khập khiễng với số liệu bài báo khác huấn luyện trên dữ liệu thô đơn biến.
* **Tệp đầu ra bàn giao:** `6` checkpoint PyTorch (`outputs/models/stgatv2_pems04_*.pt`), `5` bảng tổng hợp ([`nb05_pems04_ablation_summary.csv`](../outputs/nb05_pems04_ablation_summary.csv), [`nb05_pems04_zeroshot_summary.csv`](../outputs/nb05_pems04_zeroshot_summary.csv), [`nb05_pems04_pfi.csv`](../outputs/nb05_pems04_pfi.csv), [`nb05_cross_map_master_summary.csv`](../outputs/nb05_cross_map_master_summary.csv), [`nb05_pems04_horizon_results.csv`](../outputs/nb05_pems04_horizon_results.csv)) và `4` biểu đồ `300 DPI` tại `outputs/figures/nb05_*.png`.

---

## 1. Tự hiệu chuẩn Ngưỡng Thuần `Train (60%)` trên `PEMS04` vs. `PEMS08`

Áp nguyên công thức tự tính ngưỡng từ Biểu đồ cơ bản + quy chuẩn **FHWA (`< 60%` FFS)** duy nhất trên tập **`Train (60%)`** (`0%` rò rỉ sang `Validation` và `Test`, cách ly biên nhân quả `22` bước):

| Bản đồ (hiệu chuẩn thuần `Train 60%`) | $v_{\text{ff}}$ (`mph`) | $v_{\text{cong}} = 0{,}60\,v_{\text{ff}}$ (`mph`) | $o_{\text{crit}}$ | $o_{\text{sat}}$ (`90% Q_max`) | Tắc `Train / Val / Test` (`%`) | Trạm điểm nghẽn (`>= 1%` Train) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PEMS08`** *(San Bernardino — `170` trạm)* | 67,10 | **40,26** | 0,2875 | **0,1375** | `1,81 / 1,97 / 2,24` | `43 / 170` |
| **`PEMS04`** *(San Francisco Bay — `307` trạm)* | 67,20 | **40,32** | 0,1525 | **0,1025** | `2,84 / 4,05 / 3,12` | `155 / 307` |

* Thuật toán tự hạ ngưỡng bão hòa $o_{\text{sat}}$ từ `0,1375` (`PEMS08`) xuống **`0,1025`** (`PEMS04`) để thích nghi với mật độ nền thấp hơn của Vịnh San Francisco mà không cần chỉnh tay hằng số nào (`10.161 / 3.364 / 3.365` mẫu cửa sổ `12 → 12`).
* **Biểu đồ minh họa:** [`nb05_pems04_thresholds_vs_pems08.png`](../outputs/figures/nb05_pems04_thresholds_vs_pems08.png)

---

## 2. Đối chứng Toàn bộ `6` Cấu hình Đặc trưng trên `PEMS04` (Huấn luyện Độc lập & *Zero-Shot*)

### 2.1. Huấn luyện độc lập cả `6` cấu hình trên `PEMS04` (`Test Avg 5–60p`, `SEED = 42`)

| Cấu hình đầu vào (`PEMS04`) | Số kênh | Tham số | `Val L1` | `Test Std-MAE` | `MAE Speed` (`mph`) | `RMSE Speed` | `MAE Occ` | `MAE Flow` (`xe/5p`) | `MAE Speed Điểm nghẽn` | So với `Raw` | So với `Base` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Raw (3)`** *(ST-GAT gốc)* | 3 | 11.524 | 0,2084 | 0,1904 | 1,813 | 4,174 | 0,00910 | 24,52 | 2,386 | `+0,00%` | `+5,19%` |
| **`Base (7)`** *(`+Nhóm 4 STID`)* | 7 | 12.036 | 0,1971 | 0,1810 | 1,762 | 4,007 | 0,00863 | 22,63 | 2,308 | `-4,94%` | `+0,00%` |
| **`Base + Nhóm 1 (10)`** *(Dao động)* | 10 | 12.420 | 0,1949 | 0,1790 | 1,745 | 4,026 | 0,00849 | 22,49 | 2,297 | `-5,99%` | `-1,10%` |
| **`Base + Nhóm 2 (9)`** *(Liên kết)* | 9 | 12.292 | 0,1967 | 0,1803 | 1,763 | 4,036 | 0,00857 | 22,51 | 2,315 | `-5,30%` | `-0,39%` |
| **`Base + Nhóm 3 (9)`** *(Độ dốc)* | 9 | 12.292 | 0,1955 | 0,1795 | 1,751 | 4,016 | 0,00851 | 22,56 | 2,300 | `-5,72%` | `-0,83%` |
| **`Full (14)`** *(Gộp cả 4 nhóm)* | **14** | 12.932 | **0,1942** | **0,1785** | **1,737** | **4,000** | **0,00849** | **22,39** | **2,295** | **`-6,25%`** | **`-1,38%`** |

### 2.2. Kiểm chứng bổ trợ: Chuyển giao quy nạp *Zero-Shot* (`PEMS08 → PEMS04`, `0` epoch)

| Checkpoint từ `NB04` (`Zero-Shot`, `0` epoch) | `Std-MAE` | `MAE Speed` (`mph`) | `MAE Occ` | `MAE Flow` (`xe/5p`) | `F1 Tắc 60p` *(phụ)* | So với `Raw` | So với `Base` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Raw (3)`** | 0,1974 | 1,877 | 0,00941 | 25,54 | 0,4889 | `+0,00%` | `+0,77%` |
| **`Base (7)`** | 0,1959 | 1,912 | 0,00930 | 24,54 | 0,4982 | `-0,76%` | `+0,00%` |
| **`Base + Nhóm 1 (10)`** | 0,1947 | 1,893 | 0,00923 | 24,54 | 0,4847 | `-1,37%` | `-0,61%` |
| **`Base + Nhóm 2 (9)`** | 0,1945 | 1,903 | 0,00919 | 24,41 | 0,4858 | `-1,47%` | `-0,71%` |
| **`Base + Nhóm 3 (9)`** | **0,1932** | **1,877** | 0,00919 | **24,26** | 0,4907 | **`-2,13%`** | **`-1,38%`** |
| **`Full (14)`** | 0,1939 | 1,901 | **0,00912** | 24,36 | **0,5072** | `-1,77%` | `-1,02%` |

* **Điểm chốt:** Trên `PEMS04` huấn luyện độc lập, cả `3` nhóm tín hiệu tiền tắc nghẽn đều vượt `Base (7)` trên cả `Val L1` lẫn `Test Std-MAE` (**Nhóm 1 #1: `0,1790`**), và **`Full (14)`** dẫn đầu tuyệt đối trên mọi biến hồi quy (**`-6,25%`** vs `Raw`, **`-1,38%`** vs `Base`). Ở chế độ *Zero-Shot* (`0` epoch), vì khung giờ cao điểm hai thành phố lệch mùa nên `STID` bị *domain shift* nhẹ ở `Speed` (`1,877 → 1,912`), còn `7` tín hiệu vật lý bất biến địa lý vẫn kéo giảm `Std-MAE` đồng đều so với `Base (7)` ngay ở `0` epoch (`+Nhóm 3: -1,38%`, `Full 14: -1,02%`). Khi huấn luyện thích nghi trên `PEMS04`, `Full (14)` giảm thêm `-7,94% Std-MAE` (`0,1939 → 0,1785`).
* **Biểu đồ minh họa:** [`nb05_pems04_ablation_mae.png`](../outputs/figures/nb05_pems04_ablation_mae.png)

---

## 3. Tính Nhất quán Xuyên Thành phố: `Test PFI` & Chú ý Cạnh `GATv2`

| Nhóm đặc trưng (`Test PFI`, `5` lần lặp) | Số biến | `PEMS08 ΔStd-MAE` | `PEMS04 ΔStd-MAE` | `PEMS04 ΔMAE Speed` (`mph`) | Tính nhất quán xuyên bản đồ |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `Raw (3 biến gốc)` *(Mốc neo)* | 3 | `+0,57183` | `+0,56628` | `+3,65162` | Khớp #1 |
| **Nhóm 4 — `STID` (Chu kỳ thời gian)** | 4 | **`+0,06865`** | **`+0,05263`** | `+0,24765` | **Khớp #1 nhóm bổ sung** (`tod_cos`, `tod_sin`) |
| **Nhóm 3 — Độ dốc có dấu** | 2 | **`+0,02478`** | **`+0,02282`** | `+0,17950` | **Khớp #2 nhóm bổ sung** (`occ_slope`, `spd_slope`) |
| **Nhóm 1 — Dao động khử dốc** | 3 | **`+0,00514`** | **`+0,00585`** | `+0,06108` | **Khớp #3 nhóm bổ sung** (`occ_rstd`, `flow_rstd`, `spd_rstd`) |
| **Nhóm 2 — Liên kết đa biến** | 2 | **`+0,00423`** | **`+0,00231`** | `+0,01815` | **Khớp #4 nhóm bổ sung** (`flow_occ_decouple`, `occ_spd_corr`) |

* **Độ lặp lại cấu trúc giải thích tuyệt đối:** Toàn bộ **`14 / 14` đặc trưng** đều có `ΔStd-MAE > 0` trên cả `PEMS08` lẫn `PEMS04`, đạt tương quan hạng **Spearman `ρ = 0,9956` ($p = 1,03 \times 10^{-13}$)** (`11` biến bổ sung: **`ρ = 0,9909`**, $p = 3,76 \times 10^{-9}$).
* **Chú ý cạnh `GATv2` trên `153` trạm điểm nghẽn `PEMS04`:** Khi bước vào pha tiền tắc nghẽn, tổng chú ý lân cận tăng từ `60,9% → 65,9%` (`Self: 0,3910 → 0,3408`), trong đó chiều **`Out` (hạ nguồn dội ngược)** tăng mạnh nhất **`0,1817 → 0,2226` (`+0,0409`, `+22,5%` tương đối)** và `In` tăng `0,4272 → 0,4365` (`+0,0093`). Tại điểm nghẽn #1 **`N79`**, tổng chú ý chiều `Out` tăng vọt `23,2% → 33,6%` (`+45,2%` tương đối).
* **Biểu đồ minh họa:** [`nb05_cross_map_pfi_and_attention.png`](../outputs/figures/nb05_cross_map_pfi_and_attention.png)

---

## 4. Tổng hợp Đối chiếu Hai Thành phố (`PEMS08` vs. `PEMS04`) & Tầm Dự báo

### 4.1. Bảng tổng hợp đối xứng `6` cấu hình trên cùng kiến trúc `ST-GAT (LSTM → 2-layer GATv2)`

| Cấu hình đầu vào | Số kênh | `PEMS08 Std-MAE` | `PEMS08 Speed` | `PEMS08 Flow` | `PEMS08` vs `Raw` / `Base` | `PEMS04 Std-MAE` | `PEMS04 Speed` | `PEMS04 Flow` | `PEMS04` vs `Raw` / `Base` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Raw (3)`** | 3 | 0,1771 | 1,469 | 19,21 | `+0,00% / +6,56%` | 0,1904 | 1,813 | 24,52 | `+0,00% / +5,19%` |
| **`Base (7)`** | 7 | 0,1662 | 1,397 | 17,66 | `-6,15% / +0,00%` | 0,1810 | 1,762 | 22,63 | `-4,94% / +0,00%` |
| **`Base + Nhóm 1 (10)`** | 10 | 0,1652 | 1,393 | 17,51 | `-6,72% / -0,60%` | 0,1790 | 1,745 | 22,49 | `-5,99% / -1,10%` |
| **`Base + Nhóm 2 (9)`** | 9 | 0,1656 | 1,400 | 17,48 | `-6,49% / -0,36%` | 0,1803 | 1,763 | 22,51 | `-5,30% / -0,39%` |
| **`Base + Nhóm 3 (9)`** | 9 | 0,1659 | 1,395 | 17,63 | `-6,32% / -0,18%` | 0,1795 | 1,751 | 22,56 | `-5,72% / -0,83%` |
| **`Full (14)`** | **14** | **0,1626** | **1,371** | **17,20** | **`-8,19% / -2,17%`** | **0,1785** | **1,737** | **22,39** | **`-6,25% / -1,38%`** |

### 4.2. Chi tiết `PEMS04` tại `15p / 30p / 60p` & Đánh giá phụ đánh dấu tắc nghẽn

| Tầm dự báo | Cấu hình (`PEMS04`) | `Std-MAE` | `MAE Speed` (`mph`) | `MAE Occ` | `MAE Flow` (`xe/5p`) | `MAE Speed Điểm nghẽn` | `F1 Tắc` *(phụ)* | `Precision` *(phụ)* | `Recall` *(phụ)* |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`15 phút`** | `Raw (3)` / `Base (7)` / **`Full (14)`** | `0,1558 / 0,1531 /` **`0,1516`** | `1,422 / 1,416 /` **`1,401`** | `0,00756 / 0,00743 /` **`0,00735`** | `20,89 / 20,17 /` **`20,01`** | `1,805 / 1,797 /` **`1,786`** | `0,8139 / 0,8156 / 0,8122` | `0,8518 / 0,8483 /` **`0,8640`** | `0,7792 / 0,7853 / 0,7662` |
| **`30 phút`** | `Raw (3)` / `Base (7)` / **`Full (14)`** | `0,1896 / 0,1821 /` **`0,1794`** | `1,837 / 1,799 /` **`1,771`** | `0,00901 / 0,00864 /` **`0,00851`** | `23,98 / 22,41 /` **`22,12`** | `2,413 / 2,359 /` **`2,341`** | `0,7036 / 0,7118 / 0,7027` | `0,7730 / 0,7851 /` **`0,8140`** | `0,6456 / 0,6511 / 0,6181` |
| **`60 phút`** | `Raw (3)` / `Base (7)` / **`Full (14)`** | `0,2444 / 0,2231 /` **`0,2193`** | `2,387 / 2,261 /` **`2,225`** | `0,01153 / 0,01047 /` **`0,01022`** | `30,81 / 26,75 /` **`26,43`** | `3,236 / 3,034 /` **`3,019`** | `0,5290 / 0,5668 / 0,5434` | `0,6696 / 0,7274 /` **`0,7588`** | `0,4371 / 0,4643 / 0,4232` |

* Trung bình `5–60p` trên `PEMS04`: **`Full (14)`** dẫn đầu toàn bộ chỉ số hồi quy (`MAE Speed`, `MAE Occ`, `MAE Flow`, `MAE Speed Điểm nghẽn`) ở mọi tầm dự báo (mạnh nhất ở **`60 phút`**: `MAE Speed -6,79%`, `MAE Flow -14,22%` vs `Raw`), đồng thời đạt **`Precision` đánh dấu tắc cao nhất trên mọi mốc (`Avg 0,8308`, `+4,46%pt` vs `Raw`; `60p: 0,7588`, `+8,92%pt` vs `Raw`)**. Do tối ưu thuần `L1` đều toàn mạng, `Recall` thận trọng hơn nhẹ (`0,6180` vs `Base 0,6492`), nên `F1` của `Full (14)` (`0,7088`) vượt `Raw (3)` (`0,7065`) nhưng thấp hơn nhẹ so với `Base (7)` (`0,7192`).
* **Biểu đồ minh họa:** [`nb05_final_cross_map_synthesis.png`](../outputs/figures/nb05_final_cross_map_synthesis.png)

---

## 5. Kết luận Toàn Dự án (`RQ1 + RQ2`) và Giới hạn Nghiên cứu

1. **Kết luận `RQ1`:** Công thức tự tính ngưỡng từ Biểu đồ cơ bản + **FHWA (`< 60%` FFS)** thích nghi tự động thuần `Train` giữa `PEMS08` ($o_{\text{sat}} = 0{,}1375$) và `PEMS04` ($o_{\text{sat}} = 0{,}1025$); bộ **`14` đặc trưng cục bộ nhân quả (`H = 12` = `60p`)** kết hợp đồ thị động 2 chiều `In + Out` khắc phục triệt để giới hạn của ngưỡng `Flow` (*Capacity Drop*) và `Betweenness` tĩnh (`p = 0,3966`).
2. **Kết luận `RQ2`:** Trên cùng một kiến trúc quy nạp gọn nhẹ **`ST-GAT (LSTM → 2-layer GATv2)`** (`11.524 – 12.932` tham số), bổ sung đủ `4` nhóm đặc trưng (`Full 14`) tái lập nhất quán mức giảm `Test Std-MAE` trên **cả hai bản đồ độc lập**: **`-8,19%`** so với `Raw (3)` (`-2,17%` vs `Base`) trên **`PEMS08`** và **`-6,25%`** so với `Raw (3)` (`-1,38%` vs `Base`) trên **`PEMS04`**. Xếp hạng đóng góp `14` đặc trưng đạt tương quan **Spearman `ρ = 0,9956` ($p = 1,03 \times 10^{-13}$)** xuyên hai thành phố (`14/14` dương), `7` tín hiệu vật lý bền vững hơn `STID` trong chuyển giao *Zero-Shot*, và chú ý `GATv2` tập trung rõ vào chiều hạ nguồn dội ngược (`Out`).
3. **Giới hạn cần nêu rõ trong bài báo IEEE:** *(i)* Hàm mất mát hồi quy `L1` đều toàn mạng giúp `Full (14)` dẫn đầu toàn bộ `MAE` và `Precision` (`0,82–0,83`), nhưng đổi lại `Recall` đuôi tắc xa `60p` còn khiêm tốn (`~0,42–0,44`); *(ii)* đặc trưng chu kỳ `STID` nhạy cảm với dịch chuyển mùa/khung giờ khi chuyển giao *Zero-Shot* và cần tinh chỉnh nhẹ trên thành phố mới; *(iii)* chu kỳ `~2` tháng của `PEMS04/08` chưa bao phủ biến động mùa cả năm hay tai nạn phi chu kỳ.
