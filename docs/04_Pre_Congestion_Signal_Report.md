# Báo cáo Bước 04: Kiểm chứng Dấu hiệu Tiền tắc nghẽn & Chốt Tham số Mô hình

* **Tệp nguồn:** [`04_Pre_Congestion_Signal_Analysis.ipynb`](../notebooks/04_Pre_Congestion_Signal_Analysis.ipynb)
* **Dữ liệu đầu vào:** `datasets/cleaned/PEMS08_clean.npz` (`17.856 × 170 × 3`, `277` cạnh có hướng = `3` cặp 2 chiều + `271` cạnh 1 chiều), [`PEMS08_bottlenecks.csv`](../outputs/PEMS08_bottlenecks.csv) (Top 20 điểm nghẽn) và tiêu chí tắc nghẽn thực `(Speed < 40 mph & Occ ≥ 0,15)` từ [Báo cáo Bước 03](./03_EDA_and_Bottleneck_Report.md).
* **Nguyên tắc thiết kế tuần tự:** Mục 2–4 kiểm chứng các dấu hiệu trên mốc mặc định là bước kế tiếp tự nhiên **`Δt = 5 phút` (`t + 1`)** để không giả định trước tầm dự báo; chỉ tới **Mục 5** mới gộp toàn bộ đặc trưng vào mô hình nhẹ để quét và chốt tầm dự báo tối ưu **`Δt*`**.
* **Tệp đầu ra bàn giao:** `5` biểu đồ `300 DPI` tại `outputs/figures/nb04_*.png` cùng bộ `4` tham số định lượng chốt cho Bước 05.

---

## 1. Trực quan Quỹ đạo Pha Tiền tắc nghẽn (`n = 3.136` sự cố khởi phát)

Tổng hợp toàn bộ **`n = 3.136`** sự cố chuyển trạng thái từ thông sang tắc thực (`cg[t] & ~cg[t - 1]`) trên Top 20 điểm nghẽn `PEMS08` trong cửa sổ `t - 60 phút` đến `t + 30 phút`:

| Mốc tương đối | `Flow` TB (`xe / 5p`) | `Occupancy` TB | `Speed` TB (`mph`) | Pha vật lý |
| :---: | :---: | :---: | :---: | :--- |
| `t = -60p` | 428,3 | 0,176 | 49,2 | Bắt đầu tích tụ mật độ |
| `t = -30p` | 430,9 | 0,183 | 47,3 | Đông dần, tốc độ giảm nhẹ |
| `t = -20p` | 432,5 | 0,185 | 46,5 | Bắt đầu vùng chuyển pha |
| `t = -15p` | 434,5 | 0,187 | 45,9 | Tiệm cận đỉnh công suất |
| **`t = -10p`** | **435,2** *(đỉnh)* | 0,192 | 44,7 | **Đạt đỉnh công suất dòng xe** |
| `t = -5p` | 431,4 | **0,204** | **42,5** | `Occ` tăng dựng đứng, `Flow` bắt đầu hụt |
| **`t = 0p`** | **407,5** | **0,253** | **36,9** | **Sụp pha vào tắc thực (`< 40 mph`)** |

* **Biểu đồ minh họa:** [`nb04_phase_trajectory.png`](../outputs/figures/nb04_phase_trajectory.png)

---

## 2. Xác định Độ dài Cửa sổ Quá khứ Đủ dài (`H*`) bằng `PACF` & `ROC-AUC` (Mốc mặc định `t + 5p`)

Áp dụng tự tương quan riêng phần `PACF` (chuẩn Box–Jenkins, ngưỡng tin cậy `95%` = $1,96 / \sqrt{17.856} = \mathbf{0,0147}$) kết hợp `ROC-AUC(t - k)` liên tục `5 – 120 phút` (dự báo bước kế tiếp tự nhiên `t + 5p` khi đường tại `t` chưa tắc):

| $k$ (`bước`) | Khoảng (`phút`) | `\|PACF\| Vận tốc` | `\|PACF\| Occ` | Vượt ngưỡng `95%` (`0,0147`) | `AUC Occ(t - k)` | `AUC Speed(t - k)` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | `5` | 0,9770 | 0,9563 | Có | 0,9334 | 0,9438 |
| `2` | `10` | 0,1685 | 0,2237 | Có | 0,9139 | 0,9110 |
| `3` | `15` | 0,0413 | 0,0876 | Có | 0,9034 | 0,8918 |
| `4` | `20` | 0,0220 | 0,0463 | Có | 0,8959 | 0,8778 |
| `6` | `30` | 0,0201 | 0,0232 | Có | 0,8833 | 0,8579 |
| `7` | `35` | 0,0158 | 0,0257 | Có | 0,8774 | 0,8477 |
| **`8` *(Chốt $H^*$)*** | **`40`** | **0,0160** | **0,0236** | **Có *(cuối chuỗi)*** | **0,8724** | **0,8392** |
| **`9`** | **`45`** | **0,0142** | **0,0117** | **Không *(cùng cắt dưới)*** | 0,8670 | 0,8299 |
| `10` | `50` | 0,0135 | 0,0140 | Không | 0,8620 | 0,8208 |
| `12` | `60` | 0,0153 | 0,0149 | Có *(nhiễu biên)* | 0,8549 | 0,8112 |
| `18` | `90` | 0,0107 | 0,0128 | Không | 0,8189 | 0,7647 |
| `24` | `120` | 0,0130 | 0,0153 | Không | 0,7891 | 0,7249 |

> **Nhận xét (Chốt cửa sổ quá khứ):**
>
> * Cả `|PACF| Vận tốc` lẫn `|PACF| Độ chiếm dụng` đều duy trì vượt ngưỡng `95%` (`0,0147`) **liên tiếp từ `k = 1` đến `k = 8`** rồi cùng cắt xuống dưới ngưỡng ngay tại `k = 9` (`0,0142` và `0,0117`) → **chốt thuần thống kê `H* = 8` bước (`40 phút`)**.

* **Biểu đồ minh họa:** [`nb04_pacf_window_H.png`](../outputs/figures/nb04_pacf_window_H.png)

---

## 3. Kiểm chứng Độ biến động của `Flow`, `Occupancy`, `Speed` trong Cửa sổ `H* = 8` (`40 phút`)

Xét khi đoạn đường đang thông thoáng tại $t$ để nhận diện ca sắp tắc thực tại bước kế tiếp $t + 5$ phút:

| Biến đo lường *(trong `H* = 40p`)* | `Δ40p` Trung vị *(Sắp tắc / Thông)* | `AUC Δ40p` *(gốc / khử ngày)* | `Std40p` Trung vị *(Sắp tắc / Thông)* | `AUC Std40p` | `Risk Ratio Std Top 10%` | `p` *(Mann–Whitney)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Lưu lượng (Flow)`** | `+5,0000 / -2,0000` | `0,5315 / 0,5156` | `23,3024 / 18,2140` (`1,28x`) | 0,6323 | `2,06x` | `< 1e-300` |
| **`Độ chiếm dụng (Occ)`** | **`+0,0255 / -0,0006`** | **`0,6480 / 0,6165`** | **`0,0308 / 0,0058` (`5,32x`)** | **0,9321** | **`35,96x`** | `< 1e-300` |
| **`Vận tốc (Speed)`** | **`-4,9000 / +0,0000`** | **`0,6557 / 0,6316`** | **`4,2637 / 1,1252` (`3,79x`)** | **0,8715** | **`12,28x`** | `< 1e-300` |

> **Nhận xét (Tín hiệu dao động bất ổn trước khi tắc):**
>
> * Biên độ dao động **`Std40p`** trong đúng cửa sổ `H* = 8` tăng vọt gấp **`5,32 lần`** với `Occupancy` (`AUC = 0,9321`, `RR = 35,96x`) và **`3,79 lần`** với `Speed` (`AUC = 0,8715`, `RR = 12,28x`), chứng minh dòng xe xuất hiện bất ổn rõ rệt (`stop-and-go`) trước khi gãy pha. `Flow` yếu hơn (`1,28x`, `AUC = 0,6323`) do tính đa trị gần đỉnh công suất nhưng vẫn có ý nghĩa bổ trợ (`p < 1e-300`).

* **Biểu đồ minh họa:** [`nb04_volatility_before_congestion.png`](../outputs/figures/nb04_volatility_before_congestion.png)

---

## 4. Giá trị Cộng thêm của Không gian Đa nhánh (`Self only` vs `Self + Chiều vào / Chiều ra / 2 chiều`)

Trên **`18 / 20`** điểm nghẽn có đủ cả nhánh đi vào và nhánh đi ra (`bậc vào = 1–5`, TB `2,72`; `bậc ra = 1–5`, TB `2,44`; `N9` và `N47` có `in_degree = 0`), lấy **`Self only`** (đã có đủ lịch sử `H* = 8` và `Δ40p / Std40p`) làm nền rồi cộng thêm trực tiếp không gian lân cận bằng **Hồi quy Logistic chuẩn dùng chung** (`70% Train / 15% Val`, `n = 42.952` mẫu chưa tắc tại $t$, mốc mặc định $t + 5$ phút):

| Cấu trúc không gian ($t \to t + 5p$) | `Val PR-AUC` | Tăng `PR-AUC` vs `Self` | `Val Log-Loss` | `Val ROC-AUC` | `p` *(LRT vs `Self only`)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Self only (Nền H* = 8)`** | 0,3094 | `+0,00%` | 0,02923 | 0,9805 | — *(Nền đối chiếu)* |
| `Self + Chiều vào (Mean)` | 0,3108 | `+0,46%` | 0,02927 | 0,9804 | `1,0000` |
| `Self + Chiều vào (Max)` | 0,3269 | `+5,66%` | 0,02857 | 0,9810 | `2,29e-12` |
| `Self + Chiều vào (Mean + Max)` | 0,3308 | `+6,92%` | 0,02839 | 0,9809 | `1,64e-13` |
| `Self + Chiều ra (Mean)` | 0,3260 | `+5,38%` | 0,02848 | 0,9803 | `6,69e-14` |
| `Self + Chiều ra (Max)` | 0,3230 | `+4,42%` | 0,02852 | 0,9805 | `4,07e-13` |
| `Self + Chiều ra (Mean + Max)` | 0,3281 | `+6,04%` | 0,02846 | 0,9806 | `2,37e-12` |
| `Self + 2 chiều (Mean)` | 0,3201 | `+3,46%` | 0,02881 | 0,9800 | `2,14e-06` |
| `Self + 2 chiều (Max)` | 0,3288 | `+6,27%` | 0,02842 | 0,9808 | `4,24e-13` |
| **`Self + 2 chiều (Mean + Max)` *(Chốt)*** | **0,3383** | **`+9,36%`** | **0,02816** | **0,9808** | **`2,26e-14`** |

* **Kiểm định Likelihood-Ratio ($\chi^2$ trên `Val 15%`):**
  * `Self + 2 chiều (Mean + Max)` > **`Self only`**: $\chi^2 = \mathbf{91,8}$, **`p = 2,26e-14`** (giảm `Log-Loss` tại `13/18` điểm nghẽn).
  * `Self + 2 chiều (Mean + Max)` > **`Self + Chiều vào (Mean + Max)`**: $\chi^2 = \mathbf{19,8}$, **`p = 0,0030`** (giảm `Log-Loss` tại `12/18` điểm nghẽn).
  * `Self + 2 chiều (Mean + Max)` > **`Self + Chiều ra (Mean + Max)`**: $\chi^2 = \mathbf{25,5}$, **`p = 0,0003`** (giảm `Log-Loss` tại `9/18` điểm nghẽn).
* **Biểu đồ minh họa:** [`nb04_spatial_multi_edge.png`](../outputs/figures/nb04_spatial_multi_edge.png)

---

## 5. Gộp Đặc trưng vào Mô hình Nhẹ & Chốt Tầm Dự báo Tối ưu (`Δt*`)

Gộp toàn bộ `25` đặc trưng đã chứng minh (`Self` quá khứ `H* = 8`, biến động `Δ40p / Std40p`, nhánh vào/ra `Mean & Max` và chu kỳ giờ/thứ `sin/cos`) vào mô hình `HistGradientBoostingRegressor` dùng chung cho `18` điểm nghẽn (`70% Train / 15% Val` theo thời gian), đánh giá bằng khoảng cách hình học tới Điểm Lý tưởng Pareto (`Δt` max & `MAE` min):

$$d(\Delta t) = \sqrt{\left(1 - \widetilde{\Delta t}\right)^2 + \widetilde{\text{MAE}}(\Delta t)^2}$$

| $\Delta t$ (`phút`) | `Val MAE Speed` (`mph`) | `Val R² Speed` | `Val MAE Occ` | `Val F1 suy ra` | Khoảng cách Pareto $d(\Delta t)$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `5` | 1,4680 | 0,9651 | 0,0110 | 0,9200 | 1,0000 |
| `10` | 2,0375 | 0,9249 | 0,0128 | 0,8743 | 0,8572 |
| **`15` *(Tối ưu $\Delta t^*$)*** | **2,4654** | **0,8825** | **0,0143** | **0,8362** | **0,8067** *(Cực tiểu)* |
| `20` | 2,8022 | 0,8429 | 0,0155 | 0,8050 | 0,8249 |
| `25` | 3,0734 | 0,8066 | 0,0165 | 0,7847 | 0,8908 |
| `30` | 3,3175 | 0,7732 | 0,0173 | 0,7583 | 1,0000 |

* **Biểu đồ minh họa:** [`nb04_optimal_leadtime_pareto.png`](../outputs/figures/nb04_optimal_leadtime_pareto.png)

---

## 6. Bộ Tham số Định lượng Bàn giao sang Bước 05

| Hạng mục | Giá trị chốt | Cơ sở bằng chứng |
| :--- | :--- | :--- |
| **Tiêu chí nhãn tắc thực** | `(Speed < 40 mph) & (Occ ≥ 0,15)` | Sơ đồ cơ bản 3 biến (Bước 03) |
| **Cửa sổ quá khứ đầu vào (`H*`)** | **`8` bước (`40 phút`)** | `PACF` 95% CI `= 0,0147` cắt tại `k = 9` (Mục 2) |
| **Cấu trúc đồ thị đầu vào** | **`Self` + Đồ thị 2 chiều** (cả nhánh vào & ra) | `Self + 2 chiều` tăng `+9,36%` `PR-AUC`, `p = 2,26e-14` (Mục 4) |
| **Tầm dự báo tối ưu (`Δt*`)** | **Mốc chuẩn `15 phút`** (vùng `10–20p`; dải quét `{5, 10, 15, 20, 30}p`) | Cực tiểu Pareto `d = 0,8067`, khớp vùng chuyển pha (Mục 1 & 5) |

**Tiếp theo:** Triển khai thiết kế, so sánh khách quan các mô hình dự báo và giải thích mô hình tại [`05_Modeling_and_Explainability.ipynb`](../notebooks/05_Modeling_and_Explainability.ipynb).
