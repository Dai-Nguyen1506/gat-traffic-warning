# Báo cáo Bước 03: Khám phá Dữ liệu Sạch (EDA), Điểm nghẽn & Tín hiệu Tiền tắc nghẽn

* **Tệp nguồn:** [`03_EDA_and_Bottleneck_Analysis.ipynb`](../notebooks/03_EDA_and_Bottleneck_Analysis.ipynb)
* **Dữ liệu đầu vào:** `datasets/cleaned/PEMS08_clean.npz` (`17.856 × 170 × 3`, `277` cạnh có hướng duy nhất sau khi gộp `18` cạnh trùng từ `295` dòng CSV gốc) và `datasets/cleaned/PEMS04_clean.npz` (`16.992 × 307 × 3`, `340` cạnh).
* **Tệp đầu ra bàn giao:** [`PEMS08_bottlenecks.csv`](../outputs/PEMS08_bottlenecks.csv), [`PEMS04_bottlenecks.csv`](../outputs/PEMS04_bottlenecks.csv), [`PEMS08_network.gexf`](../outputs/PEMS08_network.gexf), [`PEMS04_network.gexf`](../outputs/PEMS04_network.gexf) cùng `8` biểu đồ `300 DPI` tại `outputs/figures/nb03_*.png`.

---

## 1. Nhịp điệu Hệ thống (Ngày – Tuần) & Tương quan Mô tả 3 Biến

1. **Chuỗi thời gian 30 ngày (`8.640` bước đo đầu tiên trên `PEMS08`):** Hệ thống lặp đều chu kỳ ngày–đêm và suy giảm lưu lượng, độ chiếm dụng rõ rệt vào hai ngày cuối tuần (vùng nền xám đánh dấu Thứ Bảy – Chủ Nhật theo lịch `PEMS08` bắt đầu từ Thứ Sáu `01/07/2016`). Dù **trung vị toàn mạng** ổn định quanh `60–67 mph`, đường **phân vị 5%** (nhóm `5%` trạm chậm nhất) liên tục lao dốc xuống `25–40 mph` đúng nhịp tăng của `Độ chiếm dụng` — cho thấy tắc nghẽn chỉ xảy ra cục bộ ở một nhóm trạm nhỏ chứ không phủ đều toàn mạng.

2. **Tổng hợp theo Thứ – Giờ & Tương quan 3 biến:** Các ngày làm việc (Thứ Hai – Thứ Sáu) xuất hiện cấu trúc **hai đỉnh cao điểm** đặc trưng tại **`07:00–09:30`** và **`16:00–19:00`** (vận tốc toàn mạng giảm xuống `~58–61 mph`), còn cuối tuần mất đỉnh sáng. Trên toàn chuỗi, `Độ chiếm dụng` tương quan nghịch với `Vận tốc` (**`r = -0.705`**), mạnh hơn gấp đôi so với `Lưu lượng – Vận tốc` (`r = -0.282`).

* **Biểu đồ minh họa:** [`nb03_clean_data_overview.png`](../outputs/figures/nb03_clean_data_overview.png) · [`nb03_one_month_timeseries.png`](../outputs/figures/nb03_one_month_timeseries.png)

---

## 2. Xét đồng thời 3 Biến để Chốt Tiêu chí Tắc nghẽn & Đối chiếu Ngày – Đêm

### 2.1. Sơ đồ cơ bản (`Flow – Occupancy – Speed`) & Chốt tiêu chí tắc nghẽn thực sự

Để định nghĩa tắc nghẽn không bị nhầm lẫn, ta xét đồng thời cả 3 biến trên `3.035.520` điểm đo của `PEMS08`:

| Khoảng `Độ chiếm dụng` | Tỷ lệ quan sát | Lưu lượng TB | Vận tốc TB (`mph`) | Tỷ lệ `< 40 mph` | Trạng thái vật lý |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `[0.00, 0.05)` | 39,47% | 105,0 | 66,6 | 0,1% | Thông thoáng mật độ thấp |
| `[0.05, 0.10)` | 47,23% | 289,1 | 64,1 | 0,1% | Thông thoáng mật độ vừa |
| `[0.10, 0.15)` | 9,50% | 386,1 | 59,3 | 1,6% | Đông dần, chưa gãy pha |
| `[0.15, 0.20)` | 1,86% | **433,4** | 50,1 | 19,4% | Chạm đỉnh công suất & chớm bão hòa |
| `[0.20, 0.30)` | 1,48% | 427,5 | **35,7** | 70,7% | Vượt ngưỡng bão hòa, dòng xe sụp pha |
| `[0.30, 1.00)` | 0,46% | 375,2 | **24,9** | 94,8% | Tắc cứng quá tải nặng |

Đối chiếu riêng 3 nhóm dưới ngưỡng `< 40 mph` để xác định vai trò từng biến:

| Nhóm trạng thái | Số điểm | Tỷ lệ toàn tập | `Flow` (`P5 / P25 / TV`) | `Occ` Trung vị | `Speed` Trung vị | Kết luận |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Tắc thực (Speed < 40 & Occ ≥ 0.15)`** | **55.848** | **1,84%** | **133 / 300 / 402** | **0,256** | **31,6 mph** | **Tiêu chí chuẩn được chốt** |
| `Chạy chậm vắng xe (Speed < 40 & Occ < 0.15)` | 6.765 | 0,22% | 30 / 77 / 203 | 0,119 | 35,2 mph | Nhiễu đường vắng, bị loại |
| `Tắc cứng nặng (Speed < 40 & Occ ≥ 0.30)` | 13.172 | 0,43% | 111 / 239 / 366 | 0,334 | 24,0 mph | `Flow` tụt ngược khi quá tải |

> **Nhận xét (Vai trò của cả 3 biến & Định nghĩa thống nhất):**
>
> * **`Vận tốc < 40 mph` kết hợp `Độ chiếm dụng ≥ 0.15`:** Loại bỏ được `6.765` điểm chạy chậm khi mặt đường thưa xe (`Flow` tại `P5–P25` chỉ `30–77` xe/5 phút), giữ lại đúng **`55.848` điểm tắc nghẽn thực sự (`1,84%`)**.
>
> * **Vì sao không cắt ngưỡng cứng theo `Lưu lượng`:** `Lưu lượng` có tính **phi đơn điệu (đa trị)**: tăng từ `105,0` lên đỉnh công suất `433,4` ở `[0.15, 0.20)` nhưng khi rơi vào tắc cứng nặng (`Occ ≥ 0.30`), dòng xe đứng bánh khiến lưu lượng tụt ngược xuống `P5 = 111`, `P25 = 239`, `TV = 366`. Do đó, `Lưu lượng` được dùng để kiểm chứng vật lý và làm biến đầu vào cho mô hình dự báo, không đặt ngưỡng cắt cứng.
>
> * **Tiêu chí chuẩn chốt thống nhất cho phần còn lại:** **`Tắc nghẽn thực sự ⇔ (Vận tốc < 40 mph) và (Độ chiếm dụng ≥ 0.15)`**.

* **Biểu đồ minh họa:** [`nb03_fundamental_diagram.png`](../outputs/figures/nb03_fundamental_diagram.png)

### 2.2. Áp dụng tiêu chí 3 biến lên Ban ngày và Ban đêm (`PEMS08`)

| Khung giờ | Số điểm tắc thực | Tỷ lệ trong khung giờ | Tỷ trọng trên toàn mạng | `Flow TV` (`Tắc / Thông`) | `Occ TV` (`Tắc / Thông`) | `Speed TV` (`Tắc / Thông`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ban ngày (`06–21h`)** | **54.330** | **2,68%** | **97,28%** | 405,0 / 275,0 | 0,257 / 0,074 | 31,7 / 64,2 |
| **Tối muộn (`22–23h`)** | 778 | 0,31% | 1,39% | 115,0 / 150,0 | 0,213 / 0,039 | 14,6 / 67,6 |
| **Đêm khuya (`00–04h`)** | **243** | **0,04%** | **0,44%** | 104,0 / 82,0 | **0,244** / 0,022 | **13,1** / 67,1 |
| **Sáng sớm (`05h`)** | 497 | 0,39% | 0,89% | 153,0 / 165,0 | 0,244 / 0,047 | 33,7 / 67,5 |

> **Nhận xét (Vì sao vẫn giữ đủ chuỗi 24h liên tục):**
>
> * Ban ngày (`06–21h`) chiếm `97,28%` thời lượng tắc thực; gộp khung `05–23h` đạt **`99,56%`**.
>
> * Vào đêm khuya (`00–04h`) vẫn tồn tại **`243` ca tắc cục bộ thật sự** (`Occ TV = 0,244`, `Speed TV = 13,1 mph`, `Flow TV = 104,0` cao hơn cả lúc thông thoáng ban đêm `82,0`), đồng thời không còn nhiễu đường vắng. Nhờ tiêu chí 3 biến đã làm sạch nhãn tự nhiên, Bước 04 có thể giữ nguyên chuỗi thời gian liên tục 24h (không cần cắt bỏ đêm làm đứt gãy chuỗi) và kiểm chứng bổ sung trên riêng khung `05–23h`.

---

## 3. Tập trung Điểm nghẽn (Pareto Top 20) & Giải phẫu Đồ thị Tĩnh

### 3.1. Phân hóa theo nguyên lý Pareto
* **`PEMS08` (`N = 170`):** Tỷ lệ tắc thực trung bình toàn mạng là **`1,84%`**, có tới **`125 / 170` trạm (`73,5%`)** hầu như không bao giờ tắc (`< 1%` thời gian). Ngược lại, **Top `20` trạm (`11,8%` mạng lưới)** có tần suất tắc từ **`4,54%` đến `30,6%`** (cao gấp **`2,5 – 16,6 lần`** mức trung bình toàn mạng `1,84%`) và **gánh tới `71,2%` tổng thời lượng tắc nghẽn thực của cả mạng lưới**.
* **`PEMS04` (`N = 307`):** Tỷ lệ tắc thực chung là **`2,74%`** (`158 / 307` trạm `< 1%`); tắc nghẽn phân tán rộng hơn nên Top 20 trạm (`9,76%–29,0%`) gánh **`33,9%`**.
* **Độ vững theo thời gian trong ngày:** Khi giới hạn riêng khung giờ `05–23h`, danh sách Top 20 trạm trùng tuyệt đối **`20 / 20` (`100%`)** trên cả `PEMS08` lẫn `PEMS04`.
* **Biểu đồ minh họa:** [`nb03_bottleneck_concentration.png`](../outputs/figures/nb03_bottleneck_concentration.png)

### 3.2. Kiểm định Spearman giữa 3 chỉ số đồ thị tĩnh và Tần suất tắc thực

| Chỉ số cấu trúc tĩnh | `PEMS08` (`N = 170`) | `PEMS04` (`N = 307`) | Kết luận |
| :--- | :---: | :---: | :--- |
| **`Degree` (Bậc kết nối)** | $\rho = 0,185, \; p = 0,0156$ | $\rho = 0,127, \; p = 0,0266$ | Tương quan thuận yếu, có ý nghĩa thống kê ($p < 0,05$) |
| **`Weighted Betweenness`** | $\rho = 0,067, \; p = 0,3842$ | $\rho = 0,051, \; p = 0,3704$ | Không có ý nghĩa thống kê ở cả 2 mạng |
| **`Weighted Closeness`** | $\rho = 0,232, \; p = 0,0024$ | $\rho = 0,035, \; p = 0,5458$ | Có ý nghĩa trên `PEMS08`, mất ý nghĩa trên `PEMS04` |

> **Nhận xét:** Hình học đồ thị tĩnh đơn lẻ không đủ dự báo nhất quán vị trí điểm nghẽn trên mọi mạng lưới (ví dụ `N9` tắc `18,0%` và `N65` tắc `15,6%` dù có `Betweenness` rất thấp) → cần kết hợp động lực chuỗi thời gian và tín hiệu từ nhánh lân cận.

* **Biểu đồ minh họa:** [`nb03_network_map_pems08.png`](../outputs/figures/nb03_network_map_pems08.png)

---

## 4. Tín hiệu Tiền tắc nghẽn, Lan truyền Đầu nguồn & Chốt Mốc Dự báo

### 4.1. Quỹ đạo pha tại điểm nghẽn (`n = 2.034` sự cố khởi phát trên Top 20 `PEMS08`)
* **`t = -60 phút`:** `Flow = 424,8`, `Occ = 0,145`, `Speed = 55,4 mph`.
* **`t = -15 phút`:** `Flow` đạt đỉnh **`433,9 xe/5 phút`**, `Occ = 0,172`, `Speed = 49,7 mph`.
* **`t = 0 phút` (Khởi phát tắc thực):** `Flow` sụp xuống **`399,7 xe/5 phút`**, `Occ` vọt lên **`0,254`** và `Speed` tụt xuống **`36,9 mph`**. Trong cửa sổ **`t - 20` đến `t - 5` phút**, độ chiếm dụng tăng dựng đứng trước khi dòng xe gãy pha.
* **Biểu đồ minh họa:** [`nb03_phase_trajectory.png`](../outputs/figures/nb03_phase_trajectory.png)

### 4.2. Tương quan trễ từ trạm đầu nguồn nối trực tiếp (`0 – 60` phút)
* Sau khi khử nhịp trung bình theo khung giờ trong ngày (`t % 288`), nhóm **trạm đầu nguồn nối trực tiếp** duy trì độ lớn tương quan nghịch cao hơn nhóm **trạm ngẫu nhiên không kết nối** (`r = -0.303` đến `-0.315`, đạt đỉnh tại **`30 phút`**, so với `-0.245` đến `-0.261`), nhưng kiểm định `t-test` ghép cặp tuyến tính trên toàn chuỗi cho `p = 0,071–0,174` (`> 0,05`) do nhiễu chuyển pha phi tuyến.
* **Biểu đồ minh họa:** [`nb03_upstream_lag_correlation.png`](../outputs/figures/nb03_upstream_lag_correlation.png)

### 4.3. Đặc trưng biến thiên tại thời điểm chưa tắc nghẽn & Kiểm định `Risk Ratio`

| Đặc trưng tại thời điểm chưa tắc $t$ | `5 phút` | `10 phút` | `15 phút` | `20 phút` | Xu hướng |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Vận tốc(t)`** | -0,2946 | -0,3198 | **-0,3228** | -0,3192 | Đạt đỉnh tại `15 phút` |
| **`Độ chiếm dụng(t)`** | +0,2534 | +0,2841 | +0,2956 | **+0,2989** | Tín hiệu nền mạnh |
| **`Độ chiếm dụng đầu nguồn(t)`** | +0,1107 | +0,1338 | +0,1505 | **+0,1617** | **Tăng đơn điệu theo tầm dự báo** |
| **`Δ Vận tốc (15p)`** | -0,0983 | **-0,1083** | -0,1013 | -0,0916 | Đạt đỉnh tại `10 phút` |
| **`Δ Độ chiếm dụng (15p)`** | +0,0799 | **+0,0889** | +0,0813 | +0,0721 | Đạt đỉnh tại `10 phút` |
| **`Δ Lưu lượng (15p)`** | -0,0019 | +0,0101 | +0,0174 | +0,0200 | Rất yếu do tính đa trị của lưu lượng |

* **Kiểm chứng `Risk Ratio` cho sự kiện hiếm:** Khi lọc riêng trạng thái chưa tắc, tỷ lệ nền sắp khởi phát tắc thực rất thấp (`1,56%` ở `10p` và `2,34%` ở `20p`), khiến `Pearson r` bị nén toán học. Xác suất có điều kiện phân tách cực mạnh:
  * **`Độ chiếm dụng đầu nguồn(t)`:** Top `10%` cao nhất (`≥ 0,133`) làm xác suất tắc sau `20 phút` tăng lên **`7,58%`** so với **`0,02%`** ở đáy `10%` (`≤ 0,022`) — **gấp `433,9 lần`**.
  * **`Δ Vận tốc (15p)`:** Nhóm tụt sâu nhất (`≤ -3,20 mph`) có xác suất sắp tắc sau `10 phút` đạt **`7,08%`** so với **`0,42%`** khi đi ngang (`|Δ| < 1 mph`) — **gấp `16,7 lần`**.
* **Biểu đồ minh họa:** [`nb03_rate_of_change_correlation.png`](../outputs/figures/nb03_rate_of_change_correlation.png)

---

## 5. Chốt Định nghĩa Nhãn & Mốc Thời gian Dự báo cho Bước 04

* **Định nghĩa nhãn mục tiêu thống nhất:** **`(Vận tốc < 40 mph) & (Độ chiếm dụng ≥ 0.15)`** trên **Top 20 điểm nghẽn** (giữ nguyên toàn bộ `170` trạm làm nút đồ thị đầu vào để truyền tín hiệu lân cận; giữ đủ chuỗi 24h liên tục).
* **Mốc thời gian dự báo trước ($\Delta t$) chuyển sang Bước 04:**
  * **Cửa sổ trọng tâm: `10 – 20 phút`**, với **mốc chuẩn trung tâm là `15 phút`** (khớp điểm gãy công suất `Flow = 433,9` tại `t = -15p` và cực đại tương quan `Vận tốc(t) = -0,323`; `10 phút` bám sát biến thiên tốc độ cục bộ; `20 phút` khai thác mạnh tín hiệu dồn ứ đầu nguồn).
  * **Dải mốc kiểm thử đầy đủ:** **$\Delta t \in \{5, 10, 15, 20, 30\}$ phút** (`5 phút` làm tham chiếu cận hạn; `30 phút` làm giới hạn trên khớp đỉnh tương quan phần dư đầu nguồn; không chọn `45–60 phút` làm mốc chính do tín hiệu đầu nguồn suy giảm rõ rệt sau `30 phút`).

**Tiếp theo:** Triển khai thử nghiệm, đối chiếu mô hình và giải thích dự báo tại [`04_Modeling_and_Explainability.ipynb`](../notebooks/04_Modeling_and_Explainability.ipynb).
