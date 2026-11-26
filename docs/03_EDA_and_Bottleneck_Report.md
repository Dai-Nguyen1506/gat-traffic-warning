# Báo cáo Bước 03: Khám phá Dữ liệu Sạch (EDA) & Phân tích Điểm nghẽn

* **Tệp nguồn:** [`03_EDA_and_Bottleneck_Analysis.ipynb`](../notebooks/03_EDA_and_Bottleneck_Analysis.ipynb)
* **Dữ liệu đầu vào:** `datasets/cleaned/PEMS08_clean.npz` (`17.856 × 170 × 3`, `277` cạnh có hướng duy nhất sau khi gộp `18` cạnh trùng từ `295` dòng CSV gốc) và `datasets/cleaned/PEMS04_clean.npz` (`16.992 × 307 × 3`, `340` cạnh).
* **Tệp đầu ra bàn giao:** [`PEMS08_bottlenecks.csv`](../outputs/PEMS08_bottlenecks.csv), [`PEMS04_bottlenecks.csv`](../outputs/PEMS04_bottlenecks.csv), [`PEMS08_network.gexf`](../outputs/PEMS08_network.gexf), [`PEMS04_network.gexf`](../outputs/PEMS04_network.gexf) cùng `5` biểu đồ `300 DPI` tại `outputs/figures/nb03_*.png`.

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
> * **Tiêu chí chuẩn chốt thống nhất cho toàn bộ các bước tiếp theo:** **`Tắc nghẽn thực sự ⇔ (Vận tốc < 40 mph) và (Độ chiếm dụng ≥ 0.15)`**.

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
> * Vào đêm khuya (`00–04h`) vẫn tồn tại **`243` ca tắc cục bộ thật sự** (`Occ TV = 0,244`, `Speed TV = 13,1 mph`, `Flow TV = 104,0` cao hơn cả lúc thông thoáng ban đêm `82,0`), đồng thời không còn nhiễu đường vắng. Nhờ tiêu chí 3 biến đã làm sạch nhãn tự nhiên, các Bước 04–05 có thể giữ nguyên chuỗi thời gian liên tục 24h (không cần cắt bỏ đêm làm đứt gãy chuỗi) và kiểm chứng bổ sung trên riêng khung `05–23h`.

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

> **Nhận xét:**
>
> * Hình học đồ thị tĩnh đơn lẻ không đủ dự báo nhất quán vị trí điểm nghẽn trên mọi mạng lưới (ví dụ `N9` tắc `18,0%` và `N65` tắc `15,6%` dù có `Betweenness` rất thấp) → cần chuyển sang kiểm chứng động lực chuỗi thời gian và tín hiệu từ các nhánh lân cận trước khi tắc nghẽn bùng phát.

* **Biểu đồ minh họa:** [`nb03_network_map_pems08.png`](../outputs/figures/nb03_network_map_pems08.png)

---

## 4. Tổng kết Bàn giao sang Bước 04

* **Tiêu chí tắc nghẽn thực thống nhất:** **`(Vận tốc < 40 mph) & (Độ chiếm dụng ≥ 0,15)`**, giữ nguyên chuỗi 24h liên tục.
* **Tập mục tiêu trọng tâm:** **Top 20 điểm nghẽn `PEMS08`** (gánh `71,2%` tổng thời lượng tắc thực, bền vững `20/20` trong khung `05–23h`), giữ toàn bộ `170` trạm làm nút đồ thị đầu vào.
* **Tệp bàn giao:** `outputs/PEMS08_bottlenecks.csv`, `outputs/PEMS04_bottlenecks.csv`, `outputs/PEMS08_network.gexf`, `outputs/PEMS04_network.gexf` và `5` biểu đồ `nb03_*.png`.

**Tiếp theo:** Kiểm chứng dấu hiệu tiền tắc nghẽn và chốt bộ tham số đầu vào – đầu ra (`H*`, đồ thị 2 chiều đa nhánh, `Δt*`) tại [`04_Pre_Congestion_Signal_Analysis.ipynb`](../notebooks/04_Pre_Congestion_Signal_Analysis.ipynb) ([Báo cáo Bước 04](./04_Pre_Congestion_Signal_Report.md)).
