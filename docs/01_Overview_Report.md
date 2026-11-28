# Báo cáo Bước 01: Tổng quan Dự án & Lựa chọn Tập dữ liệu

* **Tệp nguồn:** [`01_Overview.ipynb`](../notebooks/01_Overview.ipynb)
* **Mục tiêu:** Đặt nền tảng vấn đề nghiên cứu (`RQ1`, `RQ2`), xác lập công cụ triển khai, kiểm toán cấu trúc không gian – thời gian và chất lượng thô của 4 bộ dữ liệu PeMS (`PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`) nhằm phân vai nghiên cứu có căn cứ định lượng.

---

## 1. Bối cảnh, Câu hỏi Nghiên cứu & Lộ trình

Trên mạng lưới đường cao tốc đô thị, ùn tắc hiếm khi bùng phát đồng loạt ở mọi tuyến đường mà thường khởi phát tại một số **nút thắt cổ chai** rồi lan truyền ngược dòng sang các đoạn lân cận. Các hệ thống giám sát truyền thống chỉ cảnh báo sau khi vận tốc tại chỗ đã giảm sâu, khiến công tác điều tiết rơi vào thế bị động.

* **Đề tài nghiên cứu:** *Tín hiệu Tiền tắc nghẽn Hướng Vật lý cho Bài toán Dự báo Điểm nghẽn Cao tốc bằng Mạng Chú ý Đồ thị* (*Physics-Informed Pre-Congestion Signals for Freeway Bottleneck Forecasting with Graph Attention*).
* **Hai Câu hỏi Nghiên cứu (RQ1 – RQ2):**
  1. **RQ1 (Xác định tắc nghẽn & Tín hiệu tiền tắc nghẽn — `NB03` trên `PEMS08`):** *Làm thế nào xác định trạng thái tắc nghẽn và trích xuất các dấu hiệu tiền tắc nghẽn trực tiếp từ dữ liệu cảm biến đa biến?*
  2. **RQ2 (Đánh giá đóng góp đặc trưng trên mô hình dự báo đồ thị — `NB04` trên `PEMS08`, kiểm chứng độc lập trên `PEMS04` ở `NB05`):** *Việc bổ sung các dấu hiệu tiền tắc nghẽn có giúp các mô hình dự báo đồ thị (gồm mô hình chú ý đồ thị) dự đoán điểm tắc nghẽn tương lai tốt hơn không, và từng dấu hiệu đóng góp ra sao?*

### Hệ thống công cụ & Lộ trình 5 bước
* **Công cụ kỹ thuật:** `Python 3.12`, `PyTorch`, `Scikit-learn`, `NetworkX`, `NumPy`, `Pandas`, `SciPy`, `Matplotlib`, `Seaborn` (cố định `SEED = 42`).
* **Quy trình 5 bước:** (`01`) Tổng quan & Chọn dữ liệu → (`02`) Làm sạch & Phục hồi theo Chen et al. (2003) → (`03`) Khám phá, Xác định Tắc nghẽn & Tín hiệu Tiền tắc nghẽn (`RQ1`, thuần `PEMS08`) → (`04`) Mô hình hóa & Đánh giá Đóng góp Đặc trưng (`RQ2`, `PEMS08`) → (`05`) Kiểm chứng chéo Độc lập trên `PEMS04`.

---

## 2. Đối chiếu Cấu trúc Không gian – Thời gian của 4 Bộ dữ liệu

Mỗi tập dữ liệu PeMS ghi nhận theo chu kỳ **5 phút/bước** (`288` bước/ngày). Yêu cầu tối thiểu của đề tài là phải có **đồng thời cấu trúc cạnh có hướng** và **đầy đủ 3 biến đo lường** (`Lưu lượng`, `Độ chiếm dụng`, `Vận tốc`).

| Bộ dữ liệu | Số bước (`T`) | Thời lượng | Số trạm (`N`) | Số cạnh (`E`) | Bậc TB (`2E/N`) | Số biến (`C`) | Khớp chỉ số `0..N-1` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`PEMS03`** | 26.208 | 91,0 ngày | 358 | 547 | 3,06 | **1** (`Lưu lượng`) | Không *(dùng mã gốc + `.txt`)* |
| **`PEMS04`** | 16.992 | 59,0 ngày | 307 | 340 | 2,21 | **3** *(Đầy đủ)* | Có |
| **`PEMS07`** | 28.224 | 98,0 ngày | 883 | 866 | 1,96 | **1** (`Lưu lượng`) | Có |
| **`PEMS08`** | 17.856 | 62,0 ngày | 170 | 295 | **3,47** | **3** *(Đầy đủ)* | Có |

> **Nhận xét (Cấu trúc 4 bộ dữ liệu):**
>
> * `PEMS03` (`9,382,464` điểm) và `PEMS07` (`24,921,792` điểm) chỉ có duy nhất 1 kênh `Lưu lượng` (`C = 1`), thiếu hoàn toàn `Độ chiếm dụng` và `Vận tốc` nên không thể quan sát tình trạng tụt giảm vận tốc hay bão hòa mặt đường; thêm vào đó `PEMS03` chưa đồng bộ chỉ số nút `0..N-1` (`distance` tính theo dặm `~0.8–1.3`).
>
> * Chỉ có **`PEMS04`** (`15,649,632` điểm) và **`PEMS08`** (`9,106,560` điểm) cung cấp đủ cả 3 biến đo lường và đồng bộ chỉ số trạm, trọng số cạnh `.csv` (`cost`) tính theo mét (`~310–393 m`).
>
> **Câu hỏi nghi vấn tiếp theo:**
>
> Về mặt cấu trúc vỏ, `PEMS08` và `PEMS04` đều đạt chuẩn; nhưng bên trong giá trị số học thực tế có tồn tại giá trị khuyết (`NaN`), sự khác biệt thang đo hay các điểm bất thường bằng `0.0` cần xử lý trước khi phân tích hay không?

Biểu đồ đối chiếu: [`nb01_dataset_overview.png`](../outputs/figures/nb01_dataset_overview.png)

---

## 3. Kiểm toán Chất lượng Gốc & Phân phối 3 Biến

| Bộ dữ liệu | Biến đo lường | `NaN` | Tỷ lệ `<= 0` | `Min` | Trung vị | Trung bình | Độ lệch chuẩn | `Max` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PEMS03`** | Lưu lượng | 0 | 0,672% | 0,0 | 136,00 | 179,26 | 143,71 | 1852,00 |
| **`PEMS07`** | Lưu lượng | 0 | 0,452% | 0,0 | 304,00 | 308,52 | 188,25 | 1498,00 |
| **`PEMS04`** | Lưu lượng | 0 | 1,590% | 0,0 | 180,00 | 211,70 | 158,07 | 919,00 |
| **`PEMS04`** | Độ chiếm dụng | 0 | 1,592% | 0,0 | 0,04 | 0,05 | 0,05 | 0,77 |
| **`PEMS04`** | Vận tốc (`mph`) | 0 | 0,000% | 3,0 | 65,60 | 63,47 | 8,36 | 85,20 |
| **`PEMS08`** | Lưu lượng | 0 | **0,348%** | 0,0 | 215,00 | 230,68 | 146,22 | 1147,00 |
| **`PEMS08`** | Độ chiếm dụng | 0 | **0,348%** | 0,0 | 0,06 | 0,07 | 0,05 | 0,90 |
| **`PEMS08`** | Vận tốc (`mph`) | 0 | 0,000% | 3,0 | 64,90 | 63,76 | 6,65 | 82,30 |

> **Nhận xét (Chất lượng dữ liệu gốc & Tính tương đồng thang đo):**
>
> * **Không có `NaN` tường minh nhưng xuất hiện điểm nghi vấn `0.0`:** Kênh `Vận tốc` trên cả hai tập hoàn toàn không có giá trị `0.0` (`Min = 3.0 mph`). Ngược lại, `Lưu lượng` và `Độ chiếm dụng` ghi nhận giá trị `0.0` chiếm `0,348%` trên `PEMS08` và `1,590%–1,592%` trên `PEMS04` (cao hơn gần `4,6 lần`). Hiện tại chưa thể kết luận toàn bộ các điểm `0.0` này là đường vắng ban đêm hợp lệ hay lỗi cảm biến mất tín hiệu.
>
> * **Thang đo tương đồng cao giữa `PEMS08` và `PEMS04`:** Vận tốc trung bình của hai tập rất sát nhau (`63,76 mph` và `63,47 mph`, trung vị `64,9–65,6 mph`); `Lưu lượng` và `Độ chiếm dụng` đều cùng bậc độ lớn, khẳng định tính hợp lệ khi dùng một tập làm chặng nghiên cứu chính và tập còn lại làm kiểm chứng chéo độc lập.

Biểu đồ đối chiếu phân phối 3 biến: [`nb01_raw_feature_distributions.png`](../outputs/figures/nb01_raw_feature_distributions.png)

---

## 4. Kết luận Phân vai Dữ liệu & Chuyển tiếp sang Bước 02

| Bộ dữ liệu | Vai trò chính thức | Căn cứ định lượng |
| :--- | :--- | :--- |
| **`PEMS03`** | **Loại** | Thiếu `Độ chiếm dụng` và `Vận tốc` (`C = 1`); mã nút gốc chưa chuẩn hóa trực tiếp. |
| **`PEMS07`** | **Loại** | Thiếu `Độ chiếm dụng` và `Vận tốc` (`C = 1`); mạng thưa (`bậc TB = 1,96` trên `883` trạm). |
| **`PEMS08`** | **Chặng 1 — Nghiên cứu chính (Primary)** | Đủ 3 biến; chuỗi dài hơn (`62` ngày); mật độ kết nối dày nhất (`bậc TB = 3,47`); ít điểm `<= 0` nhất (`0,348%`). |
| **`PEMS04`** | **Chặng 2 — Kiểm chứng độc lập (Validation)** | Đủ 3 biến; quy mô mạng rộng hơn (`307` trạm, `59` ngày) ở khu vực khác, phù hợp kiểm tra tổng quát hóa. |

### Câu hỏi nghi vấn chuyển sang Bước 02
> *Trong các giá trị `0.0` ở kênh Lưu lượng và Độ chiếm dụng (`0,348%` trên `PEMS08` và `1,590%` trên `PEMS04`), đâu là trạng thái đường vắng ban đêm hợp lệ cần giữ nguyên và đâu là lỗi mất tín hiệu cảm biến cần phục hồi? Với cả đoạn mất ngắn lẫn dài hàng chục giờ, phương pháp nội suy nào bảo toàn được cả giá trị điểm lẫn nhịp dao động gốc?*

**Tiếp theo:** [`02_Data_Cleaning_Report.md`](./02_Data_Cleaning_Report.md)
