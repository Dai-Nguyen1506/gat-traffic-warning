# Báo cáo Bước 02: Làm sạch Dữ liệu & Nội suy Chuỗi Thời gian

* **Tệp nguồn:** [`02_Data_Cleaning_and_Interpolation.ipynb`](../notebooks/02_Data_Cleaning_and_Interpolation.ipynb)
* **Dữ liệu đầu ra đã lưu:** `datasets/cleaned/PEMS08_clean.npz` và `datasets/cleaned/PEMS04_clean.npz` *(lưu trong `datasets/` để `.gitignore` bỏ qua khi đẩy Git)*.
* **Mục tiêu:** Thiết lập tiêu chí thực nghiệm tách biệt trạng thái đường vắng ban đêm hợp lệ với lỗi mất tín hiệu cảm biến, kiểm định chiến lược phục hồi tối ưu và tạo bộ dữ liệu sạch hoàn toàn cho `PEMS08` và `PEMS04`.

---

## 1. Cơ sở Thực nghiệm Phân biệt Đường vắng & Lỗi Cảm biến

Do `Vận tốc` không có giá trị `0.0` (`Min = 3.0 mph`), việc tách bạch các điểm `<= 0` tập trung vào `Lưu lượng` và `Độ chiếm dụng`. Để tránh gán ngưỡng tùy tiện, toàn bộ tham số được đo trực tiếp từ hai tập con tách biệt trên `PEMS08`:

1. **Trên nhóm trạm không có giá trị `<= 0` trong suốt 62 ngày (`120 / 170` trạm sạch):** Tính phân vị cực trị dưới `1%` (`P1` — mức ngưỡng mà `99%` quan sát thông thường đều vượt qua) cho ban ngày (`06:00–22:00`) và ban đêm (`22:00–06:00`). Kết quả: **`P1_day = 69.0`** và **`P1_night = 16.0` xe/5 phút**.

2. **Trên nhóm trạm còn lại có xuất hiện `0.0` (`50 / 170` trạm):** Lọc riêng các chuỗi `0.0` xảy ra hoàn toàn vào ban đêm với lưu lượng bước liền kề ở mức thấp (`≤ 16.0` xe/5 phút), tính phân vị `95%` (`P95`) độ dài chuỗi. Kết quả: **`P95_dur = 6` bước (`30` phút)**.

> **Nhận xét (Bộ 4 tiêu chí nhận diện lỗi mất tín hiệu):**
>
> Từ hai ngưỡng thực nghiệm độc lập trên, một quan sát hoặc chuỗi `<= 0` bị gán nhãn là **lỗi cảm biến** nếu vi phạm ít nhất một trong 4 điều kiện:
>
> 1. Độ dài chuỗi liên tiếp lớn hơn **`6` bước (`30` phút)**;
>
> 2. Mất đồng bộ giữa hai kênh `Lưu lượng` và `Độ chiếm dụng` (một biến `<= 0` nhưng biến còn lại `> 0`);
>
> 3. Xuất hiện vào khung giờ ban ngày (`06:00–22:00`) với lưu lượng bước liền kề ≥ `16.0` xe/5 phút;
>
> 4. Sụt thẳng về `0.0` từ mức lưu lượng nền ban ngày ≥ `69.0` xe/5 phút.
>
> **Câu hỏi nghi vấn tiếp theo:**
>
> Khi áp dụng 4 điều kiện phân loại này lên toàn bộ `PEMS08` và `PEMS04`, có bao nhiêu điểm `0.0` thực sự là lỗi cảm biến cần phục hồi và đoạn mất tín hiệu dài nhất kéo dài bao lâu?

Biểu đồ ngưỡng thực nghiệm: [`nb02_empirical_thresholds.png`](../outputs/figures/nb02_empirical_thresholds.png)

---

## 2. Kiểm toán & Phân loại các Điểm `0.0` trên `PEMS08` và `PEMS04`

| Bộ dữ liệu | Tổng điểm `<= 0` | Đường vắng hợp lệ *(Giữ nguyên)* | Lỗi cảm biến *(Cần nội suy)* | Số trạm bị lỗi | Đoạn lỗi dài tối đa |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`PEMS08`** | 10.560 | **7.583 (`71,8%`)** | **2.977 (`28,2%`)** | `39 / 170` trạm | **400 bước (`~33,3` giờ)** |
| **`PEMS04`** | 83.065 | **28.120 (`33,9%`)** | **54.945 (`66,1%`)** | `157 / 307` trạm | **408 bước (`34,0` giờ)** |

> **Nhận xét (Sự khác biệt rõ rệt giữa hai tập dữ liệu):**
>
> * Trên `PEMS08`, phần lớn các giá trị `0.0` (`71,8%`) là trạng thái đường vắng ban đêm hợp lệ cần giữ nguyên để bảo toàn phân phối gốc. Ngược lại, `PEMS04` có tới `66,1%` (`54.945` điểm) là lỗi mất tín hiệu cảm biến.
>
> * Cả hai bộ dữ liệu đều xuất hiện các đoạn đứt tín hiệu kéo dài tới **hơn 33 giờ (`400–408` bước)** — nơi nội suy đường thẳng sẽ làm mất hoàn toàn nhịp cao điểm sáng/chiều.
>
> **Câu hỏi nghi vấn tiếp theo:**
>
> Trên các đoạn mất dài hàng giờ hoặc hàng ngày, phương pháp nào vừa dự đoán đúng mức lưu lượng/độ chiếm dụng theo khung giờ, vừa khôi phục được biến thiên thực từng 5 phút mà không chèn thêm nhiễu giả?

Biểu đồ kiểm toán lỗi: [`nb02_anomaly_audit.png`](../outputs/figures/nb02_anomaly_audit.png)

---

## 3. Thực nghiệm Nội suy & Xác định Điểm Đảo chiều `L*`

### 3.1. Thử nghiệm khôi phục phần dư trên đoạn mất dài (Che khuất 6 giờ = `72` bước trên `20` trạm sạch `PEMS08`)

Thay vì chỉ điền đường trung vị chu kỳ ngày (`Daily Profile` — làm phẳng mọi dao động), nghiên cứu thử nghiệm 4 chiến lược kết hợp:

| Phương pháp thử nghiệm | `MAE` | `RMSE` | Tương quan (`r`) | Độ lệch chuẩn sai phân (`Step-diff Std`) | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Ground Truth` (Gốc đối chiếu)** | 0,00 | 0,00 | 1,0000 | 32,25 | Chuẩn thực tế cần tiệm cận |
| **`Daily Profile` (Trung vị theo giờ)** | 26,34 | 39,83 | 0,9606 | 10,29 | Mất phần lớn dao động từng 5 phút |
| **`Profile + AR(1) Noise`** | 50,73 | 70,69 | 0,8890 | 39,02 | Sai số điểm tăng gần gấp đôi vì nhiễu giả |
| **`Profile + Spatial Mean`** *(Chọn)* | **23,45** | **33,47** | **0,9718** | **20,41** | Tốt nhất cả về độ chính xác lẫn phục hồi dao động |
| **`Profile + Spatial Ridge`** | 25,02 | 35,72 | 0,9680 | 21,15 | Kém hơn nhẹ do đa cộng tuyến nút lân cận |

### 3.2. Quét độ dài đoạn mất `L = 1..72` bước (`5 – 360` phút) để tìm điểm đảo chiều `L*`

| Độ dài đoạn (`L`) | Thời gian | `Linear MAE` | `Daily Profile MAE` | `Profile + Spatial Mean MAE` | Phương pháp tốt hơn |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `1` bước | `5 phút` | **18,61** | 26,14 | 21,62 | `Linear` |
| `2` bước | `10 phút` | **16,90** | 26,43 | 24,85 | `Linear` |
| `6` bước | `30 phút` | **20,60** | 26,29 | 25,32 | `Linear` |
| `12` bước | `60 phút` | **24,97** | 28,74 | 26,56 | `Linear` |
| **`18` bước (`L*`)** | **`90 phút`** | **23,73** | 29,02 | 26,41 | `Linear` *(Giới hạn trên)* |
| `24` bước | `120 phút` | 25,16 | 28,04 | **24,25** | `Profile + Spatial Mean` |
| `36` bước | `180 phút` | 29,61 | 27,32 | **24,82** | `Profile + Spatial Mean` |
| `72` bước | `360 phút` | 57,91 | 26,34 | **23,45** | `Profile + Spatial Mean` |

> **Nhận xét (Chiến lược nội suy kết hợp `Hybrid`):**
>
> * Trong phạm vi **`L ≤ 18` bước (`≤ 90` phút)**, nội suy tuyến tính (`Linear`) giữ sai số thấp nhất (`MAE = 16,90–24,97`) nhờ bám trực tiếp hai đầu mút thực tế ngay tại trạm đó.
>
> * Từ **`L ≥ 24` bước (`≥ 120` phút)** trở đi, sai số của `Linear` phình nhanh lên `25,16 → 57,91`, trong khi `Profile + Spatial Mean` đảo chiều vượt lên dẫn đầu ổn định quanh `23,45–24,92`. Do đó, điểm **`L* = 18` bước (`90` phút)** được chọn làm ngưỡng chuyển đổi cho chiến lược **Hybrid**.
>
> **Câu hỏi nghi vấn tiếp theo:**
>
> Sau khi áp dụng chiến lược Hybrid lên toàn bộ đoạn lỗi thực tế của `PEMS08` và `PEMS04`, bộ dữ liệu sạch có triệt tiêu hoàn toàn điểm lỗi mà vẫn giữ nguyên phân phối thống kê ban đầu hay không?

Biểu đồ đối chiếu nội suy: [`nb02_interpolation_and_residual_models.png`](../outputs/figures/nb02_interpolation_and_residual_models.png)

---

## 4. Kết quả Áp dụng Cuối cùng & Bảo toàn Phân phối

| Bộ dữ liệu | Biến phục hồi | TB Trước | TB Sau | Std Trước | Std Sau | `Step-diff Std Sau` | Số lỗi còn lại |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PEMS08`** | Lưu lượng | 230,681 | 230,766 | 146,217 | 146,155 | 25,418 | **0** |
| **`PEMS08`** | Độ chiếm dụng | 0,065 | 0,065 | 0,046 | 0,046 | 0,012 | **0** |
| **`PEMS04`** | Lưu lượng | 211,701 | 212,249 | 158,068 | 157,871 | 33,970 | **0** |
| **`PEMS04`** | Độ chiếm dụng | 0,053 | 0,053 | 0,049 | 0,049 | 0,015 | **0** |

* **Chốt tham số bàn giao từ Bước 02:** `P1_day = 69.0`, `P1_night = 16.0`, `P95_dur = 6` bước (`30 phút`), điểm đảo chiều Hybrid **`L* = 18` bước (`90 phút`)** dùng `Profile + Spatial Mean` cho đoạn mất dài.
* Hai tệp dữ liệu sạch `datasets/cleaned/PEMS08_clean.npz` và `datasets/cleaned/PEMS04_clean.npz` giữ trung bình và độ lệch chuẩn gần như trùng khít bản gốc và không còn bất kỳ điểm lỗi cảm biến nào (`0` lỗi tồn đọng).

### Câu hỏi nghi vấn chuyển sang Bước 03
> *Khi đã có chuỗi dữ liệu sạch không còn lỗi cảm biến, hệ thống giao thông vận hành theo nhịp điệu ngày – tuần ra sao? Ranh giới vật lý đa biến nào phân tách trạng thái thông thoáng với tắc nghẽn thực sự, ùn tắc có tập trung ở một nhóm trạm cố định hay không, và liệu có dấu hiệu tiền tắc nghẽn nào xuất hiện trước lúc kẹt xe?*

**Tiếp theo:** [`03_EDA_and_Bottleneck_Report.md`](./03_EDA_and_Bottleneck_Report.md)
