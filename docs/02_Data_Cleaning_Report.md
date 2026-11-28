# Báo cáo Bước 02: Làm sạch Dữ liệu & Phục hồi Cảm biến theo Chen et al. (2003)

* **Tệp nguồn:** [`02_Data_Cleaning_and_Interpolation.ipynb`](../notebooks/02_Data_Cleaning_and_Interpolation.ipynb) · Mô-đun tái sử dụng: [`src/cleaning.py`](../src/cleaning.py).
* **Dữ liệu đầu ra đã lưu:** `datasets/cleaned/PEMS08_clean.npz` và `datasets/cleaned/PEMS04_clean.npz` *(kèm mặt nạ lỗi `fault_mask_3ch`, nằm trong `datasets/` được `.gitignore` bỏ qua khi đẩy Git)*.
* **Mục tiêu:** Áp dụng thống nhất trọn vẹn quy trình chẩn đoán 4 chỉ số ngày ($S_1 \dots S_4$) và phục hồi bằng hồi quy tuyến tính cặp trạm lân cận của **Chen et al. (*Transportation Research Record*, 2003)** nhằm làm sạch hoàn toàn cả lỗi mất tín hiệu `0.0` lẫn lỗi cảm biến đóng băng khác `0.0` trên hai bộ dữ liệu `PEMS08` và `PEMS04`.

---

## 1. Chẩn đoán Lỗi Cảm biến theo 4 Chỉ số Ngày ($S_1 \dots S_4$) của Chen et al. (2003)

Áp dụng thuật toán thống kê hàng ngày (*Daily Statistics Algorithm*) của Chen et al. (2003) trên toàn bộ `10.540` trạm–ngày của `PEMS08` (`62` ngày × `170` trạm) và `18.113` trạm–ngày của `PEMS04` (`59` ngày × `307` trạm):

| Bộ dữ liệu | Tổng trạm–ngày | Ngày có $S_1 > 0$ | Ngày chết hẳn ($S_1 = 288$) | Điểm lệch pha $S_2$ | Điểm $S_3$ ($\text{Occ} \ge 0,70$) | $S_4$ Trung vị (`bit`) | $S_4$ Phân vị `P1` (`bit`) | Ngày đóng băng $S_4 = 0$ | Ngày $S_4 < 2,0$ `bit` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PEMS08`** | 10.540 | 763 | 1 | 0 | 27 | 4,075 | 1,975 | 1 | 112 |
| **`PEMS04`** | 18.113 | 2.115 | 6 | 151 | 7 | 4,032 | 2,528 | **10** | 81 |

> **Nhận xét (Phát hiện quan trọng từ chỉ số $S_4$ — Entropy ngày):**
>
> * Trên `PEMS04`, số trạm–ngày có $S_4 = 0$ (đứng hình hoàn toàn 24 giờ) lên tới `10` ngày dù chỉ có `6` ngày chết ở `0.0` ($S_1 = 288$) — chứng minh tồn tại **`4` ngày đóng băng trọn 24 giờ tại một hằng số khác `0.0`** (điển hình `N164` đứng hình liên tiếp `306` bước = `25,5 giờ` trên cả 3 kênh `Flow`, `Occ`, `Speed`).
>
> * Trên `PEMS08`, $S_4$ cũng phát hiện nhiều đoạn cảm biến đóng băng hằng số khác `0.0` như `N107` đứng hình `101` bước (`8,4 giờ`) cả 3 kênh — dạng lỗi bị bỏ sót hoàn toàn nếu chỉ lọc giá trị `0.0`.

Biểu đồ chẩn đoán 4 chỉ số và ca thực tế: [`nb02_chen2003_diagnostics.png`](../outputs/figures/nb02_chen2003_diagnostics.png)

---

## 2. Hiệu chuẩn Ngưỡng Thực nghiệm & Tách lọc Mặt nạ Lỗi 3 Kênh ($\mathbf{M}_{\text{fault}}$)

Hiệu chuẩn ngưỡng `P1_day`, `P1_night` riêng trên nhóm trạm hoàn toàn không có `0.0` của từng bản đồ và `P95_dur` trên các chuỗi đêm lưu lượng thấp, tách rạch ròi đường vắng ban đêm hợp lệ khỏi **Lỗi Loại A (mất xung `0.0`, $S_1/S_2$)** và **Lỗi Loại B (đóng băng hằng số dương $\ge 12$ bước = `1 giờ`, $S_4$)**:

| Bộ dữ liệu | Trạm sạch chuẩn | `P1_day` / `P1_night` | `P95_dur` | Đường vắng đêm *(Giữ)* | Lỗi Loại A — `0.0` ($S_1/S_2$) | Lỗi Loại B — Đóng băng ($S_4$) | Tổng lỗi `Flow/Occ` | Lỗi `Speed` đóng băng | Đoạn lỗi dài nhất |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`PEMS08`** | `120/170` | `69 / 16` | `6` bước (`30p`) | 7.583 | **2.977** (`39` trạm) | **751** (`10` trạm) | **3.728** (`0,123%`) | **2.211** (`0,073%`) | `0.0`: `400`b (`33,3h`, `N30`) · `S4`: `101`b (`8,4h`, `N107`) |
| **`PEMS04`** | `138/307` | `53 / 11` | `19` bước (`95p`) | 47.689 | **35.376** (`158` trạm) | **12.664** (`19` trạm) | **48.040** (`0,921%`) | **37.361** (`0,716%`) | `0.0`: `408`b (`34,0h`, `N126`) · `S4`: `354`b (`29,5h`, `N185`) |

Biểu đồ phân tách lỗi và phân bố độ dài đoạn mất: [`nb02_fault_mask_audit.png`](../outputs/figures/nb02_fault_mask_audit.png)

---

## 3. Kiểm chứng Có Kiểm soát Thuật toán Hồi quy Không gian Chen et al. (2003)

Với mỗi trạm lỗi $i$, khớp mô hình hồi quy tuyến tính cặp $\hat{x}_i(t) = \hat{a}_{ij} x_j(t) + \hat{b}_{ij}$ theo tối đa $K = 5$ trạm lân cận tương quan cao nhất (ưu tiên lân cận `1–2 hop` trên đồ thị cao tốc) trên các mốc cùng sạch, rồi tổ hợp có trọng số theo $R^2_{ij}$ với chặn dưới dương $x_{i,\min}^{+}$ từ phân vị `P1` dương của trạm $i$.

Kết quả kiểm thử có kiểm soát (`0%` rò rỉ dữ liệu, `SEED = 42`) trên quần thể trạm hoàn toàn sạch của `PEMS08` (`126/170` trạm) và `PEMS04` (`135/307` trạm):

| Bộ dữ liệu | Chế độ che khuất | Số điểm kiểm thử | `Flow MAE` | `Flow RMSE` | `Flow r` | `VolRat Flow` | `Occ MAE` | `Occ r` | `Speed MAE` (`mph`) | `Speed r` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PEMS08`** | Ngắn (`30 phút`, `L = 6`) | 540 | 21,74 | 37,16 | 0,9480 | 0,52 | 0,00735 | 0,9396 | 1,36 | 0,7517 |
| **`PEMS08`** | Trung bình (`2 giờ`, `L = 24`) | 1.200 | 23,86 | 38,93 | 0,9672 | 0,47 | 0,01183 | 0,9007 | 2,18 | 0,8078 |
| **`PEMS08`** | Dài (`6 giờ`, `L = 72`) | 2.880 | **24,95** | **36,87** | **0,9711** | **0,52** | 0,01523 | 0,8754 | 2,87 | 0,7870 |
| **`PEMS04`** | Ngắn (`30 phút`, `L = 6`) | 540 | 28,47 | 52,17 | 0,9384 | 0,39 | 0,00783 | 0,9144 | 1,80 | 0,7505 |
| **`PEMS04`** | Trung bình (`2 giờ`, `L = 24`) | 1.200 | 36,03 | 58,40 | 0,9364 | 0,46 | 0,01428 | 0,7509 | 2,76 | 0,6744 |
| **`PEMS04`** | Dài (`6 giờ`, `L = 72`) | 2.880 | **23,98** | **35,08** | **0,9671** | **0,38** | 0,01343 | 0,7883 | 2,69 | 0,7919 |

> **Nhận xét:**
>
> * Thuật toán Chen et al. (2003) giữ độ ổn định rất cao ngay cả khi đoạn mất kéo dài gấp `12` lần từ `30 phút` lên `6 giờ` (`Flow r = 0,9711` trên `PEMS08` và `0,9671` trên `PEMS04`), đồng thời bảo toàn được `38%–52%` biên độ dao động 5 phút (`VolRat`) nhờ bám sát biến thiên thực từ các cảm biến lân cận.

Biểu đồ đối chiếu phục hồi 3 biến: [`nb02_chen2003_imputation_eval.png`](../outputs/figures/nb02_chen2003_imputation_eval.png)

---

## 4. Kết quả Áp dụng Chính thức & Bảo toàn Phân phối

Tỷ lệ điểm lỗi được phục hồi trực tiếp từ hồi quy không gian cặp lân cận đạt **`96,5%`** trên `PEMS08` (trung vị $R^2 = 0,5862$) và **`97,8%`** trên `PEMS04` (trung vị $R^2 = 0,6447$). Kiểm toán hậu làm sạch xác nhận **`0` lỗi tồn đọng** ở tất cả các kênh:

| Bộ dữ liệu | Biến đo lường | TB Trước | TB Sau | Std Trước | Std Sau | `Step-diff Std` Trước | `Step-diff Std` Sau | Lỗi tồn đọng |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PEMS08`** | Lưu lượng (`Flow`) | 230,6807 | 230,8092 | 146,2169 | 146,1292 | 25,7801 | 25,4767 | **0** |
| **`PEMS08`** | Độ chiếm dụng (`Occ`) | 0,0651 | 0,0651 | 0,0459 | 0,0459 | 0,0125 | 0,0125 | **0** |
| **`PEMS08`** | Vận tốc (`Speed`) | 63,7630 | 63,7633 | 6,6520 | 6,6499 | 1,4963 | 1,4969 | **0** |
| **`PEMS04`** | Lưu lượng (`Flow`) | 211,7008 | 212,4267 | 158,0684 | 157,7368 | 34,1051 | 34,0147 | **0** |
| **`PEMS04`** | Độ chiếm dụng (`Occ`) | 0,0528 | 0,0530 | 0,0495 | 0,0494 | 0,0149 | 0,0149 | **0** |
| **`PEMS04`** | Vận tốc (`Speed`) | 63,4706 | 63,4563 | 8,3557 | 8,3520 | 1,7850 | 1,7864 | **0** |

Biểu đồ kiểm chứng trước–sau làm sạch: [`nb02_before_after_verification.png`](../outputs/figures/nb02_before_after_verification.png)

### Câu hỏi nghi vấn chuyển sang Bước 03 (`RQ1`)
> *Trên bộ dữ liệu sạch hoàn chỉnh của `PEMS08` (trong khi giữ `PEMS04` độc lập hoàn toàn cho kiểm chứng chéo ở `NB05`), làm thế nào xác định trạng thái tắc nghẽn trực tiếp từ Biểu đồ cơ bản, trích xuất và kiểm chứng hệ chỉ báo tiền tắc nghẽn trên cửa sổ `60 phút`, và đánh giá vai trò của cấu trúc đồ thị mạng lưới (tĩnh và động)?*

**Tiếp theo:** [`03_EDA_and_Bottleneck_Report.md`](./03_EDA_and_Bottleneck_Report.md)
