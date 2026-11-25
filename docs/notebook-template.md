# Quy chuẩn Template cho Jupyter Notebook

Tài liệu này gợi ý cấu trúc gọn nhẹ, tự nhiên cho các notebook trong dự án để dễ đọc và nhất quán giữa các bước.

---

## 1. Quy ước đặt tên file

| Số thứ tự | Tên file | Nội dung chính |
| :---: | :--- | :--- |
| **00** | `00_Notebook_Template.ipynb` | File mẫu cơ bản dùng để nhân bản khi tạo notebook mới. |
| **01** | `01_Overview.ipynb` | Tổng quan đề tài, RQ1–RQ2, công cụ, lộ trình 6 bước và kiểm toán 4 bộ dữ liệu gốc. |
| **02** | `02_Data_Cleaning_and_Interpolation.ipynb` | Kiểm toán lỗi cảm biến và nội suy làm sạch dữ liệu. |
| **03** | `03_EDA_and_Bottleneck_Analysis.ipynb` | Khám phá dữ liệu sạch, định danh điểm nghẽn và phân tích tín hiệu tiền tắc nghẽn. |
| **04** | `04_Modeling_and_Explainability.ipynb` | Thiết kế, thử nghiệm, lựa chọn mô hình dự báo cảnh báo sớm và giải thích mô hình. |
| **05** | `05_Simulation_and_Cross_Validation.ipynb` | Giả lập điều tiết lưu lượng đầu nguồn và kiểm chứng chéo sang PEMS04. |
| **06** | `06_Research_Synthesis_and_Conclusions.ipynb` | Tổng hợp toàn bộ bằng chứng định lượng và kết luận trọn vẹn RQ1 – RQ2. |

---

## 2. Cấu trúc gợi ý bên trong mỗi Notebook

Không ép buộc tách riêng phần trực quan hóa — **biểu đồ và bảng số liệu được đặt xen kẽ tự nhiên ngay tại mục phân tích tương ứng** để mạch diễn giải liền mạch:

1. **Phần mở đầu:**
   * `# [Số]. Tiêu đề Notebook`
   * 1–2 câu mô tả ngắn phạm vi notebook.
   * `### Tóm tắt` (*Bối cảnh*, *Mục tiêu*, *Kết quả đầu ra*).
2. **`## 1. Thiết lập môi trường & Nạp dữ liệu`:** Khai báo thư viện, đặt `SEED = 42`, cấu hình đường dẫn và nạp dữ liệu đầu vào.
3. **`## 2..N. Các mục phân tích chính`:** Chia theo chủ đề thực tế của từng notebook; mỗi mục kết hợp code tính toán, bảng/biểu đồ minh họa trực tiếp và khối `> **Nhận xét:** ...` ngay bên dưới.
4. **`## Tổng kết & Bước tiếp theo`:** Tóm lược các kết quả đã chốt, file đã lưu, định hướng sang notebook kế tiếp, và kết thúc bằng dòng liên kết `**Tiếp theo:** [...]` tới file notebook tiếp theo.

---

## 3. Lưu ý chung
* Đặt `SEED = 42` để đảm bảo khả năng tái lập.
* Các biểu đồ quan trọng có thể lưu vào `outputs/figures/` ở độ phân giải 300 DPI.
