# Quy chuẩn Template cho Jupyter Notebook

Tài liệu này gợi ý cấu trúc gọn nhẹ, chuẩn mực cho các notebook và báo cáo trong dự án. Trước khi biên soạn hoặc chỉnh sửa notebook mới, **bắt buộc đọc và tuân thủ toàn bộ quy tắc tại [`rules.md`](./rules.md)**.

---

## 1. Quy ước đặt tên file & Quản lý thư mục

| Số thứ tự | Tệp Notebook (`notebooks/`) | Tệp Báo cáo (`docs/`) | Nội dung chính |
| :---: | :--- | :--- | :--- |
| **00** | `00_Notebook_Template.ipynb` | — | File mẫu chuẩn hóa cấu trúc dùng để nhân bản khi tạo notebook mới. |
| **01** | `01_Overview.ipynb` | [`01_Overview_Report.md`](./01_Overview_Report.md) | Tổng quan đề tài, RQ1–RQ2, công cụ, lộ trình 7 bước và kiểm toán 4 bộ dữ liệu gốc. |
| **02** | `02_Data_Cleaning_and_Interpolation.ipynb` | [`02_Data_Cleaning_Report.md`](./02_Data_Cleaning_Report.md) | Kiểm toán lỗi cảm biến và nội suy làm sạch dữ liệu. |
| **03** | `03_EDA_and_Bottleneck_Analysis.ipynb` | [`03_EDA_and_Bottleneck_Report.md`](./03_EDA_and_Bottleneck_Report.md) | Khám phá dữ liệu sạch, chốt tiêu chí tắc nghẽn 3 biến, định danh Top 20 điểm nghẽn và phân tích đồ thị tĩnh. |
| **04** | `04_Pre_Congestion_Signal_Analysis.ipynb` | [`04_Pre_Congestion_Signal_Report.md`](./04_Pre_Congestion_Signal_Report.md) | Kiểm chứng dấu hiệu tiền tắc nghẽn: quỹ đạo pha, chốt cửa sổ `H*`, độ biến động, cấu trúc không gian đa nhánh và tìm `Δt*`. |
| **05** | `05_Modeling_and_Explainability.ipynb` | [`05_Modeling_Report.md`](./05_Modeling_Report.md) | Thiết kế, thử nghiệm, lựa chọn mô hình dự báo cảnh báo sớm (`TensorFlow / Keras`) và giải thích mô hình. |
| **06** | `06_Simulation_and_Cross_Validation.ipynb` | `06_Simulation_Report.md` *(dự kiến)* | Giả lập điều tiết lưu lượng đầu nguồn và kiểm chứng chéo sang PEMS04. |
| **07** | `07_Research_Synthesis_and_Conclusions.ipynb` | `07_Synthesis_Report.md` *(dự kiến)* | Tổng hợp toàn bộ bằng chứng định lượng và kết luận trọn vẹn RQ1 – RQ2. |

### Quy ước thư mục lưu trữ
* **Dữ liệu không đẩy lên GitHub (nằm trong `.gitignore`):** Dữ liệu thô tại `datasets/data/` và dữ liệu đã làm sạch tại `datasets/cleaned/`.
* **Đầu ra phục vụ báo cáo & trực quan hóa:** Biểu đồ `300 DPI` lưu tại `outputs/figures/nbXX_*.png`; bảng điểm nghẽn và đồ thị mạng lưu tại `outputs/*.csv`, `outputs/*.gexf`.

---

## 2. Cấu trúc chuẩn bên trong mỗi Notebook

Không tách riêng mục trực quan hóa — **biểu đồ và bảng số liệu được đặt xen kẽ tự nhiên ngay tại mục phân tích tương ứng** theo mạch hỏi – đáp liên tục:

1. **Phần mở đầu:**
   * `# [Số]. Tiêu đề Notebook`
   * 1–2 câu mô tả ngắn phạm vi notebook (văn phong khoa học, không emoji).
   * `### Tóm tắt` (*Bối cảnh*, *Mục tiêu*, *Kết quả đầu ra*).
2. **`## 1. Thiết lập môi trường & Nạp dữ liệu`:** Khai báo thư viện, đặt `SEED = 42`, cấu hình `DATA_DIR`, `CLEAN_DIR`, `OUT_DIR`, `FIG_DIR` và nạp dữ liệu đầu vào.
3. **`## 2..N. Các mục phân tích chính`:** Chia theo chủ đề thực tế; mỗi mục mở bằng câu hỏi cần giải đáp, chạy tính toán/vẽ biểu đồ minh họa và kết thúc bằng khối nhận xét thông thoáng:
   * Dòng tiêu đề `> **Nhận xét (...):**` đứng riêng 1 dòng, theo sau là **1 dòng `>` trống** trước khi vào các gạch đầu dòng.
   * Chốt khối nhận xét bằng `> **Câu hỏi nghi vấn tiếp theo:**` dẫn mượt sang mục kế tiếp.
4. **`## Tổng kết & Bước tiếp theo`:** Tóm lược định lượng toàn bộ các ngưỡng/tham số đã chốt, tệp đầu ra đã bàn giao, mốc/tham số chuyển tiếp sang bước sau, kèm `### Câu hỏi nghi vấn chuyển sang Bước XX` và dòng liên kết `**Tiếp theo:** [...]`.

---

## 3. Lưu ý kỹ thuật bắt buộc
* Cố định `SEED = 42` và chạy trong môi trường `.venv` (`Python 3.12`, `TensorFlow / Keras`).
* **Hiển thị bảng `DataFrame` bằng `display(df)`** (đã có sẵn trong Jupyter/IPython, không cần import thừa) hoặc đặt `df` ở cuối cell để hiển thị bảng HTML trực quan; khi chạy kiểm tra qua CLI thì truyền `IPython.display.display` vào namespace `exec()`.
* Mọi con số ghi trong Markdown và báo cáo `.md` bắt buộc phải khớp `100%` với `stdout` sau lần chạy cuối cùng (chi tiết xem [`rules.md`](./rules.md)).
