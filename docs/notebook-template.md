# Quy chuẩn Template cho Jupyter Notebook

Tài liệu này gợi ý cấu trúc gọn nhẹ, chuẩn mực cho các notebook và báo cáo trong dự án. Trước khi biên soạn hoặc chỉnh sửa notebook mới, **bắt buộc đọc và tuân thủ toàn bộ quy tắc tại [`rules.md`](./rules.md)**.

---

## 1. Quy ước đặt tên file & Quản lý thư mục

| Số thứ tự | Tệp Notebook (`notebooks/`) | Tệp Báo cáo (`docs/`) | Nội dung chính |
| :---: | :--- | :--- | :--- |
| **00** | `00_Notebook_Template.ipynb` | — | File mẫu chuẩn hóa cấu trúc dùng để nhân bản khi tạo notebook mới. |
| **01** | `01_Overview.ipynb` | [`01_Overview_Report.md`](./01_Overview_Report.md) | Tổng quan đề tài, kiểm toán 4 bộ dữ liệu gốc và chọn `PEMS08` (Primary) / `PEMS04` (Held-out Transfer). |
| **02** | `02_Data_Cleaning_and_Interpolation.ipynb` | [`02_Data_Cleaning_Report.md`](./02_Data_Cleaning_Report.md) | Chẩn đoán lỗi cảm biến $S_1 \dots S_4$ và hồi quy cặp lân cận thuần **Chen et al. (2003)**. |
| **03** | `03_EDA_and_Bottleneck_Analysis.ipynb` | [`03_EDA_and_Bottleneck_Report.md`](./03_EDA_and_Bottleneck_Report.md) | **RQ1 (thuần `PEMS08`):** Khám phá đa biến, công thức tự tính ngưỡng (FHWA `<60%` FFS + Biểu đồ cơ bản) & loại trừ `Flow`, kiểm chứng `7` chỉ báo tiền tắc nghẽn (khử dốc + liên kết đa biến + độ dốc có dấu), và đối chiếu đồ thị tĩnh vs. động (`In + Out`). |
| **04** | `04_Modeling_and_Feature_Contribution.ipynb` | [`04_Modeling_Report.md`](./04_Modeling_Report.md) | **RQ2 (`PyTorch`, `PEMS08`):** Đóng băng `1` kiến trúc `ST-GAT (LSTM → 2-layer GATv2)` đối chứng `6` cấu hình đặc trưng (`Raw 3` → `Base 7` → `+G1/+G2/+G3` → `Full 14`) theo trọng tâm `MAE`, `Permutation Feature Importance`, chú ý cạnh `GATv2` & đánh giá phụ tắc nghẽn (ngưỡng thuần `Train`). |
| **05** | `05_Cross_Map_Validation_PEMS04.ipynb` | [`05_Cross_Map_Validation_Report.md`](./05_Cross_Map_Validation_Report.md) | **Kiểm chứng đối xứng (`PEMS04`) & Kết luận:** Tự tính ngưỡng thuần `Train`, lặp lại đầy đủ `6` cấu hình đặc trưng trên cùng kiến trúc `ST-GAT` (huấn luyện độc lập & *Zero-Shot*), xác nhận nhất quán `PFI` (`ρ = 0,9956`), chú ý `GATv2` và kết luận toàn đề tài. |

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
* Cố định `SEED = 42` và chạy trong môi trường `.venv` (`Python 3.12`, `PyTorch`).
* **Hiển thị bảng `DataFrame` bằng `display(df)`** (đã có sẵn trong Jupyter/IPython, không cần import thừa) hoặc đặt `df` ở cuối cell để hiển thị bảng HTML trực quan; khi chạy kiểm tra qua CLI thì truyền `IPython.display.display` vào namespace `exec()`.
* Mọi con số ghi trong Markdown và báo cáo `.md` bắt buộc phải khớp `100%` với `stdout` sau lần chạy cuối cùng (chi tiết xem [`rules.md`](./rules.md)).
