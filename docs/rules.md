# Quy chuẩn & Kinh nghiệm Vận hành Dự án (`docs/rules.md`)

Tài liệu này tổng hợp toàn bộ các quy tắc bắt buộc được đúc kết từ thực tế triển khai các Notebook `01`, `02`, `03` của dự án. Mọi phiên làm việc và trợ lý AI kế tiếp **phải tuân thủ nghiêm ngặt** các quy tắc dưới đây để đảm bảo tính khoa học, mạch lạc và chặt chẽ từ khâu làm sạch dữ liệu tới mô hình hóa và viết báo cáo.

---

## 1. Kỷ luật Bằng chứng, Số liệu & Ngưỡng Tham số

1. **Không dùng con số tùy tiện / không có cơ sở dữ liệu:**
   * Tuyệt đối không đặt các ngưỡng cắt theo cảm tính nếu chưa chứng minh bằng phân phối thực tế hoặc quy ước chuẩn có căn cứ (bài học từ Notebook `03`: loại bỏ mốc tham chiếu `5%` tự đặt, thay bằng **`Trung bình toàn mạng (1.84%)`** tính trực tiếp từ dữ liệu).
   * Với mọi tham số hoặc ngưỡng quy ước/thực nghiệm (như phân vị `P1`, `P95`, `40 mph`, `Occupancy ≥ 0.15`, `Top 20`), bắt buộc phải:
     * **Chứng minh nguồn gốc và kiểm định trên dữ liệu trước khi sử dụng** làm tiêu chí chuẩn;
     * **Nhắc rõ cho người dùng biết** nếu một tham số mang tính lựa chọn/quy ước thực nghiệm hoặc ngưỡng ngẫu nhiên dùng trong đối chứng (ví dụ nhóm trạm ngẫu nhiên ở Mục `6.2`, `Top 20` theo đường cong Pareto).

2. **Chốt đồng thời các biến liên quan — Flow một chiều, không lặp lại so sánh thừa:**
   * Khi hiện tượng cần nhận diện từ nhiều biến đo lường (`Lưu lượng`, `Độ chiếm dụng`, `Vận tốc`), phải **xét đồng thời cả 3 biến để chốt định nghĩa chuẩn ngay từ đầu** (ví dụ Mục `3.1` chốt `(Vận tốc < 40 mph) & (Độ chiếm dụng ≥ 0.15)`), giải thích rõ vai trò của từng biến (kể cả lý do biến phi đơn điệu như `Lưu lượng` không cắt ngưỡng cứng).
   * Từ các mục phía sau định nghĩa chuẩn đó trở đi (ví dụ Mục `3.2`, `4`, `5`, `6`, `7`), **áp dụng nhất quán duy nhất tiêu chí vừa chốt**, tuyệt đối không quay lại so sánh rườm rà với ngưỡng đơn cũ đã bị loại bỏ.

3. **Tách rõ quần thể khi tính tham số:**
   * Không gộp lẫn nhóm trạm sạch và nhóm trạm lỗi khi tính phân vị nền. Mọi tham số thực nghiệm phải ghi rõ **lấy trên tập con nào** (ví dụ NB02: `120/170` trạm không có `<= 0` dùng tính `P1_day = 69.0`, `P1_night = 16.0`; `50/170` trạm có `0.0` dùng tính `P95 = 6` bước cho độ dài chuỗi đêm).

4. **Khớp 100% số liệu thực tế & Trung thực với giới hạn thống kê:**
   * Mọi con số đưa vào Markdown, biểu đồ hoặc báo cáo phải lấy **trực tiếp từ `stdout` lần chạy gần nhất** sau lần chỉnh sửa cuối cùng — không ước lượng, không giữ số cũ từ lần chạy trước khi đổi định nghĩa.
   * Không phóng đại kết quả (*no overclaiming*): nêu rõ giới hạn khi một kiểm định không có ý nghĩa (`p > 0.05` như `Betweenness` hoặc không lặp lại trên `PEMS04` như `Closeness`). Khi hệ số `Pearson r` bị nén toán học do sự kiện hiếm (tỷ lệ nền `1.56%–2.34%`), bổ sung thêm thước đo phân tách xác suất có điều kiện (`Risk Ratio`).

5. **Chốt dứt khoát tham số & mốc bàn giao ở cuối mỗi bước:**
   * Cuối mỗi Notebook phải tổng kết rõ các tham số đã chốt, định nghĩa nhãn và mốc thời gian bàn giao sang bước kế tiếp (ví dụ NB03 chốt cửa sổ trọng tâm `10–20 phút`, mốc chuẩn `15 phút`, dải kiểm thử đầy đủ $\Delta t \in \{5, 10, 15, 20, 30\}$ phút).

---

## 2. Văn phong Kể chuyện Khoa học & Tư duy Mở

1. **Văn phong báo cáo khoa học bằng tiếng Việt:**
   * Khách quan, súc tích, chuẩn học thuật; **tuyệt đối không sử dụng emoji hay biểu tượng trang trí**.
   * Khi lần đầu xuất hiện một khái niệm/thước đo thống kê (`P1`, `P5`, `P95`, `Trung vị`, `Step-diff Std`, `Spearman ρ`, `Risk Ratio`), phải **diễn giải trực quan bằng ngôn ngữ giao thông dễ hiểu** ngay trước hoặc ngay cạnh bảng/biểu đồ.

2. **Mạch kể chuyện Hỏi – Đáp liên kết chặt chẽ (`Question-Driven`):**
   * Đầu mỗi mục nêu rõ **câu hỏi cần trả lời** từ bước trước.
   * Phần mã nguồn tính bảng/vẽ biểu đồ minh họa giải quyết câu hỏi đó.
   * Cuối khối `> **Nhận xét (...):**` luôn có tiểu mục **`> **Câu hỏi nghi vấn tiếp theo:**`** làm cầu nối logic dẫn sang mục tiếp theo hoặc notebook kế tiếp.

3. **Không cài cứng (hardcode) kết luận tương lai vào các bước đầu:**
   * Hai câu hỏi nghiên cứu lớn (`RQ1`, `RQ2`) chỉ giới thiệu ở `01_Overview.ipynb` và kết luận ở `06_Research_Synthesis_and_Conclusions.ipynb`.
   * Các notebook khám phá và tiền xử lý (`01–03`) **không nêu trước tên kiến trúc cụ thể** của bước mô hình hóa hay giải thích mô hình (giữ câu hỏi mở để đối chiếu công bằng nhiều hướng mô hình ở `04`).

---

## 3. Quy chuẩn Trình bày Thông thoáng (Markdown & Biểu đồ)

1. **Định dạng khối `Nhận xét`:**
   * Tiêu đề `> **Nhận xét (...):**` phải đứng riêng một dòng, ngay sau đó **bắt buộc có 1 dòng `>` trống** rồi mới xuống nội dung (tránh viết dính liền trên cùng dòng tiêu đề).
   * Giữa các gạch đầu dòng dài trong khối blockquote nên có 1 dòng `>` trống để thoáng mắt.

2. **Danh sách & Khoảng cách:**
   * Danh sách đánh số (`1.`, `2.`, `3.`, ...) mỗi ý đứng một dòng riêng, có dòng trống ngăn cách giữa các đoạn dài.
   * Công thức toán LaTeX viết chuẩn (`bad_latex == []`), dùng dấu backtick `` ` `` cho tên biến, chỉ số và đường dẫn file.

3. **Biểu đồ minh họa:**
   * Đặt xen kẽ trực tiếp ngay tại mục phân tích tương ứng, lưu dưới dạng PNG `300 DPI` tại `outputs/figures/nbXX_*.png`.
   * Sau khi chạy xuất biểu đồ, kiểm tra trực quan để chắc chắn nhãn chữ, chú giải (`legend`) và tiêu đề không đè lên dữ liệu hoặc bị cắt mép.

---

## 4. Kỷ luật Kỹ thuật, Môi trường & Kiểm chứng

1. **Ngăn xếp công nghệ thống nhất:**
   * Môi trường ảo `.venv` (`Python 3.12`), sử dụng **`TensorFlow / Keras`** cho các mô hình học sâu (không dùng PyTorch), kết hợp `NumPy`, `Pandas`, `SciPy`, `Scikit-learn`, `NetworkX`, `Matplotlib`, `Seaborn`, `SHAP`, và cố định `SEED = 42`.

2. **Mã nguồn Notebook sạch, bảng HTML đẹp & Chạy được qua CLI:**
   * Giữ code gọn nhẹ, rõ ràng.
   * **Hiển thị bảng `DataFrame` bằng `display(df)`** (hàm có sẵn trong Jupyter/IPython, không cần import thêm) hoặc đặt `df` ở cuối cell để bảng render HTML đẹp mắt trên notebook.
   * Khi chạy kiểm chứng tự động qua dòng lệnh Python thuần (`exec()`), script kiểm tra sẽ tự truyền `IPython.display.display` vào namespace thực thi để đảm bảo không lỗi `NameError`.

3. **Quy trình tự kiểm chứng bắt buộc sau lần sửa cuối:**
   * Chạy toàn bộ cell code của notebook bằng `.venv\Scripts\python.exe` với `matplotlib.use('Agg')` (kèm `display` trong namespace).
   * Kiểm tra tự động: tất cả liên kết nội bộ hợp lệ (`links_ok == True`), không lỗi LaTeX (`bad_latex == []`), không `Nhận xét:` dính dòng (`inline_nx == 0`), không lọt từ khóa cài cứng ở bước chưa tới (`locks == []`).

---

## 5. Tổ chức Dữ liệu & Quy trình Làm việc Từng Bước với Người dùng

1. **Quản lý dữ liệu không đẩy lên GitHub:**
   * Toàn bộ dữ liệu thô (`datasets/data/`) và dữ liệu đã làm sạch (`datasets/cleaned/`) phải đặt trong thư mục `datasets/` để `.gitignore` tự động bỏ qua, tránh đưa tệp nhị phân `.npz` dung lượng lớn lên repository.
   * Các tệp trung gian nhẹ dùng cho báo cáo và đối chiếu (`outputs/figures/*.png`, `outputs/*.csv`, `outputs/*.gexf`) và báo cáo Markdown (`docs/*.md`) được quản lý bằng Git.

2. **Làm chắc từng bước — Duyệt xong mới Commit:**
   * Luôn giải thích kỹ cơ sở dữ liệu khi người dùng đặt câu hỏi phản biện; nếu phát hiện lỗ hổng lập luận thì chỉnh sửa đồng bộ từ trên xuống dưới rồi mới báo cáo lại.
   * **Không tự ý `git commit` / `git push`** trước khi người dùng đọc lại, kiểm tra xong notebook hoặc báo cáo hiện tại; khi commit chỉ stage đúng các tệp thuộc phạm vi bước vừa chốt.
