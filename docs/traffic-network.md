# TỔNG HỢP ĐỀ TÀI NGHIÊN CỨU CUỐI KỲ

**Môn học:** Data Visualization & Analytics (Định dạng bài báo chuẩn IEEE, tối đa 3.000 từ | Thuyết trình 10 phút)

**Sinh viên thực hiện:** Nguyễn Trọng Đại (MSSV: 11247268 – Lớp: A166A)

---

## 1. Thông tin chung về Đề tài

* **Tên đề tài (Tiếng Anh):** *Spatio-Temporal Anatomy of Urban Traffic Bottlenecks: Early-Warning Signals and Proactive Rerouting Simulation via Graph Attention Networks*
* **Tên đề tài (Tiếng Việt):** *Giải phẫu Không gian - Thời gian các Nút thắt Giao thông Đô thị: Tín hiệu Cảnh báo sớm và Thử nghiệm Điều tiết chủ động qua Mạng Nơ-ron Đồ thị*
* **Chủ đề cốt lõi:** Phân tích cấu trúc mạng lưới giao thông để nhận diện các điểm nghẽn trọng yếu, khám phá dấu hiệu cảnh báo sớm trước khi tắc nghẽn xảy ra thông qua đồ thị lân cận, và chạy thử nghiệm giả lập điều tiết lưu lượng đầu nguồn để đánh giá mức độ giảm ùn tắc.

---

## 2. Bộ dữ liệu sử dụng (Datasets)

* **Nguồn công khai:** `https://github.com/divanoresia/Traffic` (Dữ liệu cảm biến giao thông PeMS – Caltrans).
* **Datasets chính (Kaggle):** [PEMS Datasets](https://www.kaggle.com/datasets/elmahy/pems-dataset)
* **Cấu trúc chung:** Dữ liệu ghi nhận định kỳ **5 phút/lần**, gồm 3 biến số thực tế tại mỗi đỉnh: Lưu lượng xe (`Flow`), Độ chiếm dụng mặt đường (`Occupancy`), và Vận tốc trung bình (`Speed`), kèm theo sơ đồ kết nối có hướng giữa các trạm (`.csv`). Không có giá trị khuyết (`NaN = 0`).

| Vai trò trong nghiên cứu | Tên bộ dữ liệu | Khu vực địa lý | Số đỉnh (Nodes) | Số cạnh (Edges) | Kích thước chuỗi thời gian | Đặc điểm cấu trúc mạng |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **Chặng 1: Nghiên cứu chính (Primary)** | **PeMS08** | Khu vực San Bernardino | **170** | **295** *(277 duy nhất)* | `(17856, 170, 3)` *(62 ngày)* | Mạng lưới dày hơn (bậc TB `3.47`), dữ liệu gốc sạch hơn (`0.348%` giá trị `<= 0` ở Lưu lượng). |
| **Chặng 2: Kiểm chứng cuối (Validation)** | **PeMS04** | Vùng vịnh San Francisco | **307** | **340** | `(16992, 307, 3)` *(59 ngày)* | Mạng lưới thưa hơn (bậc TB `2.21`), dạng hành lang cao tốc kéo dài, `1.590%` giá trị `<= 0`. |

> **Ghi chú phân vai (từ kiểm toán ở `01_Overview.ipynb`):** Đảo ngược với dự kiến sơ bộ ban đầu, kiểm toán thực nghiệm cho thấy `PeMS08` có chuỗi dài hơn, đồ thị liên kết dày đặc hơn và tỷ lệ giá trị `0.0` thấp hơn gần 4.6 lần so với `PeMS04`, nên được chọn làm tập khám phá & huấn luyện chính; `PeMS04` đóng vai trò tập kiểm chứng chéo độc lập.

---

## 3. Hệ thống Câu hỏi Nghiên cứu (2 RQs tinh gọn)

### RQ1: Chân dung điểm nghẽn & Tín hiệu cảnh báo sớm (Node-level Early Warning)
* **Câu hỏi (Tiếng Anh):** *What network centrality traits characterize major traffic bottlenecks, and what early-warning signals appear prior to congestion?*
* **Giải nghĩa (Tiếng Việt):** Những đặc điểm trung tâm mạng lưới nào định hình nên các nút thắt tắc nghẽn chính, và những dấu hiệu cảnh báo sớm nào xuất hiện trước khi kẹt xe xảy ra?
* **Cách thức triển khai:**
  * Đối chiếu chỉ số cấu trúc mạng (**Weighted Betweenness / Closeness Centrality**) với tần suất sụt giảm vận tốc thực tế (ví dụ $< 40\text{ mph}$ so với vận tốc trung bình toàn mạng $\sim 63,5\text{ mph}$) để chọn ra **Top các Đỉnh nút thắt (Bottleneck Nodes)**.
  * Huấn luyện mô hình **Graph Attention Network (GAT)** để dự đoán trạng thái tắc nghẽn tại các đỉnh nút thắt qua nhiều mốc thời gian $\Delta t \in \{5, 10, 15, 20, 30, 45, 60\}$ phút.
  * **Chứng minh mốc thời gian tối ưu bằng thực nghiệm:** So sánh hiệu năng giữa mô hình **GAT (có thông tin mạng lưới)** và **Mô hình cục bộ (Baseline — chỉ nhìn dữ liệu tại chính trạm đó)**. Mốc thời gian tối ưu là điểm mà GAT vừa duy trì độ chính xác F1-Score cao, vừa tạo ra khoảng cách vượt trội (*Network Gain*) lớn nhất so với mô hình cục bộ.
  * Sử dụng **SHAP** và **Attention Weights ($\alpha_{ij}$)** trên các cạnh đầu vào để chỉ rõ dấu hiệu tiền tắc nghẽn (ví dụ: chỉ số `Occupancy` tại các trạm đầu nguồn tăng vọt trước khi `Speed` tại điểm nghẽn sụt giảm).

### RQ2: Thử nghiệm giả lập điều tiết lưu lượng chủ động (Counterfactual Rerouting Simulation)
* **Câu hỏi (Tiếng Anh):** *In a GNN-based simulation, how effectively can reducing upstream traffic flow upon early-warning signals mitigate congestion at these bottlenecks?*
* **Giải nghĩa (Tiếng Việt):** Trong kịch bản giả lập dựa trên mô hình GNN, việc chủ động giảm bớt lưu lượng xe đầu nguồn ngay khi có tín hiệu cảnh báo sớm giúp giảm thiểu tình trạng tắc nghẽn tại các điểm nghẽn hiệu quả đến mức nào?
* **Cách thức triển khai:**
  * Sử dụng chính mô hình GAT đã huấn luyện xong ở RQ1 làm môi trường giả lập phản thực tế (đóng băng trọng số).
  * Từ file `.csv`, xác định các **trạm đầu nguồn (upstream neighbors)** có cạnh nối trực tiếp đổ xe vào Top điểm nghẽn.
  * Tại các thời điểm phát hiện tín hiệu cảnh báo sớm, tiến hành giảm thử nghiệm lưu lượng xe (`Flow` và `Occupancy` tương ứng) tại các trạm đầu nguồn đi các mức $\alpha \in \{10\%, 20\%, 30\%\}$.
  * Cho mô hình GAT chạy dự đoán lại để đo lường **tỷ lệ sự cố tắc nghẽn được ngăn chặn thành công (%)** và mức giảm của trọng số áp lực trên cạnh ($\alpha_{ij}$).

---

## 4. Nhóm Kỹ thuật & Hệ thống Trực quan hóa

### Phương pháp luận (Method Families)
1. **Family F – Social & Network Analytics (Trọng tâm):**
   * *Graph Centrality:* Weighted Betweenness Centrality, Closeness Centrality, Degree Centrality.
   * *Deep Learning trên đồ thị:* **Graph Attention Network (GAT)** dự đoán ở cấp độ đỉnh (*Node-level classification*), tận dụng cấu trúc cạnh để tổng hợp tín hiệu từ các đỉnh lân cận.
2. **Family I – Explainable, Reliable & Responsible AI (Bổ trợ):**
   * Trích xuất **Edge Attention Weights ($\alpha_{ij}$)** để xác định tuyến đường đầu nguồn nào đang gây áp lực lớn nhất lên điểm nghẽn.
   * Sử dụng **SHAP (SHapley Additive exPlanations)** để giải thích các chỉ số cảnh báo sớm và làm cơ sở cho thí nghiệm giảm tải lưu lượng ở RQ2.

### Danh mục Biểu đồ Trực quan hóa (Chiếm 30% trọng số điểm)
1. **Bản đồ cấu trúc mạng lưới (Gephi – bố cục *ForceAtlas2*):** Hiển thị 170 trạm PeMS08, kích thước đỉnh tỷ lệ thuận với tần suất tắc nghẽn, màu sắc thể hiện chỉ số *Betweenness Centrality* để làm nổi bật các nút thắt cổ chai.
2. **Biểu đồ quỹ đạo pha (Phase Trajectory Plot):** Biểu diễn mối quan hệ `Flow – Occupancy – Speed` trong cửa sổ thời gian trước khi kẹt xe, cho thấy hiện tượng bão hòa mặt đường xảy ra trước khi vận tốc sụt giảm.
3. **Biểu đồ độ nhạy thời gian & Network Gain (Line Chart):** Đối chiếu đường cong F1-Score của **GAT** vs. **Mô hình cục bộ** qua các mốc $\Delta t$ từ 5 đến 60 phút để chứng minh bằng số liệu khoảng thời gian cảnh báo sớm hiệu quả nhất.
4. **Biểu đồ giải thích mô hình (SHAP Summary & Attention Graph):** Trực quan hóa mức độ đóng góp của các biến số từ trạm hàng xóm vào quyết định cảnh báo sớm.
5. **Biểu đồ đánh giá thử nghiệm điều tiết (Sensitivity Curve & Before–After Comparison):** So sánh xác suất tắc nghẽn trước và sau khi giảm lưu lượng đầu nguồn ($10\%, 20\%, 30\%$).
6. **Bảng đối chiếu kiểm chứng chéo (PeMS08 vs. PeMS04):** Tổng hợp kết quả đánh giá mô hình và hiệu quả điều tiết trên PeMS04 ở cuối phần *Results*.

---

## 5. Kịch bản Thuyết trình 10 Phút & Lộ trình Thực hiện

### Phân bổ thời gian thuyết trình 10 phút (Chuẩn Rubric)
* **01:30 — Vấn đề & 2 Câu hỏi nghiên cứu:** Đặt vấn đề kẹt xe dây chuyền tại các nút thắt và giới thiệu mục tiêu RQ1 (bắt dấu hiệu sớm) + RQ2 (thử nghiệm giảm tải).
* **02:00 — Giải thích kỹ thuật trực quan (25% điểm):** Dùng sơ đồ minh họa 5 đỉnh (1 đỉnh trung tâm + 4 đỉnh lân cận) để giải thích dễ hiểu cách *Centrality* tìm điểm nghẽn và cách *GAT + SHAP* gom tín hiệu từ hàng xóm để cảnh báo sớm.
* **03:00 — Bằng chứng RQ1 (Trên PeMS08):** Trình bày bản đồ nút nghẽn, biểu đồ quỹ đạo pha, biểu đồ *Network Gain* chọn mốc thời gian cảnh báo tối ưu, và biểu đồ SHAP.
* **02:30 — Bằng chứng RQ2 & Kiểm chứng cuối trên PeMS04:** Trình bày biểu đồ hiệu quả giảm tắc nghẽn khi điều tiết lưu lượng đầu nguồn trên PeMS08, chốt lại bằng bảng kiểm chứng chéo trên PeMS04.
* **01:00 — Kết luận & Hạn chế:** Tóm tắt phát hiện chính và nêu rõ giới hạn mô phỏng (chưa tính đến tải trọng phát sinh trên các tuyến đường vòng khi phân luồng).

### Lộ trình thực hiện theo các Checkpoint
* **Checkpoint 1 (Proposal):** Hoàn tất định hình đề tài, 2 RQs tinh gọn, kiểm tra toàn vẹn dữ liệu PeMS08 & PeMS04 và gửi Google Form.
* **Checkpoint 2 (Data & EDA trên PeMS08):** Kiểm tra và xử lý các chuỗi giá trị `0.0` lỗi cảm biến, tính toán *Centrality*, xác định Top nút thắt cổ chai và vẽ biểu đồ khám phá.
* **Checkpoint 3 (Modeling & Simulation):** Huấn luyện GAT và mô hình Baseline trên PeMS08 qua các mốc 5–60 phút (RQ1), trích xuất SHAP, chạy thử nghiệm giảm lưu lượng đầu nguồn (RQ2), sau đó chạy đánh giá kiểm chứng toàn bộ trên PeMS04.
* **Checkpoint 4 (Final Paper & Defence):** Hoàn thiện bài báo chuẩn IEEE ($\le 3.000$ từ) và slide thuyết trình 10 phút.