# Tổng hợp Văn học Khoa học về Bộ dữ liệu Caltrans PeMS (`docs/literature_review.md`)

Tài liệu này lưu trữ có hệ thống các bài báo học thuật nền tảng và gần đây xoay quanh hệ sinh thái dữ liệu **Caltrans PeMS** (`PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`, `PEMS-BAY`, `PeMSD7`, `LargeST`), làm cơ sở tham khảo phương pháp, trích dẫn chuẩn IEEE và định vị khoảng trống nghiên cứu cho dự án.

---

## 1. Phân loại Hệ sinh thái Bộ dữ liệu PeMS

| Bộ dữ liệu | Bài báo chuẩn hóa | Khu vực (California) | Số trạm ($N$) | Số biến ($C$) | Khung thời gian | Đặc điểm sử dụng trong văn học |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **`PEMS08`** | **ASTGCN** (Guo et al., AAAI 2019) | District 8 — San Bernardino | 170 | **3** (`Flow, Occ, Speed`) | 07–08/2016 (62 ngày) | Có đủ 3 biến vật lý; đa số bài báo chỉ trích 1 kênh `Flow`. |
| **`PEMS04`** | **ASTGCN** (Guo et al., AAAI 2019) | District 4 — SF Bay Area | 307 | **3** (`Flow, Occ, Speed`) | 01–02/2018 (59 ngày) | Có đủ 3 biến vật lý; đồ thị thưa hơn, dạng hành lang kéo dài. |
| **`PEMS03`** | **STSGCN** (Song et al., AAAI 2020) | District 3 — North Central | 358 | 1 (`Flow`) | 09–11/2018 (91 ngày) | Chỉ có 1 biến lưu lượng, không dựng được Biểu đồ cơ bản. |
| **`PEMS07`** | **STSGCN** (Song et al., AAAI 2020) | District 7 — Los Angeles | 883 | 1 (`Flow`) | 05–08/2017 (98 ngày) | Chỉ có 1 biến lưu lượng. |
| **`PEMS-BAY`** | **DCRNN** (Li et al., ICLR 2018) | District 4 — Bay Area | 325 | 1 (`Speed`) | 01–05/2017 (6 tháng) | Chuẩn dự báo vận tốc và điền khuyết (`GRIN`). |
| **`PeMSD7(M/L)`** | **STGCN** (Yu et al., IJCAI 2018) | District 7 — Los Angeles | 228 / 1.026 | 1 (`Speed`) | 05–06/2012 (ngày làm việc) | Chuẩn dự báo vận tốc đời đầu. |
| **`LargeST`** | **LargeST** (Liu et al., NeurIPS 2023) | Toàn bang California (`SD/GBA/GLA/CA`) | 716 – 8.600 | 1 (`Flow` + metadata) | 2017 – 2021 (5 năm) | Kiểm thử khả năng mở rộng quy mô lớn của STGNN. |

### Nguồn gốc tiền xử lý của các tệp `PEMS04.npz` và `PEMS08.npz` công khai
1. **Tầng Caltrans PeMS (Chen et al., 2001, 2003):** Gộp dữ liệu thô 30 giây thành bước 5 phút (`288` bước/ngày), đồng thời hệ thống PeMS ước lượng/điền sẵn kênh `Speed` (khiến kênh `Speed` không có `0.0` và không có `NaN`).
2. **Tầng chuẩn hóa ASTGCN / STSGCN (Guo et al., 2019; Song et al., 2020):** Cắt lát ~2 tháng và dùng nội suy tuyến tính cho các ô `NaN` tường minh (khiến tệp `.npz` công khai có `NaN = 0`), **nhưng vẫn tồn đọng nhiều chuỗi giá trị `0.0` kéo dài hàng chục giờ** ở `Flow` và `Occupancy` (cùng `18` dòng cạnh trùng lặp trong `PEMS08.csv`).

---

## 2. Hệ thống hóa Chi tiết 7 Nhóm Bài báo Liên quan

### Nhóm 1: Hệ thống PeMS, Phát hiện Lỗi Cảm biến & Điền khuyết Dữ liệu

1. **Chen et al. (TRR 2001) — *Freeway Performance Measurement System: Mining Loop Detector Data***
   * **Bài toán & Dữ liệu:** Xây dựng hệ thống thu thập, tổng hợp 5 phút và tính độ trễ giao thông (*Vehicle Hours of Delay*) từ cảm biến vòng từ đơn toàn bang California.
   * **Vai trò trích dẫn:** Nguồn gốc chính thức của bộ dữ liệu Caltrans PeMS.

2. **Chen et al. (TRR 2003) — *Detecting Errors and Imputing Missing Data for Single-Loop Surveillance Systems***
   * **Bài toán & Dữ liệu:** Phát hiện vòng từ bị hỏng (kẹt `0`, kẹt hằng số, lệch xung) và điền khuyết trên ~15.000 cảm biến PeMS (`Volume`, `Occupancy`).
   * **Thuật toán 2 bước:**
     * *Bước 1 — Daily Statistics Algorithm (DSA):* Tính 4 chỉ số chẩn đoán trên trọn chuỗi 24 giờ của từng trạm $i$:
       * $S_1$: Số mẫu trong ngày có $\text{Occupancy} = 0$ (bắt lỗi tắt cứng).
       * $S_2$: Số mẫu có $\text{Occupancy} > 0$ nhưng $\text{Flow} = 0$ (bắt lỗi kẹt xung đếm).
       * $S_3$: Số mẫu có độ chiếm dụng cao liên tục bất thường (bắt lỗi bật cứng).
       * $S_4 = -\sum p(x)\log p(x)$: Entropy của phân phối `Occupancy` trong ngày (cảm biến kẹt hằng số sẽ có Entropy tiến về `0`).
       * Xác định ngưỡng cắt $s^*$ từ phân phối hai đỉnh thực nghiệm (*bimodal histogram*) để gắn nhãn ngày/cảm biến lỗi.
     * *Bước 2 — Điền khuyết bằng hồi quy tuyến tính cặp trạm lân cận:* Khớp quan hệ $\hat{x}_i(t) = \hat{\alpha}_{ij} x_j(t) + \hat{\beta}_{ij}$ trên dữ liệu lịch sử sạch từ các trạm lân cận $j$ đang hoạt động tốt.
   * **Giới hạn:** Quan hệ tuyến tính tĩnh; gắn cờ cả khối 24 giờ khá thô; chưa tách biệt xử lý đoạn mất ngắn so với đoạn mất dài.

3. **GRIN — Cini, Marisca & Alippi (ICLR 2022) — *Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks***
   * **Bài toán & Dữ liệu:** Điền khuyết chuỗi thời gian đa biến có đoạn mất ngẫu nhiên (*point missing*) và đoạn mất dài liên tiếp (*block missing*) trên `PEMS-BAY` (325 nút `Speed`), `METR-LA`, `AQI`.
   * **Kiến trúc cốt lõi:**
     * **MPGRU (Message Passing GRU) hai chiều:** Thay phép nhân ma trận trong GRU bằng tích chập khuếch tán đồ thị (*diffusion convolution*, $K=2$), chạy đồng thời theo chiều xuôi ($1 \to T$) và chiều ngược ($T \to 1$), đầu vào là dữ liệu kèm mặt nạ nhị phân $\mathbf{M}$ đánh dấu vị trí bị thiếu.
     * **Điền khuyết 2 giai đoạn + Spatial Decoder:** Giai đoạn 1 dự đoán sơ bộ từ trạng thái ẩn thời gian; Giai đoạn 2 tinh chỉnh qua bộ giải mã đồ thị không gian rồi phản hồi lại vào vòng lặp GRU.
   * **So sánh với Chen et al. (2003):** GRIN vượt trội trên các đoạn mất dài phi tuyến nhờ nhìn cả quá khứ, tương lai và đồ thị lân cận, **nhưng không tự phát hiện được số `0` nào là lỗi** (cần mặt nạ $\mathbf{M}$ đầu vào) và nặng tính toán hơn. Do đó, **kết hợp chẩn đoán lỗi của Chen et al. (2003) tạo mặt nạ $\mathbf{M}$ + điền khuyết không gian–thời gian kiểu GRIN (hoặc đối chiếu với hồi quy Chen 2003)** là quy trình bổ trợ trực tiếp cho nhau.

---

### Nhóm 2: Vật lý Dòng xe, Định nghĩa Tắc nghẽn, Điểm nghẽn & Chuyển pha

4. **Cassidy & Bertini (TRB 1999) — *Some Traffic Features at Freeway Bottlenecks***
   * **Phương pháp & Kết quả:** Dùng đường cong tích lũy xe nghiêng (*oblique cumulative curves*) tại các nút thắt cao tốc Toronto; chứng minh hiệu ứng **Sụt giảm Công suất (*Capacity Drop*)**: khi hàng đợi hình thành, lưu lượng xả (*queue discharge flow*) tụt vĩnh viễn khoảng **`8%–10%`** so với lưu lượng tối đa trước khi kẹt.
   * **Giá trị áp dụng:** Giải thích trực tiếp vì sao biến `Flow` có tính **đa trị (phi đơn điệu)** trên Biểu đồ cơ bản — không thể dùng ngưỡng cứng theo `Flow` để phân loại tắc nghẽn; đồng thời chứng minh ý nghĩa của việc ngăn tắc nghẽn trước điểm gãy pha.

5. **Choe, Skabardonis & Varaiya (TRB 2002) — *Freeway Performance Measurement System (PeMS): An Operational Analysis Tool* (kết hợp *Highway Capacity Manual — HCM*)**
   * **Phương pháp & Kết quả:** Trên dữ liệu PeMS 30s và 5 phút, chuẩn hóa việc dùng **`Occupancy`** (tỷ lệ thời gian xe phủ vòng từ) thay cho mật độ không gian (`Density`) trong HCM để dựng Biểu đồ cơ bản `Flow – Occupancy – Speed` và phân định Mức phục vụ (LOS): vùng `Occupancy ~ 11%–15%` đánh dấu ranh giới chớm bão hòa/tiền gãy pha; `Occupancy ≥ 30%` biểu thị tắc cứng quá tải (*hypercongestion*).

6. **Chen, Skabardonis & Varaiya (TRR 2004) — *Systematic Identification of Freeway Bottlenecks***
   * **Phương pháp & Công thức lõi trên dữ liệu 5 phút PeMS:** Xác định **điểm nghẽn gốc kích hoạt** giữa trạm thượng lưu $x_u$ và trạm hạ lưu liền kề $x_d$ tại thời điểm $t$ bằng 3 điều kiện đồng thời:
     1. Vận tốc thượng lưu rơi vào vùng tắc: $v(x_u, t) < 40\text{ mph}$.
     2. Vận tốc hạ lưu phục hồi rõ rệt (tách điểm nghẽn gốc khỏi vùng bị kẹt lan ngược): $v(x_d, t) - v(x_u, t) \ge 20\text{ mph}$.
     3. **Tính duy trì liên tục (*Persistence*):** Điều kiện duy trì liên tiếp qua nhiều bước 5 phút (ví dụ $\ge 5/7$ bước, tức `25–35 phút`) để lọc nhiễu tức thời và phân biệt điểm nghẽn lặp lại (*recurrent*).
   * **Kết hợp với Choe (2002):** Bổ sung ràng buộc `Occupancy ≥ 0.15` giúp loại bỏ triệt để các điểm xe chạy chậm dưới `40 mph` khi đường vắng.

7. **Li et al. (PNAS 2015) — *Percolation Transition in Dynamical Traffic Network with Evolving Critical Bottlenecks***
   * **Bài toán & Dữ liệu:** Phân tích sự sụp đổ giao thông toàn mạng bằng dữ liệu GPS taxi trên mạng đường phố đô thị Bắc Kinh.
   * **Cách chứng minh:**
     * Phân loại các đoạn đường là "thông" hay "tắc" theo ngưỡng vận tốc tương đối $q$; theo dõi kích thước cụm đường thông liên thông lớn nhất (*giant connected component*) $G(q)$.
     * Khi tải tăng tới **ngưỡng tới hạn $q_c$**, cụm $G(q)$ vỡ đột ngột thành nhiều mảnh nhỏ (chuyển pha thấm — *percolation transition*). Các liên kết nối giữa các cụm lớn ngay tại sát ngưỡng $q_c$ chính là **điểm nghẽn động trọng yếu (*evolving critical bottlenecks*)**, thay đổi theo khung giờ.
     * **Thí nghiệm cải thiện một vài liên kết:** Họ giả lập nâng dung lượng/tốc độ của một số ít đoạn đường theo công thức $r'_{ij} = r_{ij}(1 + \alpha)$ rồi đo mức tăng của ngưỡng chịu tải toàn mạng $q_c$ dưới 3 chiến lược chọn đoạn đường: (i) *Evolving Critical Bottlenecks*, (ii) *Highest Betweenness Centrality* (chỉ số tô-pô tĩnh), và (iii) *Random*.
   * **Kết luận đã chứng minh:** Nâng cấp **chỉ một vài điểm nghẽn động** làm tăng mạnh $q_c$ (mạng lưới chịu tải cao hơn nhiều trước khi vỡ trận); ngược lại, nâng cấp các nút có **`Betweenness` tĩnh cao nhất** mang lại cải thiện gần như bằng `0` → khẳng định chỉ số đồ thị tĩnh như `Betweenness` không xác định được điểm nghẽn thực tế.

8. **Kerner (Springer 2004) — *The Physics of Traffic* (Lý thuyết Giao thông 3 Pha)**
   * **Chứng minh cốt lõi:** Giao thông cao tốc trải qua 3 pha thay vì 2 pha:
     1. *Free Flow ($F$):* Vận tốc cao, ổn định.
     2. *Synchronized Flow ($S$):* Sau điểm gãy pha $F \to S$, vận tốc giảm và **xuất hiện dao động vận tốc mạnh (*speed oscillations*)** do tương tác phanh–ga giữa các xe; trên biểu đồ `Flow – Occupancy`, pha $S$ phân tán thành một vùng 2 chiều rộng thay vì đường cong hẹp.
     3. *Wide Moving Jam ($J$):* Trong pha $S$ nén chặt xảy ra **hiệu ứng thắt nút (*pinch effect*)** tạo ra sóng dừng–đi (*stop-and-go*) rồi hợp nhất thành khối tắc rộng ($J$), khiến cả tốc độ lẫn lưu lượng sụp sâu và lan truyền ngược lên thượng lưu.
   * **Điểm chưa kiểm chứng trên `PEMS04/08`:** Lý thuyết được xây dựng trên cao tốc Đức và Mỹ qua dữ liệu vòng từ cục bộ, chưa chuẩn hóa thành bộ đặc trưng dao động trên đồ thị `PEMS04`/`PEMS08`.

9. **Chattopadhyay & Gupta (Royal Society Open Science, 2026) — *Do Multivariate Measures Anticipate Traffic Congestion?* (và các nghiên cứu của Ghadami & Epureanu)**
   * **Phương pháp & Kết quả:** Vận dụng nguyên lý **Chậm lại tới hạn (*Critical Slowing Down — CSD*)** trong vật lý hệ phi tuyến; đo các chỉ báo cửa sổ trượt đa biến (phương sai/độ lệch chuẩn, tự tương quan trễ 1) trên 3 biến `(mật độ, vận tốc, lưu lượng)`. Chứng minh các chỉ báo dao động này tăng vọt trước khi chuyển pha tắc nghẽn.
   * **Giới hạn quan trọng:** Mới kiểm chứng trên **dữ liệu quỹ đạo vi mô từng xe (NGSIM US-101)** ở 1 đoạn đường ngắn, **chưa kiểm chứng hệ thống trên mạng cảm biến vĩ mô 5 phút (`PEMS04`, `PEMS08`)**.

---

### Nhóm 3: Các Mô hình Dự báo STGNN / Attention / Transformer trên PeMS

> **Thiết lập mặc định chung của Nhóm 3:** Chia thời gian `60/20/20` (hoặc `70/10/20`), lấy `12` bước quá khứ (`60 phút`) dự báo `12` bước tương lai (`60 phút`), chỉ dự báo **1 biến mục tiêu duy nhất** (`Flow` trên `PEMS03/04/07/08` hoặc `Speed` trên `PEMS-BAY`/`PeMSD7`), đánh giá bằng `MAE`, `RMSE`, `MAPE` trung bình toàn mạng với `masked_mae(null_val=0.0)`.

| Mô hình & Tác giả | Dữ liệu PeMS | Kiến trúc cốt lõi | Đóng góp chính | Giới hạn cốt lõi |
| :--- | :--- | :--- | :--- | :--- |
| **DCRNN** (Li et al., ICLR 2018) | `PEMS-BAY` (`Speed`) | **Khuếch tán 2 chiều** (forward + backward random walk) trong `GRU` Seq2Seq | Chứng minh đồ thị 2 chiều bắt được cả tải đầu nguồn lẫn sóng tắc dội ngược hạ nguồn | Chậm do RNN; chỉ dự báo 1 biến `Speed`; phụ thuộc đồ thị khoảng cách tĩnh |
| **STGCN** (Yu et al., IJCAI 2018) | `PeMSD7(M/L)` (`Speed`) | Khối `Gated TCN – ChebNet – Gated TCN` | Huấn luyện nhanh hơn RNN rõ rệt | Đồ thị vô hướng tĩnh; không giải thích nút thắt |
| **ASTGCN** (Guo et al., AAAI 2019) | **`PEMS04`, `PEMS08`** | Spatial + Temporal Attention + ChebNet trên 3 nhánh chu kỳ (gần, ngày, tuần) | Chuẩn hóa `PEMS04/08`; khai thác chu kỳ ngày/tuần | Đồ thị tĩnh; chỉ dự báo 1 kênh `Flow`; chi phí cao |
| **Graph WaveNet** (Wu et al., IJCAI 2019) | `PEMS-BAY`, `PEMS04/08` | Ma trận kề tự học $A_{\text{adp}} = \text{SoftMax}(\text{ReLU}(E_1 E_2^T))$ + Dilated Causal TCN | Bắt liên kết ẩn ngoài `distance.csv` | $A_{\text{adp}}$ cố định sau train; khóa vào tập nút huấn luyện (không chuyển giao) |
| **ST-GAT** (Zhang et al., IEEE Access 2019) | PeMS | Graph Attention Network (GAT) + mạng hồi quy thời gian | Trọng số chú ý động trên cạnh đồ thị | Chỉ tối ưu sai số dự báo, chưa gắn vật lý tắc nghẽn hay XAI kiểm chứng |
| **STSGCN** (Song et al., AAAI 2020) | **`PEMS03/04/07/08`** | Ma trận đồ thị cục bộ $3N \times 3N$ nối 3 bước liền kề + STSGCM (cổng GLU) + Spatial/Temporal Embeddings + Huber Loss | Hợp nhất đồng bộ không–thời gian; chuẩn hóa bộ 4 tập `PEMS03/04/07/08` | Trường tiếp nhận cục bộ; nặng tham số theo từng bước; khóa cứng vào kích thước đồ thị |
| **GMAN** (Zheng et al., AAAI 2020) | `PEMS-BAY` | Encoder–Decoder thuần đa chú ý không–thời gian + Transform Attention | Giảm tích lũy sai số ở tầm dài (45–60p) | Chi phí $O(N^2)$ dễ OOM trên mạng lớn; cần Node2Vec cố định |
| **MTGNN** (KDD 2020) & **AGCRN** (NeurIPS 2020) | `PEMS03/04/07/08` | Học đồ thị từ dữ liệu + tham số thích nghi từng nút (`NAPL`/`DAGG`) | Không cần `distance.csv` vẫn đạt độ chính xác cao | Khóa cứng theo ID từng nút → **không tái sử dụng/chuyển giao được** sang bản đồ mới |
| **ASTGNN** (Guo et al., TKDE 2021) | `PEMS03/04/07/08` | Self-attention nhận thức xu hướng + mô hình hóa tính dị biệt động | Cải thiện xử lý chuỗi không dừng | Kiến trúc phức tạp, hộp đen |
| **STID** (Shao et al., CIKM 2022) | `PEMS04/08`, `PEMS-BAY` | Chỉ dùng MLP + Embedding định danh (`Node ID`, `Time-of-Day`, `Day-of-Week`) | Chứng minh định danh không–thời gian rất mạnh, ngang ngửa nhiều GNN phức tạp | Phụ thuộc tuyệt đối vào Node ID (không thể chuyển giao hoặc mở rộng nút mới); bỏ qua lan truyền vật lý |
| **PDFormer** (Jiang et al., AAAI 2023) | `PEMS04/07/08` | Transformer 2 nhánh (địa lý + ngữ nghĩa) + mô-đun **độ trễ lan truyền** | Mô hình hóa trễ truyền sóng tắc giữa các trạm (`PEMS08 MAE ~13,58`) | Độ phức tạp $O(N^2)$ tốn VRAM; vẫn chỉ dự báo 1 biến `Flow` |
| **MegaCRN** (Jiang et al., AAAI 2023) | `PEMS04/08`, `PEMS-BAY` | Meta-Graph Memory Bank + RNN đồ thị | Nắm bắt tốt các khuôn mẫu giao thông dị biệt | Chậm do RNN; nhạy kích thước bộ nhớ; chỉ đo MAE/RMSE trung bình |
| **STAEformer** (Liu et al., CIKM 2023) | `PEMS04/07/08`, `PEMS-BAY` | Vanilla Transformer + Spatio-Temporal Adaptive Embedding ($E_a$) | Đạt SOTA hàng đầu (`PEMS08 MAE ~13,46`, `PEMS04 ~18,22`) | Hộp đen; chi phí $O(N^2)$ dễ OOM trên mạng lớn; khóa cứng theo đồ thị huấn luyện |
| **GAMMA-Net** (2026) | `PEMS03/04/07/08` | Xen kẽ GAT không gian + State Space Model (Mamba) theo thời gian | Tuyến tính theo chiều thời gian, mạnh tầm xa | Vẫn chỉ dừng ở dự báo sai số trung bình toàn mạng |

### Nhóm 3B: Kiến trúc GNN Cảm ứng (Inductive), Kết hợp Cục bộ–Toàn cục & Tuyến tính $O(N)$ Chống OOM

Đây là nhóm bài báo then chốt giải trực tiếp 3 bài toán kỹ thuật: không khóa cứng đồ thị, chuyển giao được sang bản đồ mới, và không tràn bộ nhớ khi mạng lưới phình to:

* **GraphSAGE — Hamilton, Ying & Leskovec (NeurIPS 2017):** Khai sinh học biểu diễn đồ thị **cảm ứng (*inductive*)**. Thay vì học bảng vector cố định cho từng nút, mô hình học *hàm tổng hợp đặc trưng cục bộ* (mean/pooling/LSTM) dùng chung cho mọi nút, cho phép suy luận ngay trên nút mới hoặc đồ thị hoàn toàn mới chưa từng gặp khi huấn luyện.
* **GATv2 — Brody, Alon & Yahav (ICLR 2022):** Chứng minh GAT gốc (2018) chỉ có *chú ý tĩnh* (thứ hạng hàng xóm không phụ thuộc nút mục tiêu); sửa thứ tự phép kích hoạt `LeakyReLU` vào sau phép ghép `query || key` để tạo **Dynamic Graph Attention** linh hoạt hơn nhiều mà vẫn giữ chi phí tuyến tính theo số cạnh $O(|E|)$ và hoàn toàn cảm ứng.
* **GraphGPS — Rampášek et al. (NeurIPS 2022):** Đưa ra công thức chuẩn hóa cho Graph Transformer có khả năng mở rộng: chạy **song song hai nhánh** — nhánh **Local MPNN** ($O(|E|)$, gom tin cục bộ chuẩn xác theo cạnh) và nhánh **Global Linear Attention** ($O(N)$, nắm bắt bối cảnh toàn đồ thị mà không tốn $O(N^2)$).
* **BigST — Han et al. (PVLDB 2024):** Mạng STGNN **độ phức tạp tuyến tính $O(N)$** trên dữ liệu PeMS quy mô lớn (chứng minh chạy mượt tới `100.000` nút nhờ tách bộ trích xuất chuỗi dài có thể tiền tính toán và tích chập không gian toàn cục tuyến tính hóa).
* **PatchSTG — Fang et al. (ACM SIGKDD 2025):** Ứng dụng quản trị dữ liệu không gian (phân hoạch cây **KDTree**) chia cảm biến giao thông (`PEMS04`, `PEMS08`, `LargeST`) thành các mảnh cân bằng (*spatial patches*), phá vỡ nút cổ chai $O(N^2)$ của Transformer.

---

### Nhóm 4: Kiểm toán & Chuẩn hóa Benchmark (Meta-Research)

10. **LibCity (Wang et al., ACM SIGSPATIAL 2021):** Thư viện mã nguồn mở chuẩn hóa 35+ tập dữ liệu đô thị và 55+ mô hình giao thông trên cùng quy chuẩn chia tập (`60/20/20`), chuẩn hóa Z-score, và đánh giá (`masked_mae`, `masked_rmse`, `masked_mape` theo từng tầm dự báo `15p / 30p / 60p` hoặc trung bình `12` bước).
11. **LargeST (Liu et al., NeurIPS 2023):** Công bố bộ PeMS quy mô lớn tới `8.600` cảm biến (5 năm, 2017–2021) và chỉ ra phần lớn STGNN/Transformer (tốn $O(N^2)$) bị **tràn bộ nhớ (OOM)** khi mạng lưới phình to.
12. **BasicTS (Shao et al., IEEE TKDE 2025):** Ý tưởng cốt lõi là **kiểm toán công bằng và phân tích tính dị biệt dữ liệu (*heterogeneity*)**. Nhóm tác giả chỉ ra rằng: (i) khác biệt ngầm về tỷ lệ chia tập, chuẩn hóa và cách xử lý `null_val = 0` giữa các bài báo đã làm sai lệch bảng xếp hạng SOTA; (ii) khi chuẩn hóa pipeline công bằng, không có một kiến trúc nào thắng tuyệt đối trên mọi bộ dữ liệu (*No Real Champion*), thậm chí MLP + embedding (`STID`) cạnh tranh sòng phẳng với GNN phức tạp; (iii) hiệu năng mô hình phụ thuộc mạnh vào tính dị biệt của từng chuỗi thời gian.

---

### Nhóm 5: Dự báo Tắc nghẽn Tích hợp Vật lý trên `PEMS04` & `PEMS08` (Kiểm tra Độ trùng lặp)

13. **FD-Informed STGCN — Liu et al. (*Expert Systems with Applications*, Vol. 249, 123543, 2024)**
    * **Họ làm gì:** Nhúng tri thức tiên nghiệm từ **Biểu đồ cơ bản (Fundamental Diagram)** liên kết `(Flow, Speed, Density)` vào mô hình STGCN để dự báo lưu lượng mạng lưới.
    * **Khác biệt / Giới hạn:** Tập trung cải thiện sai số hồi quy lưu lượng trên đồ thị cố định; không xây kiến trúc cảm ứng (*inductive*) chuyển giao sang bản đồ khác và không đánh giá bài toán xác nhận điểm nghẽn gốc.

14. **STFDSGCN (MDPI, 2025) — *Spatio-Temporal Fusion GNN Based on Dynamic Sparse Graph Convolution GRU***
    * **Họ làm gì:** Kết hợp Dynamic Sparse Graph Convolution + GRU + cơ chế hợp nhất chú ý không–thời gian để xử lý dữ liệu đa biến bất quy tắc trên `PEMS04` và `PEMS08`, hướng tới cảnh báo sớm gián đoạn cao tốc.
    * **Khác biệt / Giới hạn:** Vẫn tập trung thiết kế kiến trúc hồi quy đánh giá bằng `MAE/RMSE/MAPE`; chưa kết hợp hệ tiêu chí vật lý nhận diện điểm nghẽn gốc, chưa kiểm chứng tính chuyển giao giữa các bản đồ (`PEMS08 → PEMS04`).

15. **CAP-STGCN (MDPI, 2026) — *Congestion-Aware Traffic Forecasting with Physics-Guided STGCN***
    * **Họ làm gì:** Chạy trên **`PEMS04` và `PEMS08`**, giải quyết 2 lỗi của STGCN thuần dữ liệu: (i) dự báo tạo ra trạng thái phi vật lý vi phạm Biểu đồ cơ bản, và (ii) mẫu tắc nghẽn quá hiếm (*long-tailed*) nên bị bỏ qua khi tối ưu MAE toàn cục. Họ nhúng **ràng buộc động học Biểu đồ cơ bản** + **cơ chế hiệu chỉnh trọng số vùng tắc (*Dynamic Congestion-Rectification & Weighting*)** để ép mô hình tập trung hơn vào các đợt bão hòa.
    * **Lưu ý tránh trùng lặp:** CAP-STGCN giao thoa gần về ý tưởng *khai thác vật lý 3 biến và xử lý mất cân bằng vùng tắc*, **nhưng** họ vẫn dùng STGCN gắn cố định với đồ thị huấn luyện (không *inductive*, không thiết kế mô hình cục bộ–toàn cục chống OOM và chuyển giao sang bản đồ mới), và vẫn đánh giá trọng tâm ở độ chính xác dự báo liên tục chứ chưa kết nối công thức xác nhận điểm nghẽn gốc đa bản đồ + giải thích nhân quả/phản thực tế.

16. **DCSD-Net (*IET Intelligent Transport Systems*, 2026) — *Density-Classification, Contrastive and Self-Distillation Network***
    * **Họ làm gì:** Chạy trên **`PEMS04` và `PEMS08`**, kết hợp 3 khối: (i) phân loại chế độ tắc nghẽn dựa trên mật độ (*density-based classification*), (ii) học tương phản bằng VAE (*VAE-based contrastive learning*) để đa dạng hóa biểu diễn thời gian, và (iii) cơ chế tự chưng cất (*self-distillation*) để giảm phương sai và triệt tiêu đỉnh sai số RMSE lúc kẹt xe.
    * **Lưu ý tránh trùng lặp:** DCSD-Net có yếu tố phân loại mức độ tắc nghẽn, **nhưng** kiến trúc nặng về học biểu diễn hộp đen (VAE + contrastive + self-distillation), chỉ tối ưu sai số dự báo trên từng tập riêng lẻ, không hướng tới tính nhẹ/chống OOM/tái sử dụng sang đồ thị khác kích thước và không tích hợp đặc trưng dao động vật lý (Kerner/CSD) hay XAI.

---

### Nhóm 6: AI Giải thích được (XAI), Phản thực tế & Điều tiết Chủ động

17. **GAT (Veličković et al., ICLR 2018)** & **SHAP (Lundberg & Lee, NeurIPS 2017):** Hai thuật toán nền tảng cho cơ chế trọng số chú ý cạnh $\alpha_{ij}$ và quy gán đóng góp đặc trưng theo giá trị Shapley.

18. **CGT — Yang, Du, Dai & Fang (IEEE ITSC 2023) — *Counterfactual Graph Transformer for Traffic Flow Prediction*** (pp. 521–527, DOI: `10.1109/ITSC57777.2023.10422176`)
    * **Họ làm gì trên `PEMS04` và `PEMS08`:** Xây dựng mô hình gồm 2 nhánh song song: (1) **Graph Transformer** để dự báo giao thông, và (2) **CGT-Explainer** để tạo lời giải thích **phản thực tế ở mức từng sự kiện (*instance-level counterfactual*)**.
    * **Cơ chế dễ hiểu:** Thay vì chỉ nhìn trọng số attention (dễ dính tương quan giả), CGT học đồng thời **2 mặt nạ nhiễu loạn liên tục**:
      * **Mặt nạ không gian ($M_S$):** che/giảm trọng số từng cạnh nối giữa các trạm.
      * **Mặt nạ thời gian – đặc trưng ($M_F$):** che/giảm từng bước quá khứ hoặc biến đầu vào.
      * Bộ tối ưu tìm $M_S, M_F$ sao cho **thay đổi ít nhất có thể** nhưng làm **kết quả dự báo thay đổi mạnh nhất** (ví dụ chuyển dự báo từ "tắc" thành "thông") → suy ra chính xác **tập con cạnh và khung giờ nào là nguyên nhân thực sự** gây ra dự báo đó. Cuối cùng có thể huấn luyện lại mô hình trên cấu trúc đã lọc nhiễu.
    * **Khác biệt với GAT:** GAT chỉ trả lời *"mô hình đang nhìn vào cạnh nào nhiều?"* (tương quan), còn CGT trả lời *"nếu tắt/thay đổi cạnh hay đặc trưng này thì kết quả tắc có biến mất không?"* (nhân quả phản thực tế).

19. **ALINEA — Papageorgiou, Hadj-Salem & Blosseville (TRR 1991) — *A Local Feedback Control Law for On-Ramp Metering***
    * **Biến sử dụng:** Bản gốc 1991 **chỉ dùng duy nhất 1 biến là `Occupancy`** tại 1 cảm biến ngay hạ lưu điểm nhập làn, theo luật điều khiển phản hồi tích phân:
      $$r(k) = r(k-1) - K_R \left[ O(k) - \hat{O} \right]$$
      trong đó $r(k)$ là lưu lượng cho phép vào nhánh nhập làn, $O(k)$ là độ chiếm dụng đo thực tế ở bước $k$, $\hat{O} = O_{\text{crit}}$ là độ chiếm dụng tới hạn (đỉnh công suất).
    * **Bản chất điều tiết (Thụ động vs. Chủ động):** ALINEA là bộ điều khiển **phản hồi thụ động (*reactive feedback*)** — nghĩa là phải đo thấy $O(k)$ lệch khỏi $\hat{O}$ rồi mới điều chỉnh ở bước kế tiếp. **Tuy nhiên**, nó không đợi đường kẹt cứng mới phản ứng mà kìm giữ $O(k)$ bám sát ngưỡng tới hạn $\hat{O}$ (ngay trước khi sụp pha *Capacity Drop*). Hạn chế cốt lõi là bản gốc không có tầng dự báo trước (*proactive / feed-forward*) và không nhìn được tải đang dồn từ mạng lưới rộng hơn.

---

### Nhóm 7: Các Hướng Mở rộng Khác trên `PEMS04` & `PEMS08`

20. **Định lượng Độ bất định (*Uncertainty Quantification*):** **DeepSTUQ** (IEEE TKDE), **STAC-Net** (PLOS ONE), **MemCNP** — chạy trên `PEMS04/08`, dự báo khoảng tin cậy (*aleatoric* + *epistemic uncertainty*) thay vì chỉ 1 giá trị điểm.
21. **Chuyển giao & Cảm biến Khởi động lạnh (*Cold-Start / Inductive Transfer*):** **Cold-Start Meta-Graph Learning** (MDPI, `PEMS04/08`), **STBaT** (IJCAI) — dự báo tại các trạm chưa từng thấy hoặc chuyển giữa các khu vực địa lý khác nhau.
22. **Phát hiện Bất thường & Nhân quả Không–Thời gian:** **SUMO-enhanced Anomaly Detection** (PeMS), **CGProNet / Causal STGNN** (`PEMS04/08`) — tách tắc nghẽn lặp lại theo giờ cao điểm khỏi tai nạn đột phát và tách tương quan giả (*confounder*) khỏi lan truyền nhân quả thực.

---

## 3. Danh mục Tài liệu Tham khảo Chuẩn IEEE

1. C. Chen, K. Petty, A. Skabardonis, P. Varaiya, and Z. Jia, "Freeway performance measurement system: Mining loop detector data," *Transp. Res. Rec.*, vol. 1748, no. 1, pp. 96–102, 2001. DOI: `10.3141/1748-12`.
2. C. Chen, J. Kwon, J. Rice, A. Skabardonis, and P. Varaiya, "Detecting errors and imputing missing data for single-loop surveillance systems," *Transp. Res. Rec.*, vol. 1855, no. 1, pp. 160–167, 2003.
3. A. Cini, I. Marisca, and C. Alippi, "Filling the gaps: Multivariate time series imputation by graph neural networks," in *Proc. ICLR*, 2022.
4. M. J. Cassidy and R. L. Bertini, "Some traffic features at freeway bottlenecks," *Transp. Res. Part B: Methodol.*, vol. 33, no. 1, pp. 25–42, 1999.
5. T. Choe, A. Skabardonis, and P. Varaiya, "Freeway performance measurement system (PeMS): An operational analysis tool," in *Proc. 81st Annu. Meeting Transp. Res. Board*, 2002.
6. C. Chen, A. Skabardonis, and P. Varaiya, "Systematic identification of freeway bottlenecks," *Transp. Res. Rec.*, vol. 1867, pp. 46–52, 2004. DOI: `10.3141/1867-06`.
7. B. S. Kerner, *The Physics of Traffic: Empirical Freeway Pattern Features, Engineering Applications, and Theory*. Berlin, Heidelberg: Springer, 2004.
8. D. Li, B. Fu, Y. Wang, G. Lu, Y. Berezin, H. E. Stanley, and S. Havlin, "Percolation transition in dynamical traffic network with evolving critical bottlenecks," *Proc. Natl. Acad. Sci. (PNAS)*, vol. 112, no. 3, pp. 669–672, 2015.
9. S. N. Chattopadhyay and A. K. Gupta, "Do multivariate measures anticipate traffic congestion?," *Royal Society Open Science*, 2026. DOI: `10.1098/rsos.252185`.
10. Y. Li, R. Yu, C. Shahabi, and Y. Liu, "Diffusion convolutional recurrent neural network: Data-driven traffic forecasting," in *Proc. ICLR*, 2018.
11. B. Yu, H. Yin, and Z. Zhu, "Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting," in *Proc. IJCAI*, pp. 3634–3640, 2018.
12. P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph attention networks," in *Proc. ICLR*, 2018.
13. S. Guo, Y. Lin, N. Feng, C. Song, and H. Wan, "Attention based spatial-temporal graph convolutional networks for traffic flow forecasting," in *Proc. AAAI*, vol. 33, no. 01, pp. 922–929, 2019.
14. Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, and P. S. Yu, "Graph WaveNet for deep spatial-temporal graph modeling," in *Proc. IJCAI*, pp. 1907–1913, 2019.
15. C. Zhang, J. J. Q. Yu, and Y. Liu, "Spatial-temporal graph attention networks: A deep learning approach for traffic forecasting," *IEEE Access*, vol. 7, pp. 166246–166256, 2019.
16. C. Song, Y. Lin, S. Guo, and H. Wan, "Spatial-temporal synchronous graph convolutional networks: A new framework for spatial-temporal network data forecasting," in *Proc. AAAI*, vol. 34, no. 04, pp. 914–921, 2020.
17. C. Zheng, X. Fan, C. Wang, and J. Qi, "GMAN: A graph multi-attention network for traffic prediction," in *Proc. AAAI*, vol. 34, no. 01, pp. 1234–1241, 2020.
18. Z. Wu et al., "Connecting the dots: Multivariate time series forecasting with graph neural networks," in *Proc. ACM SIGKDD*, pp. 753–763, 2020.
19. L. Bai, L. Yao, C. Li, X. Wang, and C. Wang, "Adaptive graph convolutional recurrent network for traffic forecasting," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, pp. 17804–17815, 2020.
20. S. Guo, Y. Lin, H. Wan, X. Li, and G. Cong, "Learning dynamics and heterogeneity of spatial-temporal graph data for traffic forecasting," *IEEE Trans. Knowl. Data Eng.*, vol. 34, no. 11, pp. 5415–5428, 2021.
21. J. Wang, J. Jiang, W. Jiang, C. Li, and W. X. Zhao, "LibCity: An open library for traffic prediction," in *Proc. ACM SIGSPATIAL*, pp. 145–148, 2021. DOI: `10.1145/3474717.3483923`.
22. Z. Shao, Z. Zhang, F. Wang, W. Wei, and Y. Xu, "Spatial-temporal identity: A simple yet effective baseline for multivariate time series forecasting," in *Proc. ACM CIKM*, pp. 4454–4458, 2022.
23. J. Jiang, C. Han, W. X. Zhao, and J. Wang, "PDFormer: Propagation delay-aware dynamic long-range transformer for traffic flow prediction," in *Proc. AAAI*, vol. 37, no. 4, pp. 4365–4373, 2023.
24. H. Liu et al., "STAEformer: Spatio-temporal adaptive embedding makes vanilla transformer SOTA for traffic forecasting," in *Proc. ACM CIKM*, pp. 1419–1428, 2023.
25. X. Liu et al., "LargeST: A benchmark dataset for large-scale traffic forecasting," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 36, pp. 75354–75371, 2023.
26. Z. Shao et al., "Exploring progress in multivariate time series forecasting: Comprehensive benchmarking and heterogeneity analysis," *IEEE Trans. Knowl. Data Eng.*, vol. 37, no. 1, pp. 291–305, 2025.
27. Z. Liu, F. Ding, Y. Dai, L. Li, T. Chen, and H. Tan, "Spatial-temporal graph convolution network model with traffic fundamental diagram information informed for network traffic flow prediction," *Expert Syst. Appl.*, vol. 249, 123543, 2024.
28. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, pp. 4765–4774, 2017.
29. Y. Yang, K. Du, X. Dai, and J. Fang, "Counterfactual graph transformer for traffic flow prediction," in *Proc. IEEE 26th Int. Conf. Intell. Transp. Syst. (ITSC)*, pp. 521–527, 2023. DOI: `10.1109/ITSC57777.2023.10422176`.
30. M. Papageorgiou, H. Hadj-Salem, and J. M. Blosseville, "ALINEA: A local feedback control law for on-ramp metering," *Transp. Res. Rec.*, vol. 1320, pp. 58–64, 1991.
31. W. L. Hamilton, R. Ying, and J. Leskovec, "Inductive representation learning on large graphs," in *Adv. Neural Inf. Process. Syst. (NIPS)*, vol. 30, 2017.
32. S. Brody, U. Alon, and E. Yahav, "How attentive are graph attention networks?," in *Proc. ICLR*, 2022.
33. L. Rampášek, M. Galkin, V. P. Dwivedi, A. T. Luu, G. Wolf, and D. Beaini, "Recipe for a general, powerful, scalable graph transformer," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 35, pp. 14501–14515, 2022.
34. J. Han, W. Zhang, H. Liu, T. Tao, N. Tan, and H. Xiong, "BigST: Linear complexity spatio-temporal graph neural network for traffic forecasting on large-scale road networks," *Proc. VLDB Endow.*, vol. 17, no. 5, pp. 1081–1090, 2024.
35. Y. Fang et al., "Efficient large-scale traffic forecasting with transformers: A spatial data management perspective," in *Proc. 31st ACM SIGKDD Conf. Knowl. Discov. Data Min.*, pp. 307–317, 2025.
