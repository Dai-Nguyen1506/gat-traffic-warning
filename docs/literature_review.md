# Tổng hợp Văn học Khoa học & Bản đồ Trích dẫn Dự án (`docs/literature_review.md`)

Tài liệu này tổng hợp tinh gọn **đúng các bài báo được sử dụng trực tiếp** trong lộ trình 5 notebook (`NB01 – NB05`) của đề tài v7 (*Physics-Informed Pre-Congestion Signals for Freeway Bottleneck Forecasting with Graph Attention*), kèm **bản đồ trích dẫn tới từng notebook / mục cụ thể** để phục vụ viết bài báo chuẩn IEEE ($\le 3.000$ từ).

> Các bài báo không còn nằm trong phạm vi đề tài (như điền khuyết 2 chiều `GRIN`, điều tiết phản thực tế `ALINEA`/`CGT`, mở rộng 8.600 nút `LargeST`/`BigST`/`PatchSTG`, hay các kiến trúc dự báo đơn biến hộp đen) đã được lược bỏ để giữ tài liệu tập trung tối đa.

---

## 1. Bản đồ Đối chiếu Trích dẫn theo Từng Notebook (`NB01 – NB05`)

| Notebook & Mục sử dụng | Vai trò trong dự án | Tài liệu trích dẫn trực tiếp |
| :--- | :--- | :--- |
| **`NB01`** *(Mục 1–4)* | Nguồn gốc hệ thống vòng từ Caltrans PeMS (gộp 5 phút) và chuẩn hóa 4 tập `PEMS03/04/07/08` (giải thích lý do chỉ `PEMS04/08` có đủ 3 biến `Flow, Occ, Speed`). | **Chen et al. (2001)**; **Guo et al. (2019 — ASTGCN)**; **Song et al. (2020 — STSGCN)** |
| **`NB02`** *(Toàn bộ Phần 2–5, `src/cleaning.py`)* | Chẩn đoán 4 chỉ số ngày ($S_1 \dots S_4$ gồm Entropy) tách đường vắng đêm khỏi lỗi `0.0` & lỗi đóng băng hằng số; phục hồi thuần bằng **hồi quy tuyến tính cặp trạm lân cận** ($K = 5$, trọng số $R^2$). | **Chen et al. (2003)** |
| **`NB03` — Phần 2** *(`src/bottlenecks.py`)* | Công thức tự hiệu chỉnh ngưỡng tắc nghẽn theo quy chuẩn vận tốc `< 60%` tốc độ tự do ($v_{\text{cong}} = 0{,}60\,v_{\text{ff}}$), đối chiếu mốc `~40 mph` PeMS, ước lượng ngưỡng bão hòa $o_{\text{sat}}$ theo Biểu đồ cơ bản (`11%–15%`), và chứng minh loại trừ `Flow` do hiệu ứng *Capacity Drop* (`-13,7%`). | **FHWA (23 CFR Part 490)**; **Chen, Skabardonis & Varaiya (2004)**; **Choe, Skabardonis & Varaiya (2002) & HCM**; **Cassidy & Bertini (1999)** |
| **`NB03` — Phần 3** *(`src/bottlenecks.py`)* | Định nghĩa Pha chuyển tiếp (*Synchronized Flow*, `v_sync = 0,80 × v_ff`) và hệ **`7` chỉ báo tiền tắc nghẽn (`H = 12` = `60p`)**: Nhóm 1 Dao động khử dốc (*Stop-and-Go*), Nhóm 2 Liên kết đa biến (`occ_spd_corr`, `flow_occ_decouple`) vượt trội đơn biến, Nhóm 3 Độ dốc có dấu tách pha sập tốc vs. hồi phục thoát tắc. | **Kerner (2004)**; **Chattopadhyay & Gupta (RSOS, 2026)**; **Payne & Tignor (1978 — California)**; **Persaud & Hall (1989 — McMaster)** |
| **`NB03` — Phần 4** | Kiểm định giả thuyết loại trừ hình học đồ thị tĩnh (`Betweenness`, `p = 0,3966`) và chứng minh tương tác động trên cạnh có hướng 2 chiều (`In` đầu vào + `Out` đầu ra dội ngược). | **Li et al. (PNAS, 2015)**; **Li et al. (ICLR, 2018 — DCRNN)** |
| **`NB04` & `NB05`** *(RQ2 — `PEMS08` & kiểm chứng `PEMS04`)* | Thiết kế đối chứng có kiểm soát (`Base` vs. `+ 7 chỉ báo`) trên chuỗi thời gian (`GRU`) và mạng chú ý đồ thị động (**`ST-GAT` với `GATv2`**), phân tích đóng góp nhóm đặc trưng và trực quan hóa trọng số chú ý cạnh $\alpha_{ij}$. | **Li et al. (2018 — DCRNN)**; **Veličković et al. (2018 — GAT)**; **Zhang et al. (2019 — ST-GAT)**; **Brody et al. (2022 — GATv2)** |

---

## 2. Tóm tắt Chi tiết Các Bài báo Cốt lõi Đang Sử dụng (15 Tài liệu)

### Nhóm A: Hệ thống PeMS, Chuẩn hóa Dữ liệu & Làm sạch (`NB01 – NB02`)

1. **Chen et al. (TRR, 2001) — *Freeway Performance Measurement System: Mining Loop Detector Data***
   * **Nội dung:** Công bố hệ thống cảm biến vòng từ đơn Caltrans PeMS, quy trình tổng hợp dữ liệu 30 giây thành chu kỳ **5 phút** (`288` bước/ngày) gồm bộ ba `(Flow, Occupancy, Speed)`.
   * **Dùng tại:** `NB01` (Mục 1–2) để nêu nguồn gốc dữ liệu và đơn vị đo.
2. **Guo et al. (AAAI, 2019 — ASTGCN) & Song et al. (AAAI, 2020 — STSGCN)**
   * **Nội dung:** Chuẩn hóa 4 bộ benchmark đồ thị cao tốc `PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`; trong đó chỉ `PEMS04` (`307` trạm) và `PEMS08` (`170` trạm) cung cấp đủ cả `3` biến vật lý `(Flow, Occupancy, Speed)`.
   * **Dùng tại:** `NB01` (Mục 2–4) làm căn cứ loại `PEMS03/07` và phân vai `PEMS08` (chính) / `PEMS04` (kiểm chứng độc lập).
3. **Chen et al. (TRR, 2003) — *Detecting Errors and Imputing Missing Data for Single-Loop Surveillance Systems***
   * **Nội dung:** Thuật toán chuẩn vận hành thực tế trên Caltrans PeMS gồm 2 bước: (i) **Daily Statistics Algorithm (DSA)** dùng `4` chỉ số ngày ($S_1$: số điểm `Occ = 0`, $S_2$: lệch pha `Flow/Occ`, $S_3$: kẹt cao cực trị, $S_4$: Shannon Entropy phát hiện cảm biến đóng băng hằng số); (ii) **Hồi quy tuyến tính cặp trạm lân cận** $\hat{x}_i(t) = \hat{a}_{ij} x_j(t) + \hat{b}_{ij}$ tổ hợp theo trọng số $R^2_{ij}$ từ các cảm biến lân cận đang sạch.
   * **Dùng tại:** Toàn bộ **`NB02`** và mô-đun [`src/cleaning.py`](../src/cleaning.py) (đạt `0` lỗi tồn đọng trên `PEMS08` và `PEMS04`).

---

### Nhóm B: Biểu đồ Cơ bản, Ngưỡng Tắc nghẽn & Loại trừ `Flow` (`NB03` — Phần 2)

4. **FHWA (23 CFR Part 490 / NPMRDS) — *National Performance Management Measures***
   * **Nội dung:** Quy chuẩn Cục Đường bộ Liên bang Hoa Kỳ định nghĩa chậm trễ/tắc nghẽn quá mức khi vận tốc rơi xuống dưới **`60%` tốc độ dòng tự do** ($v < 0{,}60\,v_{\text{ff}}$), giúp thích nghi với mọi tuyến cao tốc có giới hạn tốc độ khác nhau.
   * **Dùng tại:** **`NB03` Phần 2** (`calibrate_congestion_thresholds` trong [`src/bottlenecks.py`](../src/bottlenecks.py)) và áp dụng lại nguyên bản lên `PEMS04` ở `NB05`.
5. **Chen, Skabardonis & Varaiya (TRR, 2004) — *Systematic Identification of Freeway Bottlenecks***
   * **Nội dung:** Xác lập mốc vận tốc tắc nghẽn kinh điển **`< 40 mph`** trên dữ liệu 5 phút Caltrans PeMS.
   * **Dùng tại:** **`NB03` Phần 2** để đối chiếu xác nhận ngưỡng tự tính `40,26 mph` trên `PEMS08` khớp sát chuẩn PeMS.
6. **Choe, Skabardonis & Varaiya (TRB, 2002) & *Highway Capacity Manual (HCM)***
   * **Nội dung:** Chuẩn hóa Biểu đồ cơ bản `Flow – Occupancy – Speed` trên vòng từ PeMS, chỉ ra vùng bắt đầu bão hòa tại `Occupancy ~ 11%–15%` và quá tải nặng tại `≥ 30%`.
   * **Dùng tại:** **`NB03` Phần 2** để đối chiếu ngưỡng bão hòa tự tính `o_sat = 13,25%` (tại `90% Q_max` nhánh lên), lọc sạch `4.752` điểm chạy chậm khi đường vắng.
7. **Cassidy & Bertini (TRB, 1999) — *Some Traffic Features at Freeway Bottlenecks***
   * **Nội dung:** Chứng minh hiệu ứng **Sụt giảm Công suất (*Capacity Drop*)** khoảng `8%–10%`: lưu lượng xả sau khi hình thành hàng đợi tắc nghẽn thấp hơn lưu lượng đỉnh trước khi tắc.
   * **Dùng tại:** **`NB03` Phần 2** để giải thích bản chất vật lý của hiện tượng `Flow` tụt ngược `-13,7%` khi quá tải trên `PEMS08` và chứng minh vì sao không dùng `Flow` để đặt ngưỡng tắc.

---

### Nhóm C: Chuyển pha & Hệ `7` Chỉ báo Tiền tắc nghẽn (`NB03` — Phần 3)

8. **Kerner (Springer, 2004) — *The Physics of Traffic* (Lý thuyết Giao thông 3 Pha)**
   * **Nội dung:** Giao thông cao tốc chuyển từ *Free Flow ($F$)* qua **Pha chuyển tiếp *Synchronized Flow ($S$)*** (xuất hiện dao động vận tốc và mật độ dừng–đi *stop-and-go*) trước khi sụp hẳn vào tắc rộng (*Wide Moving Jam, $J$*).
   * **Dùng tại:** **`NB03` Phần 3** cho định nghĩa Pha chuyển tiếp (`v_sync = 0,80 × v_ff = 53,68 mph`) và **Nhóm 1** (`occ_rstd, spd_rstd, flow_rstd`).
9. **Chattopadhyay & Gupta (*Royal Society Open Science*, 2026) — *Do Multivariate Measures Anticipate Traffic Congestion?***
   * **Nội dung:** Áp dụng lý thuyết *Critical Slowing Down (CSD)* cho giao thông và chứng minh **các chỉ báo đa biến** kết hợp đồng thời `(mật độ, vận tốc, lưu lượng)` (tương quan chéo, hiệp phương sai đa biến) cùng bước **khử xu hướng (*detrending*)** vượt trội chỉ báo đơn biến vốn phản ứng thiếu ổn định.
   * **Dùng tại:** **`NB03` Phần 3** (`extract_precongestion_features`) cho bước khử xu hướng tuyến tính ở Nhóm 1 và **Nhóm 2 Liên kết Đa biến** (`occ_spd_corr = -corr(Occ, Speed)` đạt `AUC = 0,85`, `flow_occ_decouple = -corr(Flow, Occ)` đạt `AUC = 0,82`).
10. **Payne & Tignor (TRR, 1978 — *California Algorithm*) & Persaud & Hall (TRR, 1989 — *McMaster Algorithm*)**
    * **Nội dung:** Các thuật toán kinh điển phát hiện gãy pha/sự cố cao tốc dựa trên tốc độ biến thiên có dấu theo thời gian của `Occupancy` (tăng dương) và `Speed` (giảm âm) trên Biểu đồ cơ bản.
    * **Dùng tại:** **`NB03` Phần 3** cho **Nhóm 3 Độ dốc có dấu** (`occ_slope`, `spd_slope`) — đồng thời là phần xu hướng tách ra từ phép khử dốc Nhóm 1, giúp đảo dấu rõ rệt ở pha hồi phục thoát tắc để chặn báo động giả.

---

### Nhóm D: Cấu trúc Đồ thị Tĩnh vs. Động & Mô hình Dự báo Chú ý Đồ thị (`NB03` Phần 4, `NB04 – NB05`)

11. **Li et al. (*PNAS*, 2015) — *Percolation Transition in Dynamical Traffic Network with Evolving Critical Bottlenecks***
    * **Nội dung:** Chứng minh sự sụp đổ tắc nghẽn toàn mạng do một nhóm nút thắt động chi phối chứ **không tương quan với độ trung tâm tô-pô tĩnh như `Betweenness Centrality`**.
    * **Dùng tại:** **`NB03` Phần 4 (Mặt 1)** làm cơ sở lý thuyết cho giả thuyết $H_0$ về `Betweenness` tĩnh (`Spearman ρ = 0,0654, p = 0,3966`).
12. **Li et al. (ICLR, 2018 — DCRNN) — *Diffusion Convolutional Recurrent Neural Network***
    * **Nội dung:** Khẳng định dòng giao thông lan truyền **2 chiều trên đồ thị có hướng** (tải đổ vào từ thượng nguồn `In` và sóng tắc dội ngược từ hạ nguồn `Out`), kết hợp mô hình hồi quy `GRU`.
    * **Dùng tại:** **`NB03` Phần 4 (Mặt 2)** chứng minh trạng thái động lân cận `In + Out` nâng `Val PR-AUC` thêm `+1,35%` (`5p`) đến `+5,53%` (`30p`), và làm cơ sở cho mô hình đối sánh `Temporal GRU` + đồ thị có hướng 2 chiều ở **`NB04 – NB05`**.
13. **Veličković et al. (ICLR, 2018 — GAT), Zhang et al. (IEEE Access, 2019 — ST-GAT) & Brody et al. (ICLR, 2022 — GATv2)**
    * **Nội dung:** Cơ chế chú ý đồ thị học trọng số động $\alpha_{ij}$ trên từng cạnh có hướng; `GATv2` khắc phục giới hạn chú ý tĩnh của `GAT` gốc thành *Dynamic Graph Attention* chi phí tuyến tính $O(|E|)$; `ST-GAT` kết hợp tầng thời gian + chú ý đồ thị cho bài toán giao thông.
    * **Dùng tại:** **`NB04` & `NB05` (RQ2)** để xây dựng mô hình `ST-GAT` gọn nhẹ trên `PyTorch`, đánh giá mức cải thiện khi thêm `7` chỉ báo tiền tắc nghẽn và trực quan hóa trọng số chú ý cạnh quanh các nút tắc.

---

## 3. Các Bài báo Có Giá trị Cho Mục *Related Work / Methodology* (Chờ Bạn Duyệt Trước Khi Bỏ)

Các bài báo dưới đây **không cài đặt thuật toán trực tiếp trong `NB01 – NB03`**, nhưng có giá trị học thuật cao để củng cố phần *Related Work* và biện luận thiết lập thí nghiệm cho bài báo IEEE. Bạn xem qua nhóm nào muốn giữ hay bỏ hẳn:

1. **LibCity (Wang et al., ACM SIGSPATIAL 2021) & BasicTS (Shao et al., IEEE TKDE 2025):** Chuẩn hóa quy trình chia thời gian `60 / 20 / 20` và cửa sổ `12 → 12` bước công bằng trên `PEMS04 / PEMS08` — *nên giữ để trích dẫn chuẩn đánh giá ở `NB04` và `NB05`*.
2. **STID (Shao et al., CIKM 2022):** Chứng minh đặc trưng chu kỳ giờ–thứ (`tod / dow`) mang tín hiệu cực mạnh trên `PEMS04 / PEMS08` — *hợp thức hóa việc đưa `4` đặc trưng chu kỳ `sin/cos` vào tập nền `Base` ở `NB04`*.
3. **PDFormer (Jiang et al., AAAI 2023):** Chỉ ra hiệu ứng **độ trễ lan truyền không gian (*propagation delay*)** giữa các trạm — *giải thích khớp kết quả `NB03` Phần 4 rằng lợi ích đồ thị động `In + Out` tăng vọt ở tầm `15–30p` (`+5,5%`) so với `5p` (`+1,35%`)*.
4. **GraphSAGE (Hamilton et al., NeurIPS 2017):** Nền tảng học đồ thị cảm ứng (*inductive*, chia sẻ trọng số không khóa ID nút) — *hỗ trợ lập luận tổng quát hóa sang bản đồ mới `PEMS04` ở `NB05`*.
5. **Nhóm Physics-Guided / Congestion-Aware STGNN mới (2024–2026) trên `PEMS04 / PEMS08`:** `FD-Informed STGCN` (Liu et al., *ESWA* 2024), `CAP-STGCN` (MDPI, 2026), `DCSD-Net` (*IET ITS*, 2026) — *rất hữu ích cho phần Related Work bài báo IEEE để khẳng định hướng kết hợp vật lý tắc nghẽn đang rất thời sự và nêu rõ điểm khác biệt (họ nhúng ràng buộc vào loss/hộp đen, còn mình trích xuất trực tiếp `7` chỉ báo tiền tắc nghẽn minh bạch và định lượng đóng góp từng nhóm đặc trưng)*.

---

## 4. Danh mục Tài liệu Tham khảo Chuẩn IEEE (15 Tài liệu Cốt lõi Đang Dùng)

1. C. Chen, K. Petty, A. Skabardonis, P. Varaiya, and Z. Jia, "Freeway performance measurement system: Mining loop detector data," *Transp. Res. Rec.*, vol. 1748, no. 1, pp. 96–102, 2001. DOI: `10.3141/1748-12`.
2. S. Guo, Y. Lin, N. Feng, C. Song, and H. Wan, "Attention based spatial-temporal graph convolutional networks for traffic flow forecasting," in *Proc. AAAI*, vol. 33, no. 01, pp. 922–929, 2019.
3. C. Song, Y. Lin, S. Guo, and H. Wan, "Spatial-temporal synchronous graph convolutional networks: A new framework for spatial-temporal network data forecasting," in *Proc. AAAI*, vol. 34, no. 04, pp. 914–921, 2020.
4. C. Chen, J. Kwon, J. Rice, A. Skabardonis, and P. Varaiya, "Detecting errors and imputing missing data for single-loop surveillance systems," *Transp. Res. Rec.*, vol. 1855, no. 1, pp. 160–167, 2003.
5. Federal Highway Administration (FHWA), "National Performance Management Measures; Assessing Performance of the National Highway System, Freight Movement on the Interstate System, and Congestion Mitigation and Air Quality Improvement Program," *Code of Federal Regulations*, Title 23, Part 490 (23 CFR Part 490), U.S. Dept. of Transportation.
6. C. Chen, A. Skabardonis, and P. Varaiya, "Systematic identification of freeway bottlenecks," *Transp. Res. Rec.*, vol. 1867, pp. 46–52, 2004. DOI: `10.3141/1867-06`.
7. T. Choe, A. Skabardonis, and P. Varaiya, "Freeway performance measurement system (PeMS): An operational analysis tool," in *Proc. 81st Annu. Meeting Transp. Res. Board*, 2002.
8. M. J. Cassidy and R. L. Bertini, "Some traffic features at freeway bottlenecks," *Transp. Res. Part B: Methodol.*, vol. 33, no. 1, pp. 25–42, 1999.
9. B. S. Kerner, *The Physics of Traffic: Empirical Freeway Pattern Features, Engineering Applications, and Theory*. Berlin, Heidelberg: Springer, 2004.
10. S. N. Chattopadhyay and A. K. Gupta, "Do multivariate measures anticipate traffic congestion?," *Royal Society Open Science*, vol. 13, no. 5, 252185, 2026. DOI: `10.1098/rsos.252185`.
11. H. J. Payne and S. C. Tignor, "Freeway incident-detection algorithms based on decision trees with states," *Transp. Res. Rec.*, no. 682, pp. 30–37, 1978; and B. N. Persaud and F. L. Hall, "Catastrophe theory and patterns in 30-second freeway traffic data — Implications for incident detection," *Transp. Res. Part A*, vol. 23, no. 2, pp. 103–113, 1989.
12. D. Li, B. Fu, Y. Wang, G. Lu, Y. Berezin, H. E. Stanley, and S. Havlin, "Percolation transition in dynamical traffic network with evolving critical bottlenecks," *Proc. Natl. Acad. Sci. (PNAS)*, vol. 112, no. 3, pp. 669–672, 2015.
13. Y. Li, R. Yu, C. Shahabi, and Y. Liu, "Diffusion convolutional recurrent neural network: Data-driven traffic forecasting," in *Proc. ICLR*, 2018.
14. P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph attention networks," in *Proc. ICLR*, 2018; and C. Zhang, J. J. Q. Yu, and Y. Liu, "Spatial-temporal graph attention networks: A deep learning approach for traffic forecasting," *IEEE Access*, vol. 7, pp. 166246–166256, 2019.
15. S. Brody, U. Alon, and E. Yahav, "How attentive are graph attention networks?," in *Proc. ICLR*, 2022.
