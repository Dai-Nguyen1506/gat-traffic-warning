# Tổng hợp Văn học Khoa học & Bản đồ Trích dẫn Dự án (`docs/literature_review.md`)

Tài liệu này tổng hợp trọn vẹn **bộ `22` tài liệu tham khảo chính thức** đã được duyệt sử dụng cho lộ trình 5 notebook (`NB01 – NB05`) và bài báo cuối kỳ chuẩn IEEE ($\le 3.000$ từ) của đề tài v7 (*Physics-Informed Pre-Congestion Signals for Freeway Bottleneck Forecasting with Graph Attention*).

---

## 1. Bản đồ Ánh xạ Trích dẫn theo Từng Notebook & Mục Bài báo IEEE

| Phạm vi sử dụng | Vai trò trong dự án | Tài liệu trích dẫn trực tiếp |
| :--- | :--- | :--- |
| **`NB01`** *(Mục 1–4)* | Nguồn gốc hệ thống vòng từ Caltrans PeMS (5 phút) và chuẩn hóa 4 tập `PEMS03/04/07/08` (chọn `PEMS08` / `PEMS04` đủ 3 biến). | **Chen et al. (2001)**; **Guo et al. (2019 — ASTGCN)**; **Song et al. (2020 — STSGCN)** |
| **`NB02`** *(Phần 2–5, `src/cleaning.py`)* | Chẩn đoán 4 chỉ số ngày ($S_1 \dots S_4$) và phục hồi thuần bằng **hồi quy tuyến tính cặp trạm lân cận** ($K = 5$, trọng số $R^2$). | **Chen et al. (2003)** |
| **`NB03` — Phần 2** *(`src/bottlenecks.py`)* | Tự hiệu chỉnh ngưỡng tắc `< 60%` tốc độ tự do ($v_{\text{cong}} = 0{,}60\,v_{\text{ff}}$), đối chiếu `~40 mph` PeMS, ước lượng ngưỡng bão hòa $o_{\text{sat}}$ (`11%–15%`), và loại trừ `Flow` do *Capacity Drop* (`-13,7%`). | **FHWA (23 CFR Part 490)**; **Chen et al. (2004)**; **Choe et al. (2002) & HCM**; **Cassidy & Bertini (1999)** |
| **`NB03` — Phần 3** *(`src/bottlenecks.py`)* | Pha chuyển tiếp (*Synchronized Flow*) và chốt **`4` nhóm đặc trưng (`14` đầu vào)**: Nhóm 1 Dao động khử dốc, Nhóm 2 Liên kết đa biến (`occ_spd_corr`, `flow_occ_decouple`), Nhóm 3 Độ dốc có dấu, **Nhóm 4 Chu kỳ thời gian `sin/cos`**. | **Kerner (2004)**; **Chattopadhyay & Gupta (RSOS, 2026)**; **California (Payne & Tignor, 1978)** & **McMaster (Persaud & Hall, 1989)**; **STID (Shao et al., CIKM 2022)** |
| **`NB03` — Phần 4** | Kiểm định loại trừ hình học tĩnh (`Betweenness`, `p = 0,3966`) và chứng minh đồ thị động 2 chiều `In + Out` tăng mạnh nhất ở tầm xa `15–30p` do *độ trễ lan truyền không gian*. | **Li et al. (PNAS, 2015)**; **DCRNN (Li et al., ICLR 2018)**; **PDFormer (Jiang et al., AAAI 2023)** |
| **`NB04` & `NB05`** *(RQ2 — `PEMS08` & `PEMS04`)* | Chuẩn hóa chia thời gian `60 / 20 / 20`, cửa sổ `12 → 12` bước; biện luận chọn bộ 3 mô hình gọn nhẹ có tính cảm ứng đa bản đồ (`Persistence` → `Temporal GRU` → `ST-GAT` với `GATv2`); đánh giá đóng góp nhóm đặc trưng và trọng số chú ý cạnh $\alpha_{ij}$. | **LibCity (Wang et al., 2021)**; **BasicTS (Shao et al., TKDE 2025)**; **GraphSAGE (Hamilton et al., 2017)**; **DCRNN (2018)**; **GAT (Veličković et al., 2018)**; **ST-GAT (Zhang et al., 2019)**; **GATv2 (Brody et al., 2022)** |
| **Bài báo IEEE** *(Mục Related Work)* | Đối chiếu trực tiếp với xu hướng *Physics-Guided / Congestion-Aware STGNN* mới nhất (2024–2026) trên `PEMS04/08` để làm nổi bật điểm khác biệt của hệ chỉ báo tiền tắc nghẽn minh bạch. | **FD-Informed STGCN (Liu et al., 2024)**; **CAP-STGCN (MDPI, 2026)**; **DCSD-Net (IET ITS, 2026)** |

---

## 2. Tóm tắt Điểm Cốt lõi các Nhóm Bài báo Đã Chốt

### Nhóm A: Hệ thống PeMS, Chuẩn hóa & Làm sạch (`NB01 – NB02`)
1. **Chen et al. (2001)**, **Guo et al. (2019 — ASTGCN)**, **Song et al. (2020 — STSGCN):** Nguồn gốc vòng từ 5 phút Caltrans PeMS và bộ 4 benchmark `PEMS03/04/07/08` (chỉ `PEMS04/08` đủ 3 biến).
2. **Chen et al. (2003):** Chẩn đoán `4` chỉ số ngày ($S_1 \dots S_4$) và phục hồi thuần bằng hồi quy tuyến tính cặp trạm lân cận (`src/cleaning.py`, `NB02`, đạt `0` lỗi trên `PEMS08` & `PEMS04`).

### Nhóm B: Biểu đồ Cơ bản, Chuyển pha & Hệ Đặc trưng (`NB03` Phần 2–3)
3. **FHWA (23 CFR Part 490)**, **Chen et al. (2004)**, **Choe et al. (2002) & HCM**, **Cassidy & Bertini (1999):** Quy chuẩn `< 60%` vận tốc tự do (`40,26 mph` trên `PEMS08`, tự khớp mốc `40 mph`), dải bão hòa `Occupancy 11%–15%` (`o_sat = 13,25%`), và *Capacity Drop* (`-13,7%`) chứng minh loại `Flow`.
4. **Kerner (2004)**, **Chattopadhyay & Gupta (RSOS, 2026)**, **Payne & Tignor (1978)**, **Persaud & Hall (1989)**, **STID (Shao et al., CIKM 2022):** Nền tảng cho Pha chuyển tiếp và đủ `4` nhóm đặc trưng (`14` đầu vào): dao động khử dốc (`occ_rstd, spd_rstd, flow_rstd`), liên kết đa biến (`occ_spd_corr, flow_occ_decouple`), độ dốc có dấu (`occ_slope, spd_slope`) và `4` chu kỳ `sin/cos` (`tod/dow`).

### Nhóm C: Đồ thị Tĩnh vs. Động & Thiết lập Mô hình RQ2 (`NB03` Phần 4, `NB04 – NB05`)
5. **Li et al. (PNAS, 2015)**, **DCRNN (Li et al., 2018)**, **PDFormer (Jiang et al., AAAI 2023):** Loại trừ `Betweenness` tĩnh (`p = 0,3966`), chứng minh đồ thị động 2 chiều `In + Out` và lý giải mức tăng nổi bật ở tầm `15–30p` qua độ trễ lan truyền không gian (*propagation delay*).
6. **LibCity (2021)** & **BasicTS (2025):** Chuẩn hóa chia thời gian `60 / 20 / 20`, cửa sổ `12 → 12` bước trên `PEMS04/08`.
7. **GraphSAGE (Hamilton et al., 2017)**, **GAT (2018)**, **ST-GAT (2019)**, **GATv2 (2022):** Biện luận thiết kế kiến trúc cảm ứng chia sẻ trọng số trên cạnh (*inductive*, áp dụng trực tiếp được sang cả `PEMS08` 170 nút và `PEMS04` 307 nút) cùng trọng số chú ý động $\alpha_{ij}(t)$.

### Nhóm D: Đối chiếu Văn học Mới Nhất (*Related Work* Bài báo IEEE)
8. **FD-Informed STGCN (Liu et al., *ESWA* 2024)**, **CAP-STGCN (MDPI, 2026)**, **DCSD-Net (*IET ITS*, 2026):** Các công trình gần đây trên `PEMS04/08` tích hợp Biểu đồ cơ bản / phân loại mật độ vào STGNN — giúp định vị đóng góp riêng của đề tài: trích xuất trực tiếp `7` chỉ báo tiền tắc nghẽn minh bạch (không hộp đen) và kiểm chứng định lượng mức cải thiện + đóng góp từng nhóm đặc trưng trên `ST-GAT` đa bản đồ.

---

## 3. Danh mục `22` Tài liệu Tham khảo Chuẩn IEEE

1. C. Chen, K. Petty, A. Skabardonis, P. Varaiya, and Z. Jia, "Freeway performance measurement system: Mining loop detector data," *Transp. Res. Rec.*, vol. 1748, no. 1, pp. 96–102, 2001. DOI: `10.3141/1748-12`.
2. S. Guo, Y. Lin, N. Feng, C. Song, and H. Wan, "Attention based spatial-temporal graph convolutional networks for traffic flow forecasting," in *Proc. AAAI*, vol. 33, no. 01, pp. 922–929, 2019.
3. C. Song, Y. Lin, S. Guo, and H. Wan, "Spatial-temporal synchronous graph convolutional networks: A new framework for spatial-temporal network data forecasting," in *Proc. AAAI*, vol. 34, no. 04, pp. 914–921, 2020.
4. C. Chen, J. Kwon, J. Rice, A. Skabardonis, and P. Varaiya, "Detecting errors and imputing missing data for single-loop surveillance systems," *Transp. Res. Rec.*, vol. 1855, no. 1, pp. 160–167, 2003.
5. Federal Highway Administration (FHWA), "National Performance Management Measures," *Code of Federal Regulations*, Title 23, Part 490 (23 CFR Part 490), U.S. Dept. of Transportation.
6. C. Chen, A. Skabardonis, and P. Varaiya, "Systematic identification of freeway bottlenecks," *Transp. Res. Rec.*, vol. 1867, pp. 46–52, 2004. DOI: `10.3141/1867-06`.
7. T. Choe, A. Skabardonis, and P. Varaiya, "Freeway performance measurement system (PeMS): An operational analysis tool," in *Proc. 81st Annu. Meeting Transp. Res. Board*, 2002.
8. M. J. Cassidy and R. L. Bertini, "Some traffic features at freeway bottlenecks," *Transp. Res. Part B: Methodol.*, vol. 33, no. 1, pp. 25–42, 1999.
9. B. S. Kerner, *The Physics of Traffic: Empirical Freeway Pattern Features, Engineering Applications, and Theory*. Berlin, Heidelberg: Springer, 2004.
10. S. N. Chattopadhyay and A. K. Gupta, "Do multivariate measures anticipate traffic congestion?," *Royal Society Open Science*, vol. 13, no. 5, 252185, 2026. DOI: `10.1098/rsos.252185`.
11. H. J. Payne and S. C. Tignor, "Freeway incident-detection algorithms based on decision trees with states," *Transp. Res. Rec.*, no. 682, pp. 30–37, 1978.
12. B. N. Persaud and F. L. Hall, "Catastrophe theory and patterns in 30-second freeway traffic data — Implications for incident detection," *Transp. Res. Part A*, vol. 23, no. 2, pp. 103–113, 1989.
13. Z. Shao, Z. Zhang, F. Wang, W. Wei, and Y. Xu, "Spatial-temporal identity: A simple yet effective baseline for multivariate time series forecasting," in *Proc. ACM CIKM*, pp. 4454–4458, 2022.
14. D. Li, B. Fu, Y. Wang, G. Lu, Y. Berezin, H. E. Stanley, and S. Havlin, "Percolation transition in dynamical traffic network with evolving critical bottlenecks," *Proc. Natl. Acad. Sci. (PNAS)*, vol. 112, no. 3, pp. 669–672, 2015.
15. Y. Li, R. Yu, C. Shahabi, and Y. Liu, "Diffusion convolutional recurrent neural network: Data-driven traffic forecasting," in *Proc. ICLR*, 2018.
16. J. Jiang, C. Han, W. X. Zhao, and J. Wang, "PDFormer: Propagation delay-aware dynamic long-range transformer for traffic flow prediction," in *Proc. AAAI*, vol. 37, no. 4, pp. 4365–4373, 2023.
17. J. Wang, J. Jiang, W. Jiang, C. Li, and W. X. Zhao, "LibCity: An open library for traffic prediction," in *Proc. ACM SIGSPATIAL*, pp. 145–148, 2021. DOI: `10.1145/3474717.3483923`.
18. Z. Shao et al., "Exploring progress in multivariate time series forecasting: Comprehensive benchmarking and heterogeneity analysis," *IEEE Trans. Knowl. Data Eng.*, vol. 37, no. 1, pp. 291–305, 2025.
19. W. L. Hamilton, R. Ying, and J. Leskovec, "Inductive representation learning on large graphs," in *Adv. Neural Inf. Process. Syst. (NIPS)*, vol. 30, 2017.
20. P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph attention networks," in *Proc. ICLR*, 2018; and C. Zhang, J. J. Q. Yu, and Y. Liu, "Spatial-temporal graph attention networks: A deep learning approach for traffic forecasting," *IEEE Access*, vol. 7, pp. 166246–166256, 2019.
21. S. Brody, U. Alon, and E. Yahav, "How attentive are graph attention networks?," in *Proc. ICLR*, 2022.
22. Z. Liu, F. Ding, Y. Dai, L. Li, T. Chen, and H. Tan, "Spatial-temporal graph convolution network model with traffic fundamental diagram information informed for network traffic flow prediction," *Expert Syst. Appl.*, vol. 249, 123543, 2024 *(kèm CAP-STGCN, MDPI 2026 & DCSD-Net, IET ITS 2026)*.
