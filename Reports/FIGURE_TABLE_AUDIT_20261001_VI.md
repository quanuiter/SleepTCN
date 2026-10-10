# Rà soát hình và bảng của paper BSPC

Ngày rà soát: 01/10/2026. Phạm vi: bản tiếng Anh `Reports/paper_en`, bản dịch đọc hiểu `Reports/paper_vi_translation` và supplementary material hiện hành. Đây là báo cáo biên tập và bộ hình xem trước; **chưa chèn, thay hoặc xóa hình/bảng trong manuscript, chưa thay PDF hay gói nguồn nộp bài**.

## 1. Kết luận và phương án nên chọn

Bài không thiếu số liệu; điểm yếu hiện tại là người đọc phải đi qua nhiều bảng để hình dung thiết kế so sánh và lỗi theo từng giai đoạn. Không cần tăng số hình chỉ để bài có vẻ dày hơn.

Phương án ưu tiên:

1. **Thay Figure 1** bằng sơ đồ các cấu hình song song và thiết kế đánh giá (preview D). Sơ đồ E3 hiện tại không sai về quy trình, nhưng chưa phản ánh đầy đủ một bài đánh giá các lựa chọn thiết kế.
2. **Giữ Figure 2 về tốc độ và hiệu năng**, chỉnh kiểu trình bày, cỡ chữ, màu và caption. Không cần thêm một biểu đồ thời gian/throughput khác để lặp lại kết quả này.
3. **Thêm một Figure 3 về F1 từng lớp và dự đoán trên N3** (preview A). Đây là hình mới có ích nhất: nối kết quả trong miền, chuyển quần thể và lỗi N3 mà điểm tổng hợp không thể hiện rõ.
4. **Preview B (ma trận nhầm lẫn chuẩn hóa)** phù hợp supplementary. Có thể đưa vào main nếu chọn nó thay phần phân tích theo lớp; không bắt buộc đưa cả A và B vào main.
5. **Preview C (hiệu ứng bắt cặp và CI)** là lựa chọn bổ sung, không phải hình bắt buộc. Dùng khi muốn người đọc nhìn được độ lớn và bất định của khác biệt; ưu tiên supplementary nếu vẫn giữ đầy đủ Tables 4 và 7 trong main.

Như vậy, phương án gọn là **3 hình chính**: thiết kế nghiên cứu, tốc độ–hiệu năng, kết quả theo lớp/lỗi N3. Con số này là đề xuất biên tập cho bài hiện tại, không phải quy định số hình của BSPC.

## 2. Đối chiếu với paper liên quan

Đã đọc và xem trang hình thực tế của ba nghiên cứu sau, không chỉ đọc abstract:

| Nghiên cứu | Những hình/bảng đã đối chiếu | Điều phù hợp để học cho bài này |
| --- | --- | --- |
| [ZleepAnlystNet, Scientific Reports, 2024](https://www.nature.com/articles/s41598-024-60796-y) | Figs. 1–2: kiến trúc và huấn luyện; Fig. 3: confusion matrices, chỉ số theo lớp và lượng dữ liệu; Fig. 4: hypnogram; Tables 1–2: đối chiếu và ablation | Tách sơ đồ phương pháp khỏi kết quả; phân tích từng lớp cần được nhìn thấy. Không cần sao chép khung màu dày hay bảng số nhét vào hình. |
| [U-Sleep’s resilience to AASM guidelines, npj Digital Medicine, 2023](https://www.nature.com/articles/s41746-023-00784-0) | Fig. 1: ba thành phần mạng; Tables 1–4: dữ liệu, chuyển miền và điều kiện huấn luyện; confusion matrices/per-stage results được dẫn sang supplement | Không phải mọi chẩn đoán đều phải nằm ở main. Kiến trúc có thể biểu diễn ở mức khối; layer specification và kết quả mở rộng giữ trong bảng bổ sung. Mean ± SD trong bài đó không được mặc nhiên coi là CI. |
| [SLEEPYLAND, bản tác giả arXiv v2](https://arxiv.org/abs/2506.08574v2) | Fig. 1: phân bố có điều kiện theo tuổi; Fig. 2: boxplots theo bản ghi/chỉ số; Fig. 3: disagreement và vùng chuyển nhãn; các bảng đánh giá nhiều mô hình | Biểu đồ phải gắn với câu hỏi cụ thể và đơn vị phân tích rõ ràng. Hình phân bố cần dữ liệu từng bản ghi/người; không thể dựng boxplot từ một điểm macro-F1 tổng hợp. |

SLEEPYLAND có [bản xuất bản trên npj Digital Medicine](https://www.nature.com/articles/s41746-025-02237-2). Việc rà soát layout ở đây dùng PDF **arXiv v2 ngày 11/06/2025**, không khẳng định số hình/layout hoàn toàn giống bản xuất bản. PDF có 85 trang gồm phần phụ lục dài.

Các trang đầy đủ của hai paper BSPC đã thử truy cập trong phiên này không mở được; SleepInceptionNet/JMIR cũng trả trang chặn thay vì PDF. Vì vậy, không đưa chúng vào nhóm đã kiểm tra layout thực tế. Không xem việc chặn truy cập là đánh giá chất lượng của paper.

Đây là mẫu đối chiếu có chủ đích gồm ba paper gần về bài toán/thiết kế, không phải một khảo sát hệ thống về tần suất sử dụng các loại hình trong toàn ngành.

## 3. Rà soát các hình hiện hành

### Figure 1 — pipeline E3

Điểm tốt: vector sắc nét, thứ tự xử lý đúng, có đơn vị µV, 100 Hz, 30 s, 128 chiều và TCN không nhân quả. Không vẽ giả tín hiệu đo.

Điểm cần cải thiện:

- Chỉ mô tả E3, trong khi tiêu đề và câu hỏi nghiên cứu xoay quanh so sánh cấu hình.
- Caption gọi “proposed framework” dễ khiến người đọc nghĩ bài đề xuất kiến trúc mới; sơ đồ so sánh sẽ phù hợp trọng tâm đánh giá hơn.
- Không cho người đọc thấy rõ E1 dùng lại encoder E0, khác biệt C/P/N và current-epoch, hoặc cách nối đánh giá 10-fold với SHHS1.
- Clipping được vẽ như một bước có tác động, nhưng các audit đã lưu ghi nhận không có mẫu bị clip. Cần nói rõ trong caption, không minh họa tác động clipping bằng một waveform giả.

Đề xuất thay bằng D. Chi tiết band-pass, ngưỡng và layer vẫn giữ ở Methods/Supplementary Tables S1–S3, không phải bỏ đi.

### Figure 2 — tốc độ truyền xuôi và macro-F1

Giữ: hình trả lời quan hệ giữa hiệu năng và chi phí vận hành, không chỉ trang trí.

Chỉnh khi được duyệt:

- Giữ hai panel và ghi rõ **pooled** Sleep-EDF versus **subject-mean** SHHS1; không nối các điểm qua hai quần thể như cùng một đại lượng.
- Rút tiêu đề lớn trong hình; để điều kiện benchmark và seed trong caption. Nhãn panel a/b ngắn gọn.
- Thống nhất màu/marker E0–E3–E6 với các hình mới. Không chỉ dựa vào đỏ/xanh lá để phân biệt.
- Trục F1 hiện tại được thu hẹp. Đây không tự động là sai với scatter plot, nhưng phải hiện rõ khoảng trục; không đổi thành cột trên trục cắt vì sẽ làm mức tăng trông lớn hơn.
- Không thêm error bars nếu chưa có CI đúng của đại lượng biểu diễn. Không lấy SD của 10 folds hoặc hai seed làm CI tùy tiện.
- E3/E6 có cùng kiến trúc và tốc độ xấp xỉ nhau. Không kết luận chênh lệch timing rất nhỏ giữa chúng là lợi ích tiền xử lý; benchmark loại trừ tiền xử lý/I/O.
- Không đưa bộ nhớ vào kích thước bubble hoặc trục thứ ba. Giữ số tuyệt đối 58.81/75.54 MiB trong bảng; không nâng nó thành một nhược điểm vận hành lớn khi chưa có bằng chứng.

### Supplementary Figure S1(a) — silhouette

Nên giữ ở supplement. Kết nối hai điểm của cùng fold là cách trình bày hợp lý; các đường nối không phải diễn biến thời gian. Cần tiếp tục ghi rõ chuẩn hóa/PCA 20 thành phần và đây là kiểm tra mô tả hình học đặc trưng. Đặc trưng softmax 75 chiều và biểu diễn ResNet 128 chiều khác bản chất; silhouette thấp hơn không chứng minh encoder kém hơn về dự đoán. Không thêm t-SNE vào main như bằng chứng về chất lượng mô hình.

### Supplementary Figure S1(b) — ablation ngữ cảnh

Giữ: forest plot phù hợp hiệu ứng bắt cặp có CI. Chưa phát hiện lợi ích không đồng nghĩa hai điều kiện tương đương. Cần ghi rõ boundary được định nghĩa bằng nhãn tham chiếu và TCN vẫn dùng ngữ cảnh ngay cả khi bỏ nhóm P/N.

Hai panel silhouette/context trả lời hai câu hỏi khác nhau. Khi thêm hình supplementary khác, nên cân nhắc tách thành hai figure có caption riêng và cỡ chữ lớn hơn; không bắt buộc tách nếu muốn giữ supplement ngắn. Hiện chúng được đặt nửa chiều rộng trang, nên nhãn khá nhỏ.

## 4. Các hình xem trước và mục đích khoa học

Tất cả ở `Reports/output/figure_preview_20261001/`, định dạng PNG để xem và SVG vector có text chỉnh sửa được. Chưa xuất file PDF nộp bài, chưa chèn vào manuscript.

### A — F1 từng lớp và dự đoán N3: ưu tiên thêm vào main

![Preview A](D:/SleepTCN/Reports/output/figure_preview_20261001/A_stage_performance_and_n3.png)

**Câu hỏi:** những giai đoạn nào vẫn khó khi chuyển quần thể, và lợi ích tổng hợp của E3 có khắc phục bỏ sót N3 không?

- Panel a/b dùng cùng thứ tự lớp, màu/marker và trục 0–100%. Cả hai là **pooled class F1**, không trộn pooled F1 với trung bình theo người.
- Panel c dùng chung 22,806 epoch N3 cho cả ba cấu hình. Tỷ lệ nhận đúng N3 là 26.1%, 25.8%, 20.1%; tỷ lệ nhầm sang N2 là 72.3%, 73.1%, 77.9%.
- Nền xám đánh dấu N3, không tăng kích thước điểm E3 để ám chỉ ưu thế.
- Không kèm kiểm định từng lớp chưa được thực hiện. So sánh giữa hai quần thể là mô tả, vì mẫu người, montage và cách dự đoán out-of-fold/ensemble đều khác.

Vị trí phù hợp: mở đầu phần “Cross-model errors and their temporal distribution”, sau kết quả tổng hợp SHHS1. Nếu thêm A, có thể chuyển bảng N3 chi tiết sang supplement để tránh lặp, nhưng phải giữ precision/recall/F1 và count gốc ở đó.

### B — confusion matrices E3 chuẩn hóa theo hàng: ưu tiên supplement

![Preview B](D:/SleepTCN/Reports/output/figure_preview_20261001/B_e3_normalised_confusion.png)

**Câu hỏi:** phân bố lỗi theo nhãn tham chiếu thay đổi thế nào giữa hai mẫu?

Hàng là reference, cột là prediction; cùng thang màu 0–100%, cùng thứ tự W/N1/N2/N3/REM; số hỗ trợ mỗi hàng được ghi cạnh nhãn. Màu dựa trên tỷ lệ, không dựa trên raw counts vốn chịu ảnh hưởng độ phổ biến lớp. Counts vẫn được giữ ở Tables S5/S13.

Đặc biệt, N3→N2 ở E3 tăng từ 18.2% lên 73.1% số epoch có nhãn N3. N2→REM trên SHHS1 là 15.2% số epoch N2. “Dạng lỗi lớn thứ hai” trong phần chữ là theo **count gốc 11,605**, không phải theo thứ hạng tỷ lệ chuẩn hóa giữa các hàng. Các ô 0.0 là giá trị làm tròn, không mặc nhiên là zero count.

Hình này chỉ hiển thị E3; **không dùng riêng B để kết luận lỗi N3 lặp lại ở mọi mô hình**. A/Table N3 là bằng chứng so sánh chéo E0/E3/E6.

### C — hiệu ứng bắt cặp và CI: tùy chọn

![Preview C](D:/SleepTCN/Reports/output/figure_preview_20261001/C_primary_paired_effects.png)

**Câu hỏi:** khác biệt đo được lớn đến đâu, bất định thế nào, và kết quả kiểm định theo người có đồng thuận với điểm tổng hợp không?

Panel EDF chỉ có bốn contrast xác định trước; panel SHHS1 chỉ có hai contrast chính đã khóa. Không nhét E3–E0 hậu nghiệm của EDF hay E3–E2/E4 hậu nghiệm của SHHS vào nhóm kết quả chính.

Đơn vị trục là điểm phần trăm: 0.021319 macro-F1 tương ứng 2.1319 điểm phần trăm, không phải tăng tương đối 2.1319%. CI là bootstrap bắt cặp theo cụm người, không phải CI độc lập của hai mô hình đem trừ nhau. CI là marginal, chưa hiệu chỉnh đồng thời.

Trên EDF, CI mô tả chênh lệch pooled macro-F1; Wilcoxon–Holm kiểm tra khác biệt ở cấp người. Vì vậy E3–E2 có CI dương nhưng p Holm = 0.8989 không phải lỗi vẽ. Marker rỗng/đặc theo kết quả Holm và caption cần giải thích hai đại lượng khác nhau. Mỗi panel có họ Holm riêng.

EDF dùng số đầy đủ từ JSON. SHHS dùng số **đã làm tròn như Table 7**, được kiểm tra khớp nguồn; không tạo độ chính xác thêm. Không suy ra phân bố người từ các khoảng này.

### D — sơ đồ thiết kế nghiên cứu: đề xuất thay Figure 1

![Preview D](D:/SleepTCN/Reports/output/figure_preview_20261001/D_study_design_overview.png)

**Câu hỏi:** đang so sánh những gì, phần nào dùng chung và dữ liệu đích được dùng ra sao?

Các nhánh đặt song song, không vẽ E0→E1→E2 như các phiên bản ngày càng tốt. Phân biệt feature/context package và model chuỗi; ghi E1 dùng lại encoder E0 và thiết lập huấn luyện không hoàn toàn giống nhau. E2/E3/E4/E6 cùng khối ResNet–TCN nhưng được fit cho điều kiện đầu vào tương ứng.

Phần đánh giá hiển thị 8 train/1 validation/1 test **folds**, không phải tỷ lệ người chính xác 80/10/10. Cùng người/cả hai đêm cùng fold. SHHS1 primary có 180 người, E0/E3/E6, dùng trung bình xác suất từ 10 model nguồn trước argmax, không cập nhật trọng số trên đích. E6 vẫn dùng thống kê toàn bản ghi đích không nhãn, nên không gọi nó purely inductive. E4 trên SHHS chỉ là extension thứ cấp với seed 123, không phải một kết quả primary seed-42 mới.

## 5. Rà soát bảng và giảm lặp có kiểm soát

Các bảng đang dùng booktabs, không tô nền rực hay kẻ ô dày; nên giữ hướng này. Số liệu quan trọng vẫn cần bảng để tra cứu, không thay tất cả bằng biểu đồ.

| Bảng chính hiện tại | Đề xuất | Lý do |
| --- | --- | --- |
| 1: phân bố lớp | Giữ | Cần để hiểu support/class prior; không cần thêm pie chart cùng số liệu. |
| 2: cấu hình | Giữ, dù dùng sơ đồ D | Bảng là định nghĩa chính xác của mã E0–E6; sơ đồ cho cái nhìn tổng quát. E5 phải tiếp tục ghi audit only. |
| 3: EDF performance | Giữ | Báo cáo đầy đủ sáu cấu hình và các metric, không chỉ ba cấu hình đưa lên hình. Bold phải có nghĩa “lớn nhất trong bảng”, không mặc nhiên có ý nghĩa thống kê. |
| 4: EDF paired comparisons | Có thể rút còn contrast, Δ, CI, p Holm, W/T/L | Chuyển sign-test/sign dominance và dòng post-hoc sang supplement nếu cần gọn; không xóa evidence. Chỉ bold sign-test p trong một hàng cũng dễ khiến người đọc lẫn với family đã hiệu chỉnh. |
| 5: benchmark | Giữ | Không cần thêm bar chart memory. Có thể bỏ cột throughput khi đã có latency/speed-up vì thông lượng được dẫn xuất từ cùng input 100 epoch. |
| 6: SHHS performance | Giữ | Phải có cả subject-mean và pooled metrics, xác định primary estimand. |
| 7: SHHS paired comparisons | Có thể gộp với Table 6 thành panel A/B | Liên kết kết quả tổng hợp và contrast chính, giảm số float; tuyệt đối không tính paired CI bằng cách trừ marginal CIs. |
| 8: cross-model N3 | Giữ nếu chưa thêm A; nếu thêm A có thể chuyển supplement | A thể hiện quan hệ trực quan, bảng vẫn giữ precision/count chính xác để kiểm chứng. |

Nếu chọn toàn bộ bước rút/gộp này, main còn 6 bảng thay vì 8; đây là phương án tùy chọn, chưa áp dụng.

Supplement:

- **S1–S3:** layer, training và preprocessing specifications — giữ; không biến các chi tiết số thành một sơ đồ mạng khổng lồ.
- **S4/S5:** per-class EDF và confusion counts — giữ làm nguồn kiểm chứng cho hình.
- **S6/S7:** seed sensitivity — giữ. S7 là bảng chật nhất; khi biên tập nên tách rõ cột seed, Δ, CI và p thay vì nhét nhiều giá trị vào một ô. Hai seed không đại diện phân bố mọi khởi tạo.
- **S8:** runtime huấn luyện — giữ, ghi rõ E1 chỉ sequence training và recipes/stopping khác. Không thêm một cột tốc độ màu đậm để ngụ ý kiến trúc gây ra toàn bộ mức tăng tốc.
- **S9–S11:** SHHS extension/secondary/post-hoc — giữ ở supplement; không hòa vào các hàng primary của main.
- **S12–S14:** per-class/transfer/confusion — giữ. B chỉ chuyển phần tỷ lệ thành hình, không thay counts.
- **S15:** oracle correction — chỉ supplement. Không vẽ waterfall “phần trăm domain gap được giải quyết”, vì sửa ô bằng nhãn thật không phải cải thiện khả thi và các hiệu ứng phi tuyến không cộng được.
- **S16:** boundary/stable diagnostics — giữ. Không vẽ biểu đồ hai nhóm rồi bỏ 21,539/21,073 epoch còn lại mà không giải thích.
- **S17:** context ablation — giữ cạnh hình CI tương ứng.

## 6. Những hình chưa nên thêm

| Ý tưởng | Quyết định và điều kiện |
| --- | --- |
| Hypnogram đẹp của một đêm | Chưa cần. Nếu thêm, chọn người gần median bằng quy tắc định trước, ghi ID ẩn danh/metric, hiển thị toàn đêm và inset cố định; không chọn ca tốt nhất. Phải lấy/đối chiếu dự đoán epoch thật trước. |
| t-SNE/UMAP “tách lớp tốt hơn” | Không đưa main; hình phụ thuộc projection/sampling và hai representation khác bản chất. Không thay thế đánh giá dự đoán. |
| Radar accuracy/F1/latency/parameters/memory | Không thêm: trộn đơn vị và cách chuẩn hóa tùy ý, khó thấy đánh đổi thực. |
| SOTA bar chart lấy điểm từ nhiều paper | Không thêm khi protocol/channel/subject splits khác; dễ tạo một xếp hạng không công bằng. |
| ROC/PR/calibration curves | Chỉ làm nếu có probability outputs và câu hỏi đánh giá rõ. Không suy từ confusion counts và không coi softmax outputs là đã calibrated. |
| Training curves một fold/seed | Không thêm chỉ để minh họa hội tụ. Nếu cần, dùng mọi fold và nêu điều kiện stopping khác, không so số epoch như cùng lượng công việc. |
| PSD/biên độ EEG hai quần thể | Có thể hữu ích cho nghiên cứu tiếp theo, nhưng cần thống kê tín hiệu thật, sampling rules, đơn vị và uncertainty; không dùng hình tùy chọn để suy ra nguyên nhân thất bại N3. |
| Waveform trước/sau clipping | Không tạo giả một ví dụ clipping rồi trình bày như dữ liệu đã quan sát; audit hiện tại clipping không kích hoạt. |
| Pie chart phân bố lớp/bubble memory | Không cần, bảng đã đủ và không phục vụ quan hệ khoa học mới. |

## 7. Quy cách thiết kế và kiểm tra trước khi chèn

Theo [hướng dẫn sizing của Elsevier](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing), chữ thường nên khoảng 7 pt ở **kích thước in cuối cùng**; kích thước phải xét sau khi thu hình. Đây là hướng dẫn chung, còn journal-specific instructions cần kiểm tra khi đóng gói cuối. [Artwork overview](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-overview) chấp nhận PDF/EPS cho vector.

Thiết kế ở đây:

- Canvas 18.034 cm; typography đã bù để đưa ở **đủ text width 16.4 cm** của manuscript hiện tại, chữ thường nhỏ nhất 7 pt. Không chèn các preview ở `0.9\textwidth` rồi nghĩ cỡ chữ vẫn không đổi.
- Màu mô hình E0 xanh dương, E3 cam sẫm, E6 tím; kèm marker tròn/kim cương/tam giác. Màu stacked bars mã hóa predicted stage, không mã hóa model; legend tách riêng.
- Nền trắng, grid mảnh, không 3D/gradient trang trí, cùng class order và thang số khi so sánh.
- Caption tách khỏi tiêu đề hình, định nghĩa seed/sample/estimand/CI/ensemble. Những giới hạn quan trọng được viết cụ thể, không dùng chú thích nội bộ kiểu “cần chứng minh thêm” trong hình.
- PNG 300 dpi chỉ là bản xem trước. Khi nộp sẽ dùng PDF vector có font nhúng; không tuyên bố PNG line art 300 dpi đã đạt yêu cầu final artwork.
- SVG giữ text chỉnh sửa được; khi chuyển PDF cần kiểm tra font nhúng và render lại toàn trang. Không dùng AI image generation để vẽ dữ liệu khoa học.

Đã kiểm tra: hash nguồn, số epoch, tổng support, class order, F1 từ confusion counts, cùng denominator N3, bounding boxes của chữ, và xem trực quan cả bốn preview sau chỉnh sửa. **Chưa kiểm tra bố cục của một manuscript chứa các hình mới**, vì chưa được duyệt chèn.

## 8. Caption dự kiến để duyệt

### A — English

Stage-specific performance and N3 predictions for the primary seed-42 configurations. (a,b) Pooled class F1 on Sleep-EDF (78 subjects; out-of-fold predictions) and the locked SHHS1 sample (180 participants; source-fold ensembles). Both panels use the same class order and scale. (c) Predicted labels for the same 22,806 reference-N3 SHHS1 epochs: correct N3, N2, and other stages (W/N1/REM). Cross-cohort differences are descriptive; class-level confidence intervals are not shown. All weights were trained on Sleep-EDF; E6 additionally uses complete-record unlabelled target statistics.

### A — Tiếng Việt

Hiệu năng theo giai đoạn và dự đoán N3 của các cấu hình chính với seed 42. (a,b) F1 từng lớp trên các epoch gộp chung của Sleep-EDF (78 người; dự đoán ngoài fold) và mẫu SHHS1 đã khóa (180 người; ensemble các model fold nguồn), với cùng thứ tự lớp và thang đo. (c) Nhãn dự đoán cho cùng 22.806 epoch SHHS1 có nhãn tham chiếu N3: N3 đúng, N2 và các giai đoạn khác (W/N1/REM). Khác biệt giữa hai quần thể được mô tả, không trình bày CI theo lớp. Mọi trọng số được huấn luyện trên Sleep-EDF; E6 còn dùng thống kê toàn bản ghi đích không nhãn.

### B — English

Row-normalised E3 confusion matrices for Sleep-EDF out-of-fold evaluation and locked SHHS1 ensemble evaluation (seed 42). Rows denote reference labels and columns predictions; entries are percentages of each reference class, with supports shown alongside the rows. Both panels use a fixed 0–100% colour scale. Displayed zeros may reflect rounding; exact counts are retained in Supplementary Tables S5 and S13. Cross-cohort differences are descriptive and do not isolate a cause of dataset shift.

### C — English

Primary paired macro-F1 contrasts at seed 42. Points and bars show observed differences and marginal 95% paired subject-cluster bootstrap intervals. Sleep-EDF intervals concern pooled macro-F1 across 78 subjects; SHHS1 intervals concern subject-mean macro-F1 across 180 participants. Holm-adjusted subject-level Wilcoxon p-values are shown for the four-test Sleep-EDF and two-test SHHS1 families separately. Filled markers indicate p < 0.05. On Sleep-EDF, the bootstrap estimand and Wilcoxon test differ; interval exclusion of zero does not imply Holm significance. SHHS1 values are displayed at the precision of Table 7.

### D — English

Signal-model configurations and evaluation design. (a) Parallel configuration families, distinguishing the 75-dimensional C/P/N CNN feature package, current-epoch 128-dimensional ResNet features, and BiLSTM/TCN sequence models. Signal operations precede epoch encoding. Shared subject partitions do not imply identical training recipes; model specifications and input variants are given in Supplementary Tables S1–S3. (b) Ten-fold subject-disjoint Sleep-EDF evaluation and locked SHHS1 testing using source-fold ensembles without target weight adaptation. E5 is an input-identity audit, not an independent performance condition. Clipping did not activate in the audited recordings. E6 uses complete-record unlabelled target statistics and is label-free transductive. The models operate offline with future context.

## 9. Nguồn và tái lập

Builder: `tmp/preview_bspc_figures_20261001.py`. Provenance số liệu và SHA256: `Reports/output/figure_preview_20261001/provenance.json`. Nguồn chính:

- `runs/v2/analysis/gate5_paired_results_seed42.json`: confusion counts/paired EDF results đầy đủ.
- `Reports/SHHS_E0_E3_E6_N3_AUDIT.json`: SHHS confusion counts và N3 audit.
- `Reports/paper_en/main.tex`: thiết kế và SHHS Table 7 ở độ chính xác hiện đang báo cáo.

Không thay protocol, prediction files, checkpoint, JSON kết quả, seed hay inference family. Không huấn luyện/đánh giá lại mô hình. Các thống kê suy luận trong C là kết quả đã báo cáo, không phải kiểm định mới.
