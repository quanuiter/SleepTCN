# Bảng nối kết luận với bằng chứng: bản sửa chuẩn bị BSPC

## Bổ sung 11-09-2026

- Không diễn giải Holm p = 1.000 thành tương đương, P/N dư thừa hoặc cơ chế TCN đã được chứng minh.
  Ablation là Sleep-EDF/E1, giữ C và thay P/N bằng trung bình training rồi retrain.
- Silhouette không phải mật độ cụm; không suy quan hệ đánh đổi giữa silhouette và tốc độ.
- Hình trade-off chính chỉ gồm E0/E3/E6 seed 42. Không trộn SHHS E4 seed 123 vào ranking seed 42.
- E4 thực sự có benchmark median 3.5739345 ms trong `gate6_latency_fold00_seed42.json`;
  việc bỏ khỏi hình primary không có nghĩa thiếu số đo.
- Các xác nhận ethics, CRediT, nguồn hình và quy định upload còn ở
  `BSPC_SUBMISSION_CHECKLIST_20260911.md`; không xem khai báo trong bài là kiểm chứng độc lập.

Cập nhật 06-09-2026. Đọc cùng `BSPC_REVISION_LOG.md`, `ARTIFACT_INDEX.md` và
`BSPC_FINAL_READINESS_REPORT.md`. Đường dẫn dưới đây tương đối với repository root.
Đây là bản đồ kiểm tra câu chữ, không phải một phân tích thống kê mới hay đăng ký trước công khai.

## Các kết luận được giữ và giới hạn phải đi kèm

| Kết luận được phép viết | Bằng chứng trực tiếp | Giới hạn bắt buộc khi diễn giải |
|---|---|---|
| E1 thay cấu hình mô hình chuỗi và lịch huấn luyện, dùng lại encoder/cache E0 | `configs/experiments_v2.json`; `src/sleeptcn/experiment.py`; bảng training trong supplement | Không phải phép cô lập BiLSTM so với TCN: learning rate, batch size, số epoch tối đa và patience khác nhau |
| E2 thay gói E1 C/P/N 75 chiều bằng ResNet-1D 128 chiều chỉ dùng epoch hiện tại | Config và mã encoder/feature/training; Supplementary Methods S1 | Đồng thời thay encoder, ngữ cảnh tường minh và cách huấn luyện/chọn encoder; không gọi là encoder-only |
| E3 có EDF pooled macro-F1 0,7904, cao nhất ở seed 42 | `runs/v2/analysis/gate5_paired_results_seed42.json` | Không đồng nghĩa tốt nhất ở mọi seed: E4 nhỉnh hơn E3 ở seed 123 |
| EDF E3−E2 có CI gộp dương nhưng Wilcoxon không đạt ý nghĩa | Gate-5 seed42: Δ 0,006962, CI [0,000305; 0,014520], p Holm 0,8989, thắng/thua 37/41 | Hai phép thống kê xét hai đại lượng khác nhau. Pooled macro-F1 không phải trung bình có trọng số của F1 từng người; không suy ai tạo phần tăng gộp chỉ từ sự bất đồng này |
| EDF E3−E6 có bằng chứng chính mạnh nhất trong bốn đối chiếu | Gate-5 seed42: Δ 0,021319, CI [0,012179; 0,030698], p Holm 0,001185; seed123: Δ 0,010249, p Holm 0,1313 | Hướng gộp giữ ở hai seed, nhưng ý nghĩa Wilcoxon sau Holm không giữ ở seed 123; không gộp p-value hoặc xem hai seed là phân phối khởi tạo |
| Pipeline ResNet-1D–TCN có speed-up khoảng 3,76 lần trong forward benchmark | `runs/v2/analysis/gate6_latency_fold00_seed42.json`; bảng vận hành trong manuscript/supplement | V100, một shape đầu vào; không gồm I/O/tiền xử lý. Tham số cao hơn 4,37 lần, peak memory cao hơn 28,4%; không gọi mô hình nhỏ hơn hoặc latency quyết định online |
| Thời gian huấn luyện và validation quan sát được ngắn hơn khoảng 10,2–12,2 lần | Bảng thời gian chiến dịch seed123 trong supplement và `docs/GATE6_FINAL_RESULTS.md` | Các training recipe/early-stopping khác nhau; là chi phí workflow đã chạy, không phải hằng số độ phức tạp kiến trúc |
| Trên SHHS1, E3 cao hơn E0 và E6 theo chỉ số chính cấp đối tượng | Phân tích primary SHHS SHA `83aa53fe...943df`; Δ lần lượt 0,0412 và 0,0274, hai CI dương và Wilcoxon–Holm đạt | 180 người đã khóa; weights học từ EDF. E6 còn dùng thống kê từ toàn bản ghi đích không nhãn; không gộp với family phân tích mở rộng |
| E3−E2 là chênh lệch cấu hình lớn nhất đã quan sát trong phân tích hậu nghiệm SHHS | `Reports/SHHS_E3_E2_PAIRED_AUDIT.json`: Δ subject mean 0,04750, CI [0,03724; 0,05792], 147/180 người cải thiện | Không nâng thành quy luật “tiền xử lý luôn quan trọng hơn kiến trúc”; không quy toàn bộ cho riêng lọc hoặc riêng đổi thang |
| E4 cung cấp thêm bằng chứng về vai trò của chế độ tín hiệu | `Reports/SHHS_SEED123_E4_EXTENSION.md`; phân tích extension SHA `8563eefe...4792fe`: E4 subject mean 0,5732, E3 0,5634 | Cùng cohort đã mở, seed123 và family Holm riêng; không phải test cohort mới, equivalence hay non-inferiority |
| N3 bị bỏ sót thành N2 tái diễn ở E0 và E3 dù E3 có điểm tổng thể cao hơn | `Reports/SHHS_E0_E3_E6_N3_AUDIT.json`; 22.806 N3 thật, E0/E3 có 16.480/16.674 N3→N2, recall 0,2610/0,2582 | E3 precision N3 cao hơn (0,9440/0,9007), nên viết “chưa khắc phục bỏ sót N3”, không phủ nhận mọi cải thiện chỉ số N3 |
| Pipeline E6 đã thử không sửa được thiếu N3 | Cùng audit: recall E6 0,2005; nguồn preprocessing ước lượng mean/std trên toàn recording sau lọc/cắt biên độ, trước cửa sổ đánh giá | Label-free nhưng transductive. E6 được huấn luyện với pipeline tương ứng; không phải chỉ thay scale của E3 đóng băng. Không loại trừ biên độ/montage hoặc mọi phương pháp không nhãn |
| N3 thấp cả ở vùng ổn định, không chỉ lân cận đổi nhãn | Region JSON seed42 và hai summary seed123 trong `ARTIFACT_INDEX.md`; Supplementary Table S16 | Vùng lấy từ nhãn tham chiếu, chẩn đoán ngoại tuyến/hậu nghiệm; thành phần lớp khác nhau. Không phải phát hiện chuyển pha sinh lý độc lập |
| Kênh N3→N2 có quy mô số học lớn trong oracle E3 | Ma trận E3 và phép chuyển số đếm: macro-F1 oracle 0,7444, tương ứng 74,5% chênh lệch điểm gộp quan sát | Oracle dùng nhãn thật; không phải hiệu năng có thể đạt hoặc bound của thuật toán. Gap EDF OOF–SHHS ensemble không cô lập nguyên nhân domain shift. Các oracle không cộng tuyến tính |
| Ablation không thấy lợi ích tăng thêm có ý nghĩa của nhóm P/N tường minh trong thiết kế đã dùng | `runs/v2/gate8/analysis_seed42.json`: ba p Holm 1,000 | Có huấn luyện lại, giữ 75 chiều và thay nhóm bằng trung bình train. Không chứng minh ngữ cảnh thời gian vô dụng hay tương đương giữa mô hình |
| Silhouette E2 thấp hơn E1 ở các fold trong pipeline phân tích | Aggregate Gate6/Gate8 được pin trong `scripts/build_english_supplement_figures.py` | Không xếp hạng năng lực dự đoán của embedding, không suy cơ chế sinh lý từ khoảng cách Euclid |
| Các artifact dự đoán chính còn giữ nguyên và khớp hash/count đã lưu | `Reports/ARTIFACT_INDEX.md`: 540/540 primary ensemble hashes và confusion matrices khớp; nguồn seed123 được kiểm tra riêng | Integrity/provenance không bằng independent replay. Chưa kiểm toàn chuỗi raw input–preprocessing–checkpoint–fold prediction hoặc tái sinh predictions |

## Đơn vị và cách đọc thống kê

- EDF headline dùng macro-F1 tính từ tổng confusion matrices; bootstrap lấy lại mẫu theo đối tượng.
- SHHS primary dùng trung bình macro-F1 của từng đối tượng; mỗi người có trọng số ngang nhau.
- Các CI là khoảng biên 95%; p Holm được hiệu chỉnh trong đúng family đã mô tả, không làm CI trở thành simultaneous CI.
- Δ 0,0412 điểm trên thang 0–1 tương đương 4,12 điểm phần trăm, không phải tăng tương đối 4,12%.
- Ký hiệu N3→N2 là nhãn thật N3 bị dự đoán thành N2, không phải một chuyển giai đoạn theo thời gian.
- Vùng near = hợp {j−1, j, j+1}, j là epoch đầu nhãn mới; stable cách ít nhất ba epoch với cả j−1 và j của mọi thay đổi nhãn. Phần còn lại không tự gộp vào stable.

## Quy tắc cho lần chỉnh tiếp

1. Sửa claim trong ENG, VI và bản báo cáo đầy đủ cùng một lần; không thay số JSON để làm câu văn đúng.
2. Giữ nhãn primary/secondary/post-hoc/sensitivity, cohort, seed, estimand và family thống kê.
3. Khi thêm số chưa có trong artifact, dừng và lập kế hoạch tính toán; không nội suy CI hoặc p-value từ số đã làm tròn.
4. Không cần thêm adaptation chỉ để bài có một “remedy”: đây là nghiên cứu đánh giá pipeline. Nếu muốn claim hiệu quả khắc phục, phải có thực nghiệm riêng được tác giả đồng ý và test độc lập với tập chọn biện pháp.
5. Sau sửa nguồn, build, kiểm PDF và cập nhật publication manifests; không đổi hash kỳ vọng của run lịch sử.
