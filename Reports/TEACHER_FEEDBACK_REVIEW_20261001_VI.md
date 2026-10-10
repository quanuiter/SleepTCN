# Rà soát bản thảo, kế hoạch và góp ý của thầy Trí

Ngày 01-10-2026. Đây là bản review và phương án thực hiện được điều chỉnh, chưa phải kết quả của các thí nghiệm mới.

Phạm vi: paper BSPC tiếng Anh hiện hành, bản dịch tiếng Việt, supplement, giao thức EDF/SHHS, kế hoạch sửa cũ, các kết quả tổng hợp và mã huấn luyện/tiền xử lý liên quan. Không coi `Reports/paper` là bản dịch hiện hành; bản dịch hiện hành nằm ở `Reports/paper_vi_translation`. Lượt này không sửa manuscript hay mã huấn luyện, không chạy inference/training/bootstrap.

Nguồn chính được rà soát:

- [Paper](D:/SleepTCN/Reports/paper_en/main.tex), [supplement](D:/SleepTCN/Reports/paper_en/supplement.tex), [giao thức v2](D:/SleepTCN/docs/EXPERIMENT_PROTOCOL_V2.md), [kế hoạch trước nộp cũ](D:/SleepTCN/Reports/BSPC_PRE_SUBMISSION_AUDIT_AND_EXECUTION_PLAN_VI.md).
- [SHHS seed 123/E4](D:/SleepTCN/Reports/SHHS_SEED123_E4_EXTENSION.md), [bảng nối claim với bằng chứng](D:/SleepTCN/Reports/BSPC_CLAIM_EVIDENCE_LEDGER.md), [artifact index](D:/SleepTCN/Reports/ARTIFACT_INDEX.md).
- [Huấn luyện](D:/SleepTCN/src/sleeptcn/experiment.py), [mô hình](D:/SleepTCN/src/sleeptcn/models.py), [tiền xử lý](D:/SleepTCN/src/sleeptcn/preprocessing.py), [SHHS preprocessing](D:/SleepTCN/src/sleeptcn/shhs_preprocessing.py), [SHHS inference](D:/SleepTCN/src/sleeptcn/shhs_zero_shot.py).

Mã băm nguồn tại thời điểm review: main EN `6f357f31c8a23c7fc7499ba3be0374fc9321786dfb4dc16f6625dbba4f3a1d95`; supplement `28c1ae98a10e490d3a0e5e8bf22692559e64ba421d65d869424eeb653414d7c9`.

## Đánh giá tổng thể

Bản hiện tại có bằng chứng cho một nghiên cứu đánh giá các cấu hình: chia theo đối tượng, so sánh bắt cặp, công khai khác biệt training, kiểm tra lỗi theo lớp và phân biệt phân tích chính với mở rộng. Những kết quả này vẫn có giá trị khi E4 vượt E3 hoặc một can thiệp mới không thành công.

Điểm yếu chính là câu chuyện trong abstract và phần mở đầu Discussion tập trung vào E3, trong khi bằng chứng về lựa chọn preprocessing phức tạp hơn. Thêm vào đó, phạm vi hiện tại chỉ có một chiều chuyển cohort, hai seed cố định, một họ đối chứng CNN-BiLSTM và chưa kiểm tra biện pháp xử lý lỗi N3. Đây là hạn chế khoa học thực sự; thêm lời giải thích hoặc hình ảnh không thể thay thế bằng chứng còn thiếu.

Góp ý của thầy xác định đúng nhu cầu củng cố bằng chứng. Một số cách diễn đạt cần điều chỉnh: phép so sánh 0,5732 với 0,5680 trộn seed; cơ chế vật lý chưa được xác định; bài không kết luận N3 là không thể khắc phục; và hạng Q1/Q2 tự nó không xác lập một quy tắc rằng mọi nghiên cứu phải đề xuất biện pháp chữa lỗi. Tuy nhiên, kỳ vọng nghiên cứu của thầy là lý do rõ ràng để xây dựng thêm nhánh can thiệp và đối chứng ngoài.

## Các phát hiện cần ưu tiên

### R1 — E3/E4: bằng chứng hiện có mạnh hơn cách trình bày, nhưng chưa chứng minh cơ chế riêng của SHHS

| Bộ dữ liệu và đại lượng | Seed | E3 | E4 | Cách đọc |
|---|---:|---:|---:|---|
| Sleep-EDF, pooled macro-F1 | 42 | 0,7904 | 0,7891 | E3 nhỉnh hơn trong chiến dịch chính |
| Sleep-EDF, pooled macro-F1 | 123 | 0,7883 | 0,7887 | E4 đã nhỉnh hơn ngay ở miền nguồn |
| SHHS1, subject-mean macro-F1 | 42 | 0,5680 | Chưa thấy kết quả trong báo cáo hiện tại | Chưa đối chiếu đủ cặp |
| SHHS1, subject-mean macro-F1 | 123 | 0,5634 | 0,5732 | E4 cao hơn E3 cùng seed |

Nguồn: bảng seed trong supplement và báo cáo SHHS seed 123. Các ô được làm tròn theo báo cáo. Phân tích bắt cặp SHHS seed 123 đã lưu cho E3−E4 là −0,0098, CI [−0,0125; −0,0073], Holm p = 1,89×10⁻¹⁰. Không tính lại CI trong lượt này.

Hai hệ quả chưa được phương án trả lời trước nêu đủ rõ:

1. Khác seed không bác bỏ góp ý: cùng seed 123, E4 vẫn vượt E3.
2. Riêng E3/E4 chưa cho thấy đảo thứ hạng do chuyển miền ở seed 123, vì E4 đã dẫn trên EDF. Chênh lệch EDF đang là pooled, SHHS là subject-mean; không lấy hai chênh lệch này để lượng hóa tương tác với domain. Nếu muốn kiểm tra tương tác, cần cùng đại lượng ở hai cohort, và xử lý khác biệt OOF/ensemble.

Abstract hiện không nhắc E4, contribution nói E3 có kết quả tổng thể tốt hơn mà không nêu đối tượng so sánh ngay trong câu, còn Discussion dẫn bằng ưu điểm E3. Main và Conclusion đã thừa nhận E4 vượt E3; vì vậy vấn đề là mức độ nổi bật và phạm vi câu chữ, không phải thiếu hoàn toàn thông tin.

Đề nghị biên tập: gọi E3 là cấu hình được đánh giá chính theo giao thức lịch sử, ghi rõ lợi ích so với E0/E6; đưa kết quả E4 cùng seed vào phần kết quả chính và tóm tắt ngắn trong abstract. Việc khóa E3 giải thích lựa chọn thực nghiệm trước đây, không chứng minh E3 là lựa chọn tốt nhất cho ứng dụng. Nếu sau này chuyển khuyến nghị sang E4, phải công bố việc lựa chọn đó dựa trên dữ liệu nào.

### R2 — Không đủ cơ sở để gán tác hại của phép chia 100 cho hệ thống ghi SHHS

Code EDF và SHHS dùng chung `preprocess_signal_variant`; SHHS thêm resampling trước bước này. Khi clipping không kích hoạt, khác biệt tín hiệu E3/E4 là đổi thang. Phép nhân với hằng số dương bảo toàn hình dạng, thứ tự thời gian và tỷ lệ năng lượng giữa các dải trong số học lý tưởng; năng lượng tuyệt đối đổi theo bình phương hằng số. Vì vậy lời giải thích kiểu “chia 100 làm mất sóng chậm” chưa có căn cứ.

Mạng có BatchNorm và được huấn luyện riêng cho từng điều kiện. Với cùng activation z, BN trong train mode có dạng `(z−mean)/sqrt(var+epsilon)`. Đổi z thành c·z tương đương thay epsilon bằng epsilon/c² trong biểu thức đã khử c; với c=0,01, hệ số tương ứng là 10.000. Đây chỉ là lý do để kiểm tra mức độ ảnh hưởng của epsilon, không chứng minh ảnh hưởng đáng kể trong dữ liệu thực. Running statistics, trạng thái optimizer, lịch dừng và trọng số cuối cũng cần đối chiếu. Nghiên cứu về normalization và tối ưu hóa là cơ sở xem xét tương tác này, không phải bằng chứng cơ chế cho E3/E4: [van Laarhoven, 2017](https://arxiv.org/abs/1706.05350).

Thứ tự chẩn đoán nên là: đúng đơn vị/variant/checkpoint → kiểm tra quan hệ `100*x_E3 ≈ x_E4` với sai số float phù hợp và căn chỉnh epoch → khởi tạo/training → thống kê activation/BN trên nguồn và đích. So sánh PSD hoặc biên độ giữa hai cohort chỉ mô tả khác biệt; không tách được montage, tuổi, thiết bị và scoring.

Một kiểm tra số học hữu ích cho chiến dịch mới: với stem convolution không bias hiện tại, thay input x bằng x/100 và nhân trọng số convolution đầu tiên với 100 phải giữ đầu ra gần như cũ trong eval mode. Đây là kiểm tra tính nhất quán của pipeline bằng bản sao mô hình, không phải cách cải thiện mô hình hay một kết quả sinh lý. Không nên chỉ đưa input chưa scale vào checkpoint E3 rồi diễn giải suy giảm như chứng cứ chống fixed-scale.

### R3 — Có vấn đề thứ tự đặt seed trong mã hiện tại cần xử lý trước thực nghiệm mới

Trong [train_sequence_model](D:/SleepTCN/src/sleeptcn/experiment.py:511), `build_sequence_model` ở dòng 521 chạy trước `seed_everything(context.seed)` ở dòng 526. [Model factory](D:/SleepTCN/src/sleeptcn/workflows/model_factory.py:41) khởi tạo trực tiếp BiLSTM/TCN và không tự đặt seed. Do đó seed đặt trong hàm này chưa kiểm soát bước khởi tạo; khởi tạo có thể phụ thuộc trạng thái RNG để lại từ giai đoạn trước, kể cả khác biệt chạy mới/resume.

Đây là phát hiện qua đọc mã, chưa phải phép thử tái lập hoặc chứng minh nguyên nhân của E3/E4. Cần đối chiếu revision mã thực sự tạo checkpoint cũ trước khi suy về chiến dịch đã công bố. Các điểm số từ checkpoint đã lưu không tự trở thành sai vì phát hiện này.

Trước chiến dịch mới: đặt seed trước khởi tạo; kiểm tra state khởi tạo bằng nhau khi cùng seed dù RNG bên ngoài bị thay đổi; xác minh resume khôi phục checkpoint/RNG đúng. Nếu sửa implementation rồi huấn luyện lại, chạy các nhánh đối chiếu theo cùng phiên bản mới và lưu phiên bản thí nghiệm riêng. Không ghép một nhánh mới với nhánh cũ rồi gọi đó là ablation kiểm soát hoàn toàn.

### R4 — Can thiệp N3 cần một câu hỏi kiểm chứng cụ thể

Thầy đúng ở chỗ bài sẽ mạnh hơn nếu kiểm tra được ít nhất một giả thuyết có thể can thiệp. Tuy nhiên, hiện tại chưa chẩn đoán được nguyên nhân sinh lý: E0/E3 cùng lỗi chỉ cho thấy lỗi tái diễn giữa hai cấu hình; E6 thất bại không loại trừ mọi can thiệp về biên độ.

Một câu hỏi vừa sức là: “Một hiệu chỉnh đầu ra hoặc thay đổi loss được chọn trước có giảm N3→N2, đồng thời duy trì F1 N3 và hiệu năng tổng thể hay không?” Không cần cam kết trước rằng can thiệp phải thành công hoặc N3 là limitation không thể khắc phục.

ResNet đã có weighted cross-entropy `N/(5*n_c)`; sequence model dùng unweighted cross-entropy. Nếu thử cost-sensitive learning, điểm can thiệp rõ nhất là loss ở TCN, giữ encoder/cache cố định và có đối chứng TCN huấn luyện lại bằng loss cũ với cùng thiết lập. Đừng đổi đồng thời encoder, sampler, loss, learning rate và stopping rule rồi gán toàn bộ hiệu quả cho loss.

Báo cáo macro-F1 theo đối tượng, N3 F1/recall/precision, true-N2→predicted-N3, recall N2 và ma trận nhầm lẫn đầy đủ. Dùng CI bắt cặp theo đối tượng; các fold không phải mười cohort độc lập. Đối tượng không có N3 cần quy tắc xác định trước cho metric N3 theo người, cùng số đối tượng có N3 và kết quả pooled; không mặc định bỏ họ khỏi mọi phép đánh giá false positive.

### R5 — Phương án EM/calibration trước đây chưa đủ chi tiết để triển khai

Calibration xác suất và hiệu chỉnh prior giải quyết hai vấn đề khác nhau. EM prior adjustment dựa vào giả định label shift và chất lượng xác suất; lệch montage/scoring khiến giả định cần được xem xét, không thể suy từ việc tỷ lệ N3 tăng. [Alexandari và cộng sự, ICML 2020](https://proceedings.mlr.press/v119/alexandari20a.html).

Đề xuất thí điểm phải có ba nhánh: đầu ra gốc; calibration từ validation nguồn; calibration cộng prior adjustment từ dữ liệu đích không nhãn. Báo cáo cả bước calibration đơn lẻ để tránh gán mọi thay đổi cho EM. Chốt cách ước lượng prior, smoothing, stopping và cách xử lý xác suất gần 0 trước đánh giá.

Vì SHHS dùng ensemble mười fold, kế hoạch còn phải chốt thứ tự calibration/EM/averaging. Không được dựng một calibrator cho ensemble bằng dự đoán của cả mười mô hình trên 78 người EDF rồi coi đó là dữ liệu held-out: phần lớn mô hình thành viên đã huấn luyện trên mỗi người. Một lựa chọn cần đánh giá là calibration theo fold bằng validation nguồn tương ứng, sau đó áp dụng quy trình hiệu chỉnh đã định nghĩa và tổng hợp dự đoán. Việc calibration và early stopping cùng dùng validation nguồn phải được mô tả; EDF outer test vẫn dành cho đánh giá.

Softmax cuối đến từ TCN huấn luyện unweighted, nên không tự chia xác suất cuối cho class weights của encoder để “khôi phục prior”. NLL/Brier/ECE trên dữ liệu thích hợp giúp kiểm tra xác suất, nhưng không chứng minh giả định label shift hay bảo đảm N3 sẽ cải thiện.

### R6 — Baseline UDA cần đối chứng source-only của chính nó và kiểm tra việc dùng nhãn đích

E0 đã dựa trên một thiết kế công bố, nhưng hệ so sánh hiện tại còn hẹp. Một baseline ngoài có thể giúp định vị bài. Phương án trước đề xuất ADAST có lý ở mức ứng viên, chưa đủ để chốt là lựa chọn tốt nhất hay dễ tích hợp nhất.

Khi so một mô hình UDA với E3, khác biệt bao gồm backbone, cách huấn luyện và quyền tiếp cận dữ liệu đích. Cần chạy cả baseline source-only của mô hình ngoài và phiên bản có adaptation để thấy lợi ích của adaptation trong cùng mô hình. Kênh, subject splits, cửa sổ đánh giá, số mô hình ensemble và cách tính metric cần được đối chiếu; các thay đổi so với triển khai tác giả phải được ghi lại.

Kiểm tra sơ bộ [mã ADAST của tác giả](https://raw.githubusercontent.com/emadeldeen24/ADAST/main/trainer/ADAST.py) cho thấy có gọi đánh giá loss/accuracy trên `trg_valid_dl` sau mỗi epoch. Đoạn đọc được trả mô hình ở cuối quá trình; chưa thấy chọn checkpoint theo accuracy tốt nhất. Điều này không đủ để cáo buộc test leakage, nhưng cho thấy không thể mặc định toàn bộ quy trình model selection là label-free. Nếu dùng nhãn validation SHHS để chọn hyperparameter hoặc dừng, cần công bố ngân sách nhãn và cách chọn; nếu yêu cầu UDA không nhãn đích, phải thiết kế selection phù hợp và tránh dùng các log đó để điều chỉnh.

Trước khi tích hợp cần kiểm tra toàn bộ mã, license, dữ liệu đầu vào và phiên bản thư viện. Code công khai không tự động đồng nghĩa tái lập được. Chọn một baseline có thể kiểm chứng trước; chưa có cơ sở chạy nhiều phương pháp “tốt nhất” chỉ theo bảng điểm từ các paper khác giao thức.

### R7 — Dữ liệu đích và cửa sổ đánh giá là điểm phương án trước đã bỏ sót

SHHS preprocessing hiện cắt cửa sổ theo first/last true sleep và ±30 phút, trước khi lưu input cho inference. Manuscript đã công khai đây là benchmark window phụ thuộc nhãn. Nếu dùng chính các input đó để ước lượng prior/huấn luyện UDA, lựa chọn đoạn tín hiệu đã có thông tin từ nhãn đích, dù loss không nhận nhãn.

Giao thức mới nên xác định dữ liệu adaptation bằng quy tắc không cần nhãn; có thể đánh giá trong cửa sổ benchmark đã công bố nhưng phải tách rõ hai bước. Nếu chuyển sang inference toàn bản ghi, cần chạy lại baseline tương ứng vì ngữ cảnh chuỗi cũng thay đổi. Nếu giữ cửa sổ cũ, công khai điều kiện benchmark này, tránh mô tả toàn bộ quy trình từ raw recording là hoàn toàn label-free.

180 người test đã được xem kết quả. Phân tích mới trên đó vẫn có giá trị nhưng mang tính mở rộng sau quan sát. Một holdout mới từ SHHS giúp kiểm tra khả năng lặp lại trên người mới trong cùng dataset; một cohort khác trả lời câu hỏi rộng hơn. Không coi hai điều này tương đương, không coi thêm seed là thay thế holdout, và không tự coi 20 reserve là mẫu đủ mạnh.

### R8 — Kế hoạch và hồ sơ phụ trợ chưa đồng bộ với mục tiêu mới

Kế hoạch trước nộp cũ có các số trang, vị trí dòng và trạng thái kiểm tra lịch sử; nhánh T20-D vẫn coi adaptation là tùy chọn. Nhận định này hợp lý với scope cũ, nhưng chưa phản ánh kỳ vọng mới của thầy. Cần một bản bổ sung hiện hành, giữ giao thức gốc cho provenance và cập nhật riêng trạng thái việc đã xong.

Highlights hiện vẫn viết “higher parameter and memory costs”, trong khi narrative đã giảm nhấn mạnh chênh lệch bộ nhớ theo yêu cầu tác giả. Câu “Preprocessing gave the largest observed post-hoc transfer contrast” cũng cần giới hạn rõ là các phép so sánh cấu hình đã thực hiện. Đây là việc đồng bộ câu chữ, không phải bằng chứng khoa học mới.

Hai tên/email thầy cung cấp là Tri Nguyen Ho Duy (`trinhd@uit.edu.vn`) và Tri Nguyen (`tringuyen@uit.edu.vn`). Việc bổ sung cần đồng bộ EN, VI, supplement, PDF metadata, running header, CRediT và acknowledgements; người đang được cảm ơn vì hướng dẫn có thể cần chuyển cách ghi khi trở thành đồng tác giả. Thứ tự, affiliation, corresponding author và vai trò cụ thể cần thông tin xác nhận; không suy từ email.

## Phương án sửa sau review

Mục tiêu đề nghị: củng cố nghiên cứu đánh giá preprocessing và lỗi chuyển cohort, thêm một phép kiểm tra khắc phục và một hệ quy chiếu ngoài. Chưa cần chuyển bài thành nghiên cứu đề xuất kiến trúc mới hoặc phát triển thuật toán UDA mới.

| Thứ tự | Công việc | Đầu ra để quyết định bước sau | Phụ thuộc |
|---|---|---|---|
| 1 | Chỉnh câu chuyện E3/E4, phân biệt kết quả chính và mở rộng; đồng bộ abstract/contributions/highlights | Một cách trình bày nhất quán, có đối chiếu cùng seed | Làm được với nguồn hiện có |
| 2 | Kiểm kê checkpoint/probabilities/metadata; kiểm tra seed và quan hệ tín hiệu E3/E4 | Biên bản xác nhận đầu vào và code tạo checkpoint; chỉ rõ sai lệch nếu có | Checkpoint và artifact nguồn |
| 3 | Hoàn thiện E4 seed 42 trên SHHS từ checkpoint đã có, nếu khôi phục được; phân tích E3/E4 ở hai seed | Bảng bắt cặp cùng seed, lớp lỗi và CI phù hợp | Không mặc định phải train lại; thêm inference khi có đủ artifact |
| 4 | Thí điểm calibration + EM với đối chứng calibration-only; lựa chọn thêm loss TCN nếu phù hợp câu hỏi và nguồn lực | Kết quả N3 và tổng thể, cả cải thiện lẫn suy giảm; đầy đủ selection rule | Probabilities/validation nguồn; cache/checkpoint nếu train TCN |
| 5 | Một baseline ngoài gồm source-only và UDA, sau kiểm tra code/giao thức | So sánh công bằng và định lượng lợi ích adaptation | Dữ liệu nguồn/đích, GPU, implementation đã kiểm tra |
| 6 | Xác nhận phương án đã chốt trên holdout phù hợp, nếu mục tiêu là kết luận khả năng khắc phục có thể lặp lại | Bằng chứng độc lập với lựa chọn trên cohort đã mở | Thiết kế sampling và cỡ mẫu trước mở kết quả |

Bước 1 và lập giao thức cho các bước sau có thể làm song song với khôi phục artifact. Một can thiệp UDA cũng có thể trả lời câu hỏi về N3; không bắt buộc chạy đồng thời EM, weighted loss, nhiều UDA và một backbone mới. Ưu tiên một gói nhỏ nhưng đối chứng đầy đủ.

Quy tắc đánh giá cần chốt cho phần bổ sung: primary metric và primary contrast; cách lựa chọn hyperparameter; quyền dùng dữ liệu/nhãn đích; đơn vị lấy mẫu là đối tượng; xử lý hai đêm EDF cùng người; cách tổng hợp seed và fold; family Holm nếu kiểm định nhiều đối chiếu; báo cáo toàn bộ biến thể đã định trước. Nếu dùng ngưỡng “duy trì hiệu năng” hoặc non-inferiority, cần định nghĩa margin có cơ sở trước đánh giá, không đặt sau khi thấy số.

Nếu muốn so “tác động preprocessing ở nguồn và đích”, dùng cùng metric và ghi rõ khác biệt OOF/ensemble; với checkpoint cố định, CI theo subject không bao hàm mọi bất định do training. Không cộng hai seed thành 360 người độc lập, không cộng các epoch thành số quan sát độc lập để làm CI hẹp.

## Khả thi tại môi trường hiện tại

Kiểm tra read-only ngày 01-10-2026 thấy các biến thể EDF processed và 57 file NPZ trong cây feature cache; số lượng này chưa chứng minh cache đủ cho mọi fold/cấu hình. Không tìm thấy `.pt/.pth/.ckpt` trong `runs` và `data/cache` đã kiểm tra. Ổ E: và đường dẫn SHHS trong artifact index không hiện diện ở phiên hiện tại. Đây là trạng thái truy cập hiện tại, không có nghĩa dữ liệu/checkpoint gốc đã bị mất.

Vì vậy có thể review/sửa cách trình bày và chuẩn bị mã/giao thức, nhưng chưa thể hứa chạy ngay E4, EM, TCN hoặc UDA từ các aggregate JSON. Cần khôi phục đường dẫn artifact hoặc kết nối môi trường tính toán chứa dữ liệu. Lượt review này không ước lượng thời gian GPU hay chi phí khi chưa xác minh đầu vào và môi trường chạy.

## Điều chỉnh trực tiếp đối với phương án trả lời trước

- Giữ việc sửa cách định vị E3 và bổ sung đối chiếu cùng seed; thêm phát hiện E4 đã nhỉnh hơn E3 ở EDF seed 123.
- Đưa kiểm tra preprocessing/seed/provenance lên trước chẩn đoán cơ chế hoặc chạy thêm phương pháp.
- Giữ EM như một can thiệp có điều kiện; bổ sung calibration-only, giao thức ensemble và cửa sổ adaptation không dùng nhãn.
- Giữ weighted TCN là nhánh có thể kiểm tra, với đối chứng retrain tương ứng; chưa cần thay kiến trúc.
- Hạ ADAST từ lựa chọn ngầm định xuống ứng viên cần kiểm tra đầy đủ, kèm source-only counterpart và selection không nhìn test.
- Giữ nhận định cần holdout để xác nhận độc lập; không diễn giải rằng thiếu holdout làm mọi kết quả mở rộng vô giá trị.
- Thừa nhận nguồn lực hiện tại chưa đủ để thực thi phần thí nghiệm, thay vì đưa ra thứ tự chạy như thể checkpoint và SHHS đã sẵn sàng.

Bản review này hoàn thành việc đánh giá phương án; các hàng thực nghiệm trong bảng là công việc đề xuất, chưa được đánh dấu hoàn tất.
