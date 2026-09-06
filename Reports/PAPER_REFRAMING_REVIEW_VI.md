# Đánh giá và định hướng lại bản thảo SleepTCN

Ghi chú biên tập hiện hành, cập nhật 05-09-2026. Tài liệu này đã được sửa để không biến một bài đánh giá
pipeline thành yêu cầu phải đề xuất thuật toán adaptation mới. Kế hoạch chi tiết nằm tại
`BSPC_PRE_SUBMISSION_AUDIT_AND_EXECUTION_PLAN_VI.md`; trạng thái thực thi phải đọc ở revision log,
không suy từ một đề xuất trong ghi chú này rằng thí nghiệm đã được chạy.

## Kết luận biên tập

Project không nên được trình bày như một phương pháp ResNet-1D--TCN mới hoặc một mô hình vượt SOTA.
Hai hướng đó không được số liệu hiện tại hỗ trợ. Giá trị của bản thảo nằm ở một câu hỏi thực nghiệm
cụ thể về xử lý tín hiệu và độ bền của kết quả khi chuyển cohort:

> Khi một pipeline sleep staging được phát triển trên Sleep-EDF nhưng phải dùng trên cohort có montage,
> thiết bị và thành phần mẫu khác, các cấu hình đã thử đem lại đánh đổi dự đoán–vận hành nào,
> và lỗi theo lớp nào vẫn tồn tại dù score tổng thể được cải thiện?

Đánh giá dùng chung subject splits và phép đo bắt cặp giúp phân biệt lợi ích quan sát được của từng
cấu hình với một thứ hạng benchmark đơn thuần. Các đối chiếu không phải thiết kế factorial: E1 thay
cả mô hình chuỗi và recipe huấn luyện; E2 thay gói feature/context và recipe encoder. Nghiên cứu chưa
đo lợi tức đầu tư phát triển, hiệu quả thu nhãn hay hiệu quả của một phương pháp thích nghi.

## Ba câu hỏi nghiên cứu cần giữ

| Câu hỏi | Quyết định thực tế mà câu hỏi hỗ trợ | Bằng chứng hiện có | Câu trả lời được phép kết luận | Giá trị |
|---|---|---|---|---|
| RQ1. Các thay thế cấu hình mô hình chuỗi và feature/context đem lại lợi ích nào? | Cân nhắc đánh đổi dự đoán và vận hành trong các cấu hình đã thử | E1−E0 và E2−E1 trên cùng 10 fold và đối tượng; khác recipe được khai báo; hai seed | Chưa thiết lập lợi thế dự đoán ổn định qua các kiểm tra đã báo cáo. ResNet–TCN có forward pass nhanh hơn trong benchmark, nhưng nhiều tham số và tốn peak memory hơn. | Phân biệt lợi ích vận hành đo được với claim ưu thế kiến trúc. |
| RQ2. Lợi ích pipeline nào giữ được khi chuyển sang SHHS1 mà không cập nhật trọng số? | Đánh giá lại signal handling khi đổi cohort | Hai so sánh chính E3−E0/E3−E6 trên 180 người; E6 transductive; E1/E2 và E4 là các extension trên cùng cohort | E3 cao hơn hai đối chứng chính ở seed 42. E3−E2 hậu nghiệm lớn hơn các contrast E1−E0/E2−E1 đã thử; E4 seed 123 còn cao hơn E3. | Bằng chứng cụ thể rằng lựa chọn preprocessing đáng được kiểm tra cùng với pipeline, không phải quy luật preprocessing luôn quan trọng hơn kiến trúc. |
| RQ3. Score tổng thể che khuất lỗi theo lớp nào? | Xác định mục tiêu cho kiểm chứng tiếp theo | Confusion counts E0/E3/E6, oracle riêng E3 và chẩn đoán vùng nhãn | N3→N2 tái diễn ở E0/E3; N2→REM đứng thứ hai trong oracle E3. Pipeline z-score E6 đã thử không khắc phục N3. | Chỉ ra lỗi còn tồn tại qua pipeline và giới hạn của cải thiện aggregate; chưa khẳng định nguyên nhân, cách khắc phục hay cách phân bổ nhãn tối ưu. |

## Vai trò đúng của E6 và z-score

“E6 không cải thiện N3” không phải đóng góp độc lập. Nếu viết thành một contribution riêng, reviewer có
thể phản bác đúng rằng đây chỉ là một phép thử tuỳ ý không thành công.

E6 là một contrast preprocessing đã định trong chiến dịch và cung cấp thêm thông tin cho RQ3:

1. Record-wise normalisation là một cách sửa rẻ và phổ biến khi nghi ngờ scale mismatch.
2. Nếu nó cứu N3, nhóm có thể thử một biện pháp không cần target labels.
3. Nó không cứu N3: recall 0.2005 so với 0.2582 của E3; gần transition là 0.0721 so với 0.0733.
4. Vì vậy, pipeline z-score theo bản ghi đã thử không khắc phục lỗi N3. E6 được huấn luyện riêng;
   đây không phải phép can thiệp chỉ thay biên độ lên cùng model đóng băng.

Kết quả không loại trừ vai trò của biên độ, mọi cách chuẩn hóa không nhãn hoặc những chiến lược
thích nghi khác. Calibration/fine-tuning là hướng nghiên cứu tiếp theo có thể chọn, không phải bước
bắt buộc để bài đánh giá thực nghiệm này có đóng góp.

## Những thông tin cần nằm trong main paper

1. Bối cảnh triển khai: Sleep-EDF và SHHS khác cohort, age distribution, acquisition và EEG derivation.
2. Subject-wise 10-fold protocol và cách test prediction được tạo out-of-fold.
3. Bốn pre-specified contrasts và nhãn rõ ràng cho secondary/post-hoc analyses.
4. Một bảng kết quả Sleep-EDF, một bảng paired effects và seed sensitivity.
5. Locked SHHS results của E0, E3, E6; component contrasts chỉ dùng để hỗ trợ diễn giải.
6. Đối chiếu E0/E3 cho N3→N2, phân tích N2→REM của E3 và transition-region N3 recall; ghi rõ phản thực 74,5% được tính trên E3.
7. E6 N3 metrics như sensitivity result, không gọi là contribution.
8. Operational trade-off: latency, parameter count và memory.
9. Limitations ảnh hưởng trực tiếp đến inference: hai seed cố định, một hướng chuyển cohort,
   ground-truth-anchored evaluation window, EDF out-of-fold khác SHHS ensemble, chưa kiểm chứng một
   biện pháp calibration/adaptation và chưa tách được montage khỏi thành phần quần thể.

## Những thông tin không nên chiếm chỗ trong main paper

- Silhouette analysis.
- C/P/N context-group ablation và transition-pair breakdown.
- Group-interaction index.
- Mọi diễn giải dài nhằm biến null context result thành “mechanistic contribution”.
- Toàn bộ lịch sử tuning, candidate không hoàn tất hoặc fold đơn lẻ của `resnet_tuning_v3`.
- Các oracle corrections được diễn giải như performance đạt được hoặc bằng chứng causal.

Các nội dung này có thể giữ làm supplementary material hoặc internal record. Chúng không giúp trả lời
ba quyết định chính và làm người đọc mất dấu câu chuyện.

## Cách tăng sức nặng mà không bóp méo số liệu

Có thể chọn trọng tâm, giảm chi tiết không liên quan và dùng ngôn ngữ chính xác nhưng có lợi cho bài.
Không cần tự làm yếu bài bằng những câu dài xin lỗi cho từng limitation. Tuy nhiên, không được bỏ qua
kết quả làm thay đổi kết luận hoặc gọi post-hoc result là confirmatory.

Các cách diễn đạt nên dùng:

- Thay “the architecture did not improve” bằng “the evaluated sequence-model and feature/context
  replacements did not establish a stable predictive advantage across the reported checks”.
- Thay “z-scoring failed” bằng “record-wise normalisation was not sufficient to resolve the dominant N3
  transfer error”.
- Thay “E3 is superior” bằng “E3 was the strongest evaluated complete procedure under the locked SHHS
  comparisons”.
- Thay “preprocessing causes better transfer” bằng “the observed post-hoc preprocessing contrast was
  larger than the evaluated sequence-model and feature/context contrasts on this SHHS1 sample”.
- Thay “we identified an amplitude-threshold mechanism” bằng “the pattern is consistent with a
  conservative N3 boundary; montage and age are plausible but confounded contributors”.
- Thay “correcting two channels restores performance” bằng “the counterfactual analysis ranks these two
  channels by potential leverage; it is not an achievable-performance estimate”.

## Claims tuyệt đối không nên dùng

- “Novel ResNet-1D--TCN architecture.”
- “State-of-the-art performance.”
- “ResNet-1D is superior to the CNN encoder.”
- “Preprocessing causally dominates architecture.”
- “Prior shift is excluded as the cause.”
- “The N3 collapse is proven to be an amplitude-threshold/montage mechanism.”
- “The failure is correctable without retraining.”
- “The context ablation proves neighbouring epochs are unimportant.”
- “Two confusion channels explain essentially all cross-dataset generalisation.”

## Vị trí so với literature

Cross-scenario sleep staging đã có các phương pháp domain adaptation dùng Sleep-EDF và SHHS1. Vì vậy,
đánh giá chuyển miền không cập nhật trọng số tự nó không còn là novelty đủ lớn. Bài phải phân biệt rõ rằng nó không đề xuất một
adaptation algorithm; nó kết hợp đánh giá pipeline bắt cặp, đánh đổi vận hành và phân tích lỗi theo lớp
trên hai cohort. Những lỗi nổi bật là ứng viên cho kiểm chứng tiếp theo, chưa chứng minh target labels
nên được phân bổ thế nào. Các đối chiếu literature cần giữ gồm:

- He et al., *Cross-scenario automatic sleep stage classification using transfer learning and
  single-channel EEG*, BSPC 2023, DOI 10.1016/j.bspc.2022.104501.
- Van Der Donckt et al., *Do not sleep on traditional machine learning*, BSPC 2023, DOI
  10.1016/j.bspc.2022.104429.
- SleepInceptionNet của Haghayegh và cộng sự, JMIR 2023, đã khảo sát preprocessing/representation
  single-channel; không gọi ý tưởng đánh giá preprocessing là mới tự thân.
- ADAST là đối chiếu về thích nghi miền không nhãn có cập nhật mô hình, khác trọng số nguồn cố định
  và chuẩn hóa input ở E6. Related work không thay thế một baseline chạy cùng protocol.

## Việc còn chặn submission

1. **Provenance:** hash trong run manifest SHHS khớp snapshot lịch sử `configs/shhs_zero_shot_v1.json`;
   `configs/shhs_v1_protocol.json` là hồ sơ mở rộng sau chạy. Đã ghi biên bản đối chiếu tại
   `Reports/SHHS_PROTOCOL_PROVENANCE.md`; không thay hash snapshot bằng hash của hồ sơ mở rộng.
   Ngày 05-09-2026 đã kiểm tra trực tiếp 540 ensemble prediction hashes và confusion counts. Đây là
   kiểm tra toàn vẹn và tái tính từ prediction, không phải tái huấn luyện/tái sinh độc lập. Link archive
   và quyền chia sẻ vẫn cần tác giả xác nhận.
2. **External baseline:** bản thảo đã nói rõ không claim SOTA, nhưng reviewer vẫn có thể yêu cầu một
   baseline hiện đại chạy cùng protocol. Nếu không chạy thêm, phải nhấn mạnh internal paired control và
   protocol non-comparability; đây vẫn là điểm yếu.
3. **Uncertainty:** aggregate package chưa cung cấp đầy đủ absolute CI và phân bố theo đối tượng của
   mọi kết quả. Có prediction gốc thì có thể khôi phục; nếu cần chạy phân tích mới phải chốt phạm vi
   và báo tác giả trước. Không giả rằng thiếu trong một JSON đồng nghĩa không thể tính.
4. **Thông tin trước nộp:** tác giả đã xác nhận Quân đứng đầu/liên hệ, nhưng CRediT chưa được điền;
   ethics, quyền NSRR/DUA, funding, competing interests và archive chưa được tác giả xác nhận.
   Guide for Authors BSPC trả HTTP 403 khi audit, nên template/word limit/review mode vẫn cần kiểm tra.

## Phán quyết

Bài có đóng góp thực nghiệm có thể trình bày rõ: các lựa chọn pipeline được đánh giá bắt cặp, lợi ích
vận hành đi cùng chi phí cụ thể, và score chuyển cohort tốt hơn không đồng nghĩa lỗi N3 đã được xử lý.
Không cần tự hạ giá trị này chỉ vì backbone quen thuộc; đồng thời cần đối chiếu literature và giữ claim
trong phạm vi đã đo. Khả năng nộp phải dựa trên các gate khoa học, provenance, khai báo và package cuối,
không trên một nhãn “mức vừa”, xác suất chấp nhận suy đoán hoặc điều kiện adaptation mặc định.
