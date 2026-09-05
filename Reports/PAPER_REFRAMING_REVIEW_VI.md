# Đánh giá và định hướng lại bản thảo SleepTCN

## Kết luận biên tập

Project không nên được trình bày như một phương pháp ResNet-1D--TCN mới hoặc một mô hình vượt SOTA.
Hai hướng đó không được số liệu hiện tại hỗ trợ. Bản thảo có cơ sở công bố hơn khi trả lời một quyết
định thực tế trước khi triển khai mô hình sang cohort khác:

> Khi một pipeline sleep staging được phát triển trên Sleep-EDF nhưng phải dùng trên cohort có montage,
> thiết bị và phân bố tuổi khác, nên ưu tiên thay kiến trúc, thay preprocessing, hay dành dữ liệu có nhãn
> để thích nghi một số decision boundary cụ thể?

Đây là một vấn đề thực tế vì nguồn lực phát triển có hạn. Một nhóm không thể đồng thời thay encoder,
sequence model, preprocessing và thu thập nhiều nhãn target-domain mà vẫn biết thành phần nào tạo ra lợi
ích. Đánh giá chuyển miền không cập nhật trọng số cho biết pipeline hiện tại hỏng ở đâu trước khi chọn chiến lược thích nghi.

## Ba câu hỏi nghiên cứu cần giữ

| Câu hỏi | Quyết định thực tế mà câu hỏi hỗ trợ | Bằng chứng hiện có | Câu trả lời được phép kết luận | Giá trị |
|---|---|---|---|---|
| RQ1. TCN và ResNet-1D có tạo ra incremental benefit đủ ổn định để đáng thay kiến trúc không? | Có nên tiếp tục đầu tư vào architecture search hay không? | E1−E0 và E2−E1 trên cùng 10 fold, cùng subject, cùng protocol; kiểm định theo subject; hai seed | Không thiết lập được lợi thế dự đoán ổn định. Lợi ích chắc hơn là vận hành: pipeline nhanh hơn, nhưng lớn hơn về parameter và memory. | Ngăn việc diễn giải các chênh lệch benchmark nhỏ như bằng chứng kiến trúc vượt trội. |
| RQ2. Lựa chọn pipeline/preprocessing nào giữ được lợi ích khi chuyển sang SHHS1 mà không cập nhật trọng số? | Trước khi có target labels, nên giữ hoặc thay phần nào của pipeline? | Hai locked comparisons E3−E0 và E3−E6 trên 180 SHHS subject; E0/E3 inductive, E6 transductive ở cấp bản ghi; các component contrasts phụ; seed-123 sensitivity | E3 tốt hơn hai locked references; các preprocessing contrasts quan sát được lớn hơn architecture contrasts. Band-pass là ứng viên giải thích phần lớn khác biệt, nhưng E3−E2 và E4 extension là secondary/post-hoc nên chưa chứng minh nhân quả cho một operation. | Chuyển ưu tiên từ tiếp tục đổi backbone sang signal handling và target-domain evaluation. |
| RQ3. Lỗi nào phải xử lý trước khi triển khai và generic normalisation có đủ không? | Nếu chỉ có ít nhãn SHHS, nên dùng chúng ở đâu và đánh giá metric nào? | Class-wise F1, confusion channels, đối chiếu E0/E3, transition-region metrics và E6 sensitivity | N3→N2 là failure mode chung của E0/E3 và là ưu tiên thứ nhất; N2→REM trên E3 là ưu tiên thứ hai. Per-record z-score không cứu N3, do đó không phải stand-alone remedy. Thí nghiệm kế tiếp nên là class-specific calibration hoặc limited fine-tuning có nhãn. | Biến một aggregate domain gap thành mục tiêu thích nghi, đồng thời tránh quy sai lỗi N3 cho riêng kiến trúc E3. |

## Vai trò đúng của E6 và z-score

“E6 không cải thiện N3” không phải đóng góp độc lập. Nếu viết thành một contribution riêng, reviewer có
thể phản bác đúng rằng đây chỉ là một phép thử tuỳ ý không thành công.

E6 chỉ có lý do xuất hiện khi đặt trong RQ3:

1. Record-wise normalisation là một cách sửa rẻ và phổ biến khi nghi ngờ scale mismatch.
2. Nếu nó cứu N3, nhóm có thể thử một biện pháp không cần target labels.
3. Nó không cứu N3: recall 0.2005 so với 0.2582 của E3; gần transition là 0.0721 so với 0.0733.
4. Vì vậy, generic normalisation không đủ và nguồn lực nên chuyển sang labelled, class-specific
   adaptation.

Giá trị không nằm ở việc “z-score thất bại”, mà ở quyết định mà kết quả đó loại bỏ: không nên tiếp tục
coi một thay đổi scale toàn cục là giải pháp deployment cho lỗi N3.

## Những thông tin cần nằm trong main paper

1. Bối cảnh triển khai: Sleep-EDF và SHHS khác cohort, age distribution, acquisition và EEG derivation.
2. Subject-wise 10-fold protocol và cách test prediction được tạo out-of-fold.
3. Bốn pre-specified contrasts và nhãn rõ ràng cho secondary/post-hoc analyses.
4. Một bảng kết quả Sleep-EDF, một bảng paired effects và seed sensitivity.
5. Locked SHHS results của E0, E3, E6; component contrasts chỉ dùng để hỗ trợ diễn giải.
6. Đối chiếu E0/E3 cho N3→N2, phân tích N2→REM của E3 và transition-region N3 recall; ghi rõ phản thực 74,5% được tính trên E3.
7. E6 N3 metrics như sensitivity result, không gọi là contribution.
8. Operational trade-off: latency, parameter count và memory.
9. Limitations ảnh hưởng trực tiếp đến inference: hai seed, một external cohort, ground-truth-anchored
   evaluation window, chưa có calibration và không tách được montage khỏi age/population.

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

- Thay “the architecture did not improve” bằng “the architectural substitutions did not establish a
  stable predictive advantage under paired inference”.
- Thay “z-scoring failed” bằng “record-wise normalisation was not sufficient to resolve the dominant N3
  transfer error”.
- Thay “E3 is superior” bằng “E3 was the strongest evaluated complete procedure under the locked SHHS
  comparisons”.
- Thay “preprocessing causes better transfer” bằng “in the observed post-hoc comparison,
  preprocessing was the stronger development axis; no individual operation was identified as causal”.
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
adaptation algorithm; nó cung cấp bước chẩn đoán trước adaptation: lựa chọn pipeline nào giữ được lợi
ích và target labels nên được dùng cho boundary nào. Bản thảo đã bổ sung hai đối chiếu literature:

- He et al., *Cross-scenario automatic sleep stage classification using transfer learning and
  single-channel EEG*, BSPC 2023, DOI 10.1016/j.bspc.2022.104501.
- Van Der Donckt et al., *Do not sleep on traditional machine learning*, BSPC 2023, DOI
  10.1016/j.bspc.2022.104429.

## Việc còn chặn submission

1. **Provenance:** hash trong SHHS USB manifest khớp snapshot lịch sử `configs/shhs_zero_shot_v1.json`;
   `configs/shhs_v1_protocol.json` là hồ sơ mở rộng sau chạy. Đã ghi biên bản đối chiếu tại
   `Reports/SHHS_PROTOCOL_PROVENANCE.md`; không thay hash snapshot bằng hash của hồ sơ mở rộng.
   và metrics. Không nên submit khi đường provenance này còn mơ hồ.
2. **External baseline:** bản thảo đã nói rõ không claim SOTA, nhưng reviewer vẫn có thể yêu cầu một
   baseline hiện đại chạy cùng protocol. Nếu không chạy thêm, phải nhấn mạnh internal paired control và
   protocol non-comparability; đây vẫn là điểm yếu.
3. **Uncertainty:** thiếu absolute CI và full subject-level vectors trên Sleep-EDF. Nếu artifacts cho phép
   khôi phục với chi phí thấp, đây là bổ sung đáng làm hơn tuning.
4. **Target venue:** bài phù hợp hơn với biomedical signal processing/robustness hoặc applied sleep
   technology; không nên định vị như một paper phương pháp mới cho JBI.

## Phán quyết

Bản thảo sau khi định hướng lại có một câu hỏi có ích: nó giúp quyết định nên đầu tư vào đâu trước khi
đưa một sleep-staging pipeline sang dataset khác. Đóng góp nằm ở paired evidence và error prioritisation,
không nằm ở ResNet-TCN hay ở một negative z-score experiment. Đây là một paper mức vừa, có thể nộp sau
khi giải quyết provenance và làm rõ uncertainty; nó không phải SOTA paper và không nên được bán theo
hướng đó.
