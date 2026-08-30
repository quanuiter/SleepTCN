# Giao thức thí nghiệm v2 — thiết kế công bằng và loại bỏ thành phần

> Tài liệu này là giao thức gốc của chiến dịch chính seed 42 và được giữ để truy nguyên quyết định trước
> khi mở test. Seed 123 được chạy sau giao thức như phân tích độ nhạy, không được hồi tố thành một phần
> của tập seed xác nhận định trước; xem `MULTISEED_SENSITIVITY_RESULTS.md`.

## Câu hỏi nghiên cứu

1. TCN thay BiLSTM có cải thiện khi giữ nguyên 15CNN và dữ liệu hay không?
2. ResNet-1D thay 15CNN có cải thiện khi giữ nguyên TCN và dữ liệu hay không?
3. Lọc dải, cắt ngoại lai và cách đổi thang biên độ đóng góp bao nhiêu?
4. Sự khác biệt có ổn định theo đối tượng và có ý nghĩa thống kê hay không?
5. Việc giảm số mô hình thành phần đánh đổi thế nào với số tham số, thời gian và bộ nhớ?

## Các thí nghiệm

| Mã | Dữ liệu | Bộ trích xuất | Mô hình chuỗi | Phép so sánh |
|---|---|---|---|---|
| E0 | `paper_raw_v1` | 15CNN | BiLSTM | Mốc tham chiếu đã sửa rò rỉ |
| E1 | `paper_raw_v1` | 15CNN của E0 | TCN chung | E1−E0: mô hình chuỗi |
| E2 | `paper_raw_v1` | ResNet-1D | TCN chung | E2−E1: bộ trích xuất/biểu diễn |
| E3 | `filtered_v2` | ResNet-1D | TCN chung | E3−E2: toàn bộ gói tiền xử lý |
| E4 | `bandpass_v2` | ResNet-1D | TCN chung | E4−E2: riêng lọc dải |
| E5 | `bandpass_clip_v2` | ResNet-1D | TCN chung | E5−E4: riêng cắt ±800 µV |
| E6 | `filtered_zscore_v2` | ResNet-1D | TCN chung | E3−E6: chia 100 so với z-score |

E3−E5 là tác động của chia hằng số 100 sau khi đã giữ cố định lọc và cắt.
E6−E5 là tác động của z-score theo bản ghi sau khi đã giữ cố định lọc và cắt.
Không diễn giải E3−E2 là tác động riêng của chuẩn hóa vì phép so sánh đó thay ba thao tác.

## Quyết định thực thi sau kiểm tra E4/E5 (2026-08-11)

Không chạy E5 ở các outer fold 01--09. E5 fold 00 đã hoàn tất được giữ lại làm bằng chứng
kiểm toán, nhưng E5 không được đưa vào phân tích hiệu năng hay kiểm định thống kê cuối cùng.

Lý do là kiểm tra độc lập toàn bộ 153 cặp NPZ xác nhận các trường khoa học `x`, `y`,
`valid_mask` và `original_epoch_index` của `bandpass_v2` (E4) và `bandpass_clip_v2` (E5)
trùng **bitwise**. Hai trường khác nhau duy nhất là metadata nhận diện biến thể
`normalization` và `preprocess_version`. Manifest tiền xử lý đồng thời ghi tổng và giá trị lớn
nhất của `clip_fraction` đều bằng 0. Vì vậy phép cắt ±800 µV là phép đồng nhất trên dataset này;
E5 không tạo ra một điều kiện dữ liệu khác E4 và không thể trả lời thêm câu hỏi khoa học.

Bằng chứng tái lập: `data/manifests/bandpass_clip_identity_v2.json`, được tạo bởi
`scripts/verify_variant_identity.py`. Nếu tiền xử lý, ngưỡng clip hoặc dữ liệu nguồn thay đổi,
phải chạy lại kiểm tra này trước khi áp dụng quyết định loại E5.

Đợt chạy giới hạn ngân sách dùng đúng một training seed là **42** cho E0, E1, E2, E3, E4 và E6
trên các fold 00--09. Các seed 123 và 2025 là mở rộng xác nhận trong tương lai, không thuộc đợt
hiện tại. Báo cáo cuối cùng phải nêu rõ giới hạn một seed; không diễn giải nó như đánh giá độ ổn
định theo khởi tạo ngẫu nhiên.

## Bất biến thiết kế

- Tất cả điều kiện được chạy dùng đúng manifest 10-fold v2, có cùng membership seed 42.
- Seed chia dữ liệu luôn là 42. Seed huấn luyện không được thay đổi split.
- Trong một đợt so sánh, mọi E dùng cùng training seed.
- Hai đêm của cùng đối tượng luôn cùng vai trò.
- Checkpoint chỉ được chọn bằng validation Macro-F1 (15CNN dùng validation weighted loss như baseline).
- Tập test bị khóa trong toàn bộ giai đoạn phát triển và chọn cấu hình.
- Sau khi mở test, không thay siêu tham số, biến thể tiền xử lý hoặc tiêu chí chọn checkpoint.
- Mỗi lỗi khoa học làm thay đổi kết quả phải tăng phiên bản giao thức và chạy lại mọi E bị ảnh hưởng.

`SleepTCN` là TCN 1D không nhân quả: convolution theo chuỗi epoch dùng padding đối xứng, nên mỗi
dự đoán có ngữ cảnh quá khứ và tương lai trong receptive field. Đây là đối chiếu chức năng phù hợp
với BiLSTM hai chiều cho bài toán chấm điểm toàn đêm offline; không được mô tả là TCN 2D hoặc mô
hình causal/thời gian thực. ResNet-1D xử lý waveform 3.000 mẫu bên trong từng epoch.

## Seed và mức công bố

- Đợt hiện tại: seed huấn luyện 42 cho E0, E1, E2, E3, E4 và E6; E5 bị loại theo quyết định
  bitwise ở trên.
- Bản xác nhận mạnh hơn: seed 42, 123 và 2025 cho tất cả thí nghiệm được đưa vào kết luận.
- Không được chạy một seed cho baseline và seed khác cho mô hình đề xuất rồi ghép cặp.

Phải quyết định tập seed trước khi mở test. Nếu ngân sách chỉ đủ seed 42, báo cáo minh bạch giới hạn này.

## Chỉ số và thống kê

- Chỉ số chính: Macro-F1 năm lớp.
- Chỉ số phụ: Accuracy, Cohen's kappa, precision/recall/F1 từng lớp.
- Khoảng tin cậy chính: bootstrap ghép cặp theo cụm đối tượng.
- Kiểm định hỗ trợ: Wilcoxon signed-rank hai phía trên điểm theo đối tượng.
- Hiệu chỉnh Holm cho nhóm so sánh chính.
- Báo cáo chênh lệch tuyệt đối, CI 95%, số đối tượng thắng/hòa/thua và p đã hiệu chỉnh.

## Phạm vi kết luận

E0–E6 trên Sleep-EDF chỉ hỗ trợ kết luận in-domain. Không được tuyên bố giải quyết domain shift,
zero-shot SHHS hoặc thực tế lâm sàng cho tới khi có giao thức SHHS riêng, không tinh chỉnh trên test SHHS.

“Đơn giản hóa” chỉ được dùng theo nghĩa giảm số mô hình/giai đoạn vận hành. E2–E6 có nhiều tham số
hơn E0 nên không được gọi là parameter-efficient hay mô hình nhẹ nếu chưa có bằng chứng khác.

## Quan hệ với paper gốc

E0 là tái triển khai trung thành về kiến trúc 15CNN tách rời + BiLSTM, kênh Fpz-Cz, epoch 30 giây
và các hyperparameter chính. Đây không phải tái tạo bitwise/nguyên xi: paper dùng MATLAB, không
công bố code/seed, validation cadence khác và kết quả headline 87,02%/82,09% được báo cáo trên
Sleep-EDF-13, trong khi chiến dịch này dùng Sleep-EDF-18 SC 153 bản ghi với validation theo đối
tượng chặt hơn. Vì vậy không dùng con số headline của paper làm ngưỡng bắt buộc cho E0. E1--E6
là ablation mới của dự án, không phải experiment được paper công bố.
