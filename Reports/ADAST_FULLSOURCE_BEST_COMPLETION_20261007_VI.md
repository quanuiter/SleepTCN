# Đánh giá phụ ADAST toàn nguồn đã hoàn tất — 07-10-2026

## Hoàn thành phần bị dừng

Lượt tiếp tục kết thúc lúc 20:59 ngày 07-10, trong khoảng sáu phút. Giữ nguyên
149 tệp dự đoán đã có và tạo thêm đúng 31 tệp còn thiếu; đã chấm đủ 180 người
SHHS và outer test Sleep-EDF gồm 78 người theo đúng fold OOF. Không huấn luyện
lại. Mã suy luận, mô hình và cách chọn checkpoint được giữ nguyên; ngân sách
còn lại được tính sau khi trừ thời gian của lượt bị dừng.

Windows từng từ chối thay tệp tiến độ. Bộ tiếp tục xử lý riêng việc ghi trạng
thái, dùng lại các dự đoán đã kiểm tra và không thay đổi phép tính suy luận.
Kiểm chứng độc lập đạt: 20 checkpoint, 180 tệp dự đoán đích, 10 fold nguồn;
tính lại confusion và bootstrap; suy luận lại 40 cặp mô hình–bản ghi đích và
20 cặp mô hình–batch nguồn. Hash của cả 149 tệp dùng lại không đổi.

## Kết quả SHHS: checkpoint chọn bằng validation nguồn

Source-only là mô hình chỉ học nguồn; ADAST thêm thích nghi với EEG không nhãn
của năm người SHHS. Checkpoint của mỗi mô hình được chọn theo macro-F1 trên
validation Sleep-EDF, không dùng điểm SHHS để chọn. Mỗi nhánh kết hợp xác suất
đủ mười fold. Các hàng dưới là kết quả tổng hợp, không chứa danh tính cá nhân.

| Chỉ số | Source-only | ADAST |
|---|---:|---:|
| Macro-F1 trung bình theo người | 0,5073 | 0,5396 |
| Macro-F1 gộp | 0,5552 | 0,5854 |
| Accuracy | 0,6818 | 0,6709 |
| Recall N3 | 0,2828 | 0,6095 |
| N3 bị nhầm thành N2 | 71,10% | 36,35% |
| N2 bị nhầm thành N3 | 0,50% | 2,15% |

Chênh macro-F1 trung bình theo người là +0,0322, CI 95% [0,0234; 0,0412];
macro-F1 gộp tăng +0,0302, CI [0,0212; 0,0395]. Bootstrap bắt cặp theo người
có 10.000 lượt, seed 2031. Khoảng này mô tả biến thiên giữa người với các mô
hình đang xét, không bao gồm biến thiên do huấn luyện lại qua seed.

| Giai đoạn | Precision nguồn / ADAST | Recall nguồn / ADAST | F1 nguồn / ADAST |
|---|---:|---:|---:|
| W | 0,8367 / 0,8256 | 0,8253 / 0,8374 | 0,8310 / 0,8314 |
| N1 | 0,1276 / 0,1271 | 0,1807 / 0,1231 | 0,1495 / 0,1251 |
| N2 | 0,7278 / 0,8112 | 0,7555 / 0,5992 | 0,7414 / 0,6893 |
| N3 | 0,9333 / 0,8891 | 0,2828 / 0,6095 | 0,4340 / 0,7232 |
| REM | 0,5302 / 0,4163 | 0,7466 / 0,8464 | 0,6201 / 0,5581 |

ADAST nhận ra nhiều N3 hơn và ít nhầm N3 thành N2 hơn. Chi phí đi kèm là F1
N1, N2 và REM giảm; REM recall tăng nhưng precision giảm. Accuracy giảm
1,09 điểm phần trăm. Đây là cải thiện cân bằng giữa lớp theo macro-F1, không
phải cải thiện mọi lớp hoặc tổng tỷ lệ dự đoán đúng.

## Kiểm tra trên outer test Sleep-EDF

Mỗi người được chấm bằng mô hình của đúng outer fold, không dùng ensemble
mười fold để chấm nguồn. Cả hai nhánh dùng source attention.

| Chỉ số | Source-only | ADAST |
|---|---:|---:|
| Macro-F1 trung bình theo người | 0,6681 | 0,6368 |
| Macro-F1 gộp | 0,7237 | 0,6921 |
| Accuracy | 0,7947 | 0,7730 |
| F1 W | 0,9154 | 0,9126 |
| F1 N1 | 0,3844 | 0,3224 |
| F1 N2 | 0,8328 | 0,8026 |
| F1 N3 | 0,7861 | 0,7865 |
| F1 REM | 0,6997 | 0,6365 |

Macro-F1 trung bình theo người giảm −0,0313, CI [−0,0416; −0,0205];
macro-F1 gộp giảm −0,0316, CI [−0,0416; −0,0221]. Lợi ích chuyển sang SHHS
đi kèm suy giảm hiệu suất nguồn, không chỉ là đánh đổi giữa các lớp trên đích.

## Đối chiếu với kết quả chính và nhận xét

| Cách lấy checkpoint | Macro-F1 theo người nguồn / ADAST trên SHHS | Chênh ADAST − nguồn |
|---|---:|---:|
| Epoch 30 — đối chiếu chính | 0,5010 / 0,5455 | +0,0445 |
| Tốt nhất theo validation nguồn — đối chiếu phụ | 0,5073 / 0,5396 | +0,0322 |

Cả hai cách đều cho cùng hướng kết quả: ADAST toàn nguồn tăng macro-F1 trên
SHHS, cải thiện N3 và làm giảm F1 N1/N2/REM cùng hiệu suất nguồn. Vì vậy,
nhận xét này không chỉ xuất hiện ở checkpoint epoch 30. Không dùng bảng này
để đổi lựa chọn chính theo SHHS; epoch 30 vẫn là đối chiếu chính của phân tích
hai ngân sách. Hai hàng dùng cùng người và mô hình liên quan, không phải hai
lần xác nhận trên quần thể độc lập. Không suy ra cơ chế sinh lý hay hiệu quả
trên quần thể khác từ riêng các số liệu này.

Kết quả epoch 30 và phân tích hai ngân sách vẫn là đối chiếu chính trong PDF.
Theo yêu cầu sửa paper tiếp theo, bản Anh/Việt nay có phương pháp và kết quả
đánh giá phụ; phụ lục bổ sung Bảng S32-S33. PDF/source ZIP/manifests được dựng
lại và rà bố cục, không thay số liệu chính. Thông tin “dừng ở 149/180” thuộc
trạng thái giao trước đó; bản hiện tại ghi rõ phần này đã hoàn tất. Hồ sơ bàn
giao mới ở `../tmp/adast_best_paper_revision_20261007/delivery_verification.json`.

## Dấu kiểm chứng

SHA-256 aggregate của đánh giá phụ:
`930edf8bdfbdacf4a76a1cb679da03ae3427edfee75e360c253678df006ce6a6`.
Hash thực tế khớp cả verification trong lượt, independent verification và
resumption verification. Kiểm chứng tiếp tục xác nhận mã/specification cũ
được giữ nguyên, 149 hash dùng lại không đổi, đúng 31 dự đoán mới, không
training và không tăng ngân sách tài nguyên.
