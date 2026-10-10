# ADAST/source-only đủ mười fold: kết quả đã kiểm chứng — 04/10/2026

## Kết quả chính

ADAST cải thiện nhận diện N3 so với đối chứng source-only tương ứng, nhưng không
cải thiện kết quả tổng thể. Đây là hai tổ hợp mười mô hình mới, không trộn pilot
CPU một fold hoặc kết quả E3 lịch sử vào đối chứng.

| Chỉ số trên 180 người SHHS | Source-only | ADAST |
|---|---:|---:|
| Macro-F1 trung bình theo người | 0,4949 | 0,4497 |
| Macro-F1 gộp | 0,5539 | 0,5072 |
| Accuracy | 0,6977 | 0,6951 |
| N3 precision | 0,9266 | 0,9225 |
| N3 recall | 0,3195 | 0,4269 |
| N3 F1 | 0,4752 | 0,5837 |
| N1 recall | 0,1974 | 0,0000 |
| N2 recall | 0,8712 | 0,8655 |
| REM recall | 0,4343 | 0,2938 |
| Tỷ lệ N3 bị nhầm thành N2 | 0,6747 | 0,5588 |

Bootstrap ghép cặp theo người, 10.000 lần lấy mẫu, seed 2031:

- N3 recall: chênh +0,1074; CI 95% [0,0714; 0,1422].
- N3 F1: chênh +0,1085; CI [0,0721; 0,1442].
- Macro-F1 trung bình theo người: chênh −0,0452; CI [−0,0558; −0,0347].
- Macro-F1 gộp: chênh −0,0467; CI [−0,0585; −0,0347].

Nhánh ADAST không dự đoán N1 nào trên toàn bộ cửa sổ được chấm. Đây không chỉ
là một điểm recall thấp do làm tròn. Nhận diện REM cũng giảm; accuracy thay đổi
ít không thể thay thế macro-F1 và đánh giá từng lớp. Dù N3→N2 giảm, hơn một nửa
epoch N3 vẫn bị gán N2. Không dùng CI của N3 precision chứa không để tuyên bố
hai nhánh tương đương.

## Đánh giá ngoài fold trên nguồn

Mỗi người Sleep-EDF chỉ được chấm bởi mô hình của fold mà họ thuộc tập test;
source attention được dùng cho cả hai nhánh. Đây **không phải tổ hợp mười mô hình**.
Đủ 78 người, 195.469 epoch hợp lệ, không có đối tượng test trùng giữa các fold.

| Chỉ số nguồn | Source-only | ADAST |
|---|---:|---:|
| Macro-F1 trung bình theo người | 0,6042 | 0,4655 |
| Macro-F1 gộp | 0,6665 | 0,5276 |
| Accuracy | 0,7724 | 0,7116 |
| N3 recall | 0,7689 | 0,5933 |
| N1 recall | 0,1538 | 0,0039 |
| REM recall | 0,5895 | 0,2337 |

Chênh macro-F1 trung bình nguồn là −0,1388, CI [−0,1520; −0,1250]. Như vậy,
suy giảm không chỉ xuất hiện trên SHHS; không thể quy toàn bộ cho chuyển quần thể.

## Giao thức và thời gian

- Hai mươi mô hình mới trên CUDA/T4, seed 123, float32; ghép cặp cùng khởi tạo,
  thứ tự mẫu và 1.140 update/nhánh/fold. Upstream model/config/similarity penalty
  được ghim revision, không sửa trong lượt train.
- Hai round × 15 epoch × 38 batch, batch 128; mỗi epoch dùng 4.864 epoch nguồn,
  không phải một lượt qua toàn bộ nguồn. Năm người adaptation cung cấp 4.989
  epoch không nhãn, tách biệt 180 người test.
- Checkpoint cuối theo ngân sách cố định; không chọn bằng điểm test hoặc nhãn
  adaptation. Khóa đủ 20 checkpoint trước đánh giá cục bộ.
- SHHS suy luận toàn bản ghi, không dùng annotation để chọn đầu vào; sau đủ mọi
  dự đoán mới áp mặt nạ chấm lịch sử. Có 183.528 epoch suy luận, 169.012 epoch
  hợp lệ được chấm. Maximum dual-head logits → softmax từng fold → trung bình
  xác suất mười fold; source-only dùng source attention, ADAST dùng target attention.
- Runner huấn luyện: 509,59 giây, khoảng **8 phút 30 giây**, không tính upload.
- Suy luận nguồn/đích, chấm và kiểm chứng cục bộ: 1.240,58 giây, khoảng
  **20 phút 41 giây**. Không train thêm trên CPU, không truyền dữ liệu test lên cloud.
- Không dùng các thời gian CUDA/CPU mới này để thay benchmark Tesla V100 lịch sử
  trong paper, vì công việc và điều kiện đo khác nhau.

## Kiểm chứng và phạm vi diễn giải

Archive tải về khớp hash quan sát trên Colab. Đủ 20 checkpoint, khởi tạo, thứ tự
mẫu, lịch học và ngân sách đều đạt kiểm chứng độc lập. Sai khác khởi tạo native
Windows được xử lý bằng dựng lại phép làm tròn float32 của Colab để **khớp hash
tuyệt đối**, không dùng tolerance để bỏ qua sai khác. Chỉ số permutation được
kiểm tra theo byte int64 Linux; không thay mẫu huấn luyện.

Lượt chấm có cả verification nội bộ và độc lập: 180 tệp dự đoán, 3.600 mảng xác
suất fold/nhánh, mọi ma trận nhầm lẫn và bootstrap được tính lại. Tiến trình riêng
chạy lại 40 cặp mô hình–bản ghi đích và 20 cặp mô hình–batch nguồn, khớp chính xác.
Aggregate SHA-256:
`cf472a96dfeaf6cdbd0eb59d46e591c7c7863609a17eec35dad8f9b7b012950a`.

Kết quả áp dụng cho implementation theo giao thức chung, một seed và ngân sách
adaptation hạn chế đã nêu. SHHS đã được khảo sát trước khi thiết kế các can thiệp;
đây là phân tích bổ sung, không phải holdout mới. CI điều kiện trên mô hình đã
khớp, không bao gồm biến thiên huấn luyện. Không kết luận ADAST/UDA nói chung
không hiệu quả, không xếp hạng E3 bằng điểm của một phương pháp có ngân sách khác.

**Còn chưa thực hiện:** độ nhạy theo ngân sách, nhiều seed và các baseline UDA
khác. Các kết quả hiện tại đáp ứng phần mở rộng đối chứng mười fold đã yêu cầu,
nhưng chưa đưa ra giải pháp tăng N3 mà đồng thời giữ tốt các giai đoạn khác.
Không tự mở thêm chiến dịch train, không nộp bài hoặc gán vai trò tác giả.
