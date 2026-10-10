# Chuẩn bị và khởi chạy ADAST trên Colab — 04/10/2026

## Phạm vi đã được người dùng cho phép

Người dùng yêu cầu tải phần ADAST lên Colab để tận dụng thời gian trong khi đánh giá weighted/unweighted tại máy. Sau khi lượt tải đầu tiên bị kiểm tra an toàn chặn vì gói có thêm dữ liệu nguồn, người dùng đã xác nhận rõ: **đồng ý tải EEG nguồn Sleep-EDF đã bỏ ID lên Colab**, ngoài năm người adaptation SHHS không nhãn đã đồng ý trước đó. Lượt bị chặn không truyền thành công các phần gói; sau xác nhận mới tiếp tục tải qua Chrome.

Gói chỉ có tín hiệu nguồn đã tiền xử lý cùng nhãn nguồn, tín hiệu adaptation toàn bản ghi không nhãn, chỉ số phân hoạch nguồn không có ID và mã huấn luyện allowlist. Không có nhãn thật adaptation, dữ liệu/nhãn của 180 người SHHS đánh giá, tên bản ghi, ID người tham gia, raw EDF hoặc private manifest trong gói tải. Private mapping và kiểm tra adaptation/test không trùng đối tượng giữ tại máy.

## Giao thức

- Mười cặp ADAST/source-only mới trên CUDA, không tái sử dụng pilot CPU một fold.
- Upstream pin commit `e0fb503544ddd38f71027c09e3401b900f3dabc3`; giữ nguyên các file model, config, utils và license.
- Giữ ngân sách pilot đã kiểm tra: seed 123, batch 128, hai round × 15 epoch × 38 bước = **1.140 update/nhánh/fold**; cùng khởi tạo và thứ tự lấy mẫu nguồn trong từng cặp.
- Source-only giữ CNN, source attention, dual-head/similarity regularization và cùng ngân sách nguồn; không target forward, adversarial loss hoặc pseudo-label loss.
- ADAST dùng đúng 4.989 epoch không nhãn của năm người adaptation. Pseudo-label cố định đầu mỗi round bằng trung bình logits hai head; inference maximum logits như upstream.
- Chọn checkpoint cuối theo lịch cố định, không theo điểm nguồn/đích. Source loss 1 → 0,1; LR giảm sau epoch 10 của round 0 trên tất cả joint-optimizer groups, đúng hành vi thực tế của adapter CPU trước đây (không gọi nhầm là chỉ encoder).
- Đây là implementation theo giao thức chung, không phải tái lập nguyên dữ liệu của paper ADAST. Ngân sách 38 bước/epoch không tương đương một lượt qua toàn bộ dữ liệu nguồn. SHHS đã được xem trước đó, nên đây là phân tích bổ sung hậu nghiệm, không phải holdout chưa được quan sát.
- Kế hoạch tổ hợp: softmax maximum logits từng mô hình, rồi trung bình xác suất mười fold; chấm SHHS cục bộ sau khi khóa đủ 20 checkpoint mới.

## Kiểm chứng trước upload

195.469 epoch nguồn của 153 bản ghi; từng fold có train/validation/test tách đối tượng, class counts và valid-epoch support khớp phân hoạch lịch sử. Năm người adaptation không trùng 180 người test. Array adaptation float32 và hash khớp cache label-independent dùng ở pilot cũ. Không lượng tử hóa hoặc đổi dtype để giảm dung lượng.

Gói ZIP lossless: **2.243.804.267 byte**, hash `9d2bf6e938d54c6443c05427a88f4b8be4fb9da739302fa60004a73e861305e3`. Chia chín phần (tám phần 256 MiB và phần cuối 96.320.619 byte) để truyền qua trình duyệt. Gói kiểm tra adaptation-only khoảng 55,8 MB được chuẩn bị lúc chờ xác nhận nhưng **không tải riêng** vì người dùng đã chấp thuận gói đầy đủ.

**7 tests passed**: cập nhật optimizer/loss trên CPU cho kết quả và model state giống adapter trước đây; ghép cặp khởi tạo/thứ tự mẫu; source-only không target forward; nạp checkpoint đã hoàn thành; budget guard. Đây là unit/integration tests nhỏ, không phải thêm benchmark hoặc CPU campaign. Script kiểm chứng archive sau tải về đã được viết và compile, chưa có kết quả GPU để chạy nó.

## Cập nhật sau khi huấn luyện hoàn tất

Colab đã hoàn tất mười cặp CUDA, đủ 20 checkpoint cuối, 30 epoch và 1.140 update
mỗi nhánh/fold. Tổng thời gian lượt runner là 509,59 giây, khoảng 8 phút 30 giây;
không bao gồm upload và không dùng thay benchmark vận hành lịch sử. Gói kết quả
đã tải về trước khi phiên Colab ngắt kết nối, SHA-256
`1f9545cb8ad2790cf6e5f111e0124632c59582d8eaf50b9a72185fb0e6dda778`.

Kiểm chứng archive, payload, checkpoint, khởi tạo ghép cặp, thứ tự mẫu, lịch học,
ngân sách và 20 mảng pseudo-label đều đạt. Aggregate huấn luyện có hash
`821f985f6337b5acb8a72d0aae0dd7594750cbe409c4f34d798e107110fcf9a9`;
cả verification nội bộ và độc lập đều khớp.

Bộ kiểm chứng ban đầu giả định byte khởi tạo và dtype chỉ số giống nhau giữa
Windows và Linux. Chẩn đoán xác nhận hai khác biệt môi trường, không thay đổi
huấn luyện: (1) float32 uniform được làm tròn một lần khi dùng phép nhân-cộng gộp;
dựng lại cách này cho hash khởi tạo **khớp tuyệt đối** với Colab, sai khác lớn nhất
so với khởi tạo native Windows là 1,49×10^-8; (2) permutation cùng giá trị nhưng
NumPy Windows cũ trả int32, Colab trả int64. Kiểm chứng dùng byte int64 đúng môi
trường gốc, không bỏ hoặc nới lỏng yêu cầu khớp hash. Các module không cập nhật
của source-only vẫn khớp chính xác với khởi tạo dựng lại.

Ngày 04-10, đã khóa đủ 20 checkpoint và mở lượt suy luận/chấm cục bộ giới hạn
18.000 giây. Không train mới, không truyền tín hiệu/nhãn test lên cloud. Tại lúc
cập nhật khởi chạy, lượt này còn đang suy luận. **Cập nhật cuối:** đánh giá và
kiểm chứng độc lập đã hoàn tất trong 1.240,58 giây (khoảng 20 phút 41 giây).
Xem `COLAB_ADAST_RESULTS_20261004_VI.md`; N3 cải thiện nhưng macro-F1 tổng thể
giảm, không còn trạng thái chờ chạy/chờ kết quả.

## Trạng thái lịch sử tại thời điểm khởi chạy upload

Đã đưa ô mới (ô 9, execution `[11]`) vào notebook Colab hiện có, giữ nguyên kết quả TCN. Đã gửi cả chín phần cho upload; kích thước các tệp trên Colab đang tăng. Ô mới đang **chờ upload**, chưa thực hiện update huấn luyện. Nó sẽ kiểm tra size/hash từng phần, ZIP tổng và từng payload trước khi tự mở lượt CUDA. Chờ upload tối đa 3.600 giây; giới hạn lượt training riêng 18.000 giây, lưu checkpoint model/optimizer/CPU và CUDA RNG sau mỗi epoch. Không tự restart khi chạm giới hạn training.

Lượt SHHS weighted/unweighted ở máy vẫn độc lập, không thay code/config đóng băng. Chưa có kết quả ADAST mười fold hoặc kết luận về khả năng khắc phục N3.
