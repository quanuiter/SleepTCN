# Kiểm tra ngắn SleepTCN trên Colab Tesla T4 — 04/10/2026

## Kết luận và trạng thái xác minh

Phép thử ngắn đã hoàn tất trên Colab và đầu ra notebook báo `passed` cho CPU và CUDA. T4 có lợi ích tốc độ rõ ràng trên batch đặc trưng nguồn được thử. Đây không phải một fold huấn luyện hoàn chỉnh, không phải kết quả phân loại SHHS và chưa phải bằng chứng rằng toàn bộ quy trình nhanh hơn với cùng tỷ lệ.

**Đã nhận ZIP người dùng cung cấp và hoàn tất kiểm chứng cục bộ.** ZIP có đúng bảy tệp được phép; CRC và hash SHA-256 của sáu tệp payload đều khớp manifest. Hash mô hình, training, benchmark và dữ liệu mẫu trong CPU/CUDA JSON khớp gói nguồn cùng các tệp hiện có. Hai log kết thúc với `FINAL_STATUS passed`; progress báo `completed`. Trung vị từ 15 lần đo, tỷ lệ tốc độ, sai lệch gradient tổng hợp và bộ nhớ đã được tính lại, khớp summary.

Đây là kiểm chứng tính toàn vẹn tệp và tính nhất quán số liệu, không phải thực hiện lại phép so sánh tensor GPU trên máy cục bộ. Hash toàn ZIP được ghi nhận khi nhận tệp; chưa đối chiếu được với dòng hash toàn ZIP trên Colab trước khi mất kết nối. Manifest nội bộ không phải chữ ký số xác thực nguồn.

## Phạm vi đã thực hiện

- Tạo gói có danh sách tệp cho phép: mã mô hình, hàm huấn luyện/loss, script benchmark và tám bản ghi đặc trưng Sleep-EDF cùng nhãn nguồn. Không có ID người tham gia, EEG thô, SHHS, checkpoint chiến dịch, thông tin đăng nhập hoặc lịch sử Git.
- Tải gói vào bộ nhớ phiên Colab; kiểm tra SHA-256 toàn gói và từng tệp trước khi giải nén.
- Chạy cùng bài kiểm tra CPU và CUDA trong môi trường Colab: Python 3.13.15, PyTorch 2.11.0+cu130, Tesla T4, bộ nhớ thiết bị khoảng 14,56 GiB.
- Kiểm tra đầu ra, loss, gradient và bước Adam đầu tiên giữa tham chiếu CPU và thiết bị ứng viên. So sánh số học tắt dropout; phép đo tốc độ bật chế độ train/dropout thông thường.
- Mỗi cấu hình đo có 3 bước làm nóng và 15 lần lặp. Thời gian gồm chuyển batch lên thiết bị, forward, loss, backward, gradient clipping, Adam và đồng bộ CUDA; không gồm đọc/giải nén dữ liệu, trích xuất đặc trưng, validation cả fold hay ghi checkpoint.
- Chạy riêng bài tương ứng trên máy người dùng với Python 3.11.9, PyTorch 2.5.1+cpu, 4 luồng.
- Kiểm tra cú pháp và 9 kiểm thử mô hình/hàm huấn luyện đều đạt.

## Số liệu Colab đã đối chiếu với log tải về

| Batch đặc trưng | CPU Colab (giây/bước) | T4 (giây/bước) | CPU Colab/T4 | Gradient: sai lệch tuyệt đối cực đại | CUDA peak allocated (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 × 256 × 128 | 0,142939 | 0,012940 | 11,05× | 9,54 × 10⁻⁷ | 88,46 |
| 8 × 1236 × 128 | 1,579441 | 0,081538 | 19,37× | 7,75 × 10⁻⁷ | 310,69 |

Controller báo tổng thời gian khoảng 95,72 giây và `no_campaign_started: true`. Bộ nhớ bảng này là lượng tensor được PyTorch cấp phát cực đại trong phép thử, không phải tổng VRAM sử dụng của phiên hoặc dự báo bộ nhớ huấn luyện đầy đủ.

CPU máy người dùng đo khoảng 0,301 giây/bước ở batch tám bản ghi: T4 nhanh hơn khoảng 3,69 lần trong phép thử tương ứng. **19,37 lần là so với CPU Colab, không phải CPU máy người dùng.** So sánh giữa hai máy còn khác phiên bản PyTorch, hệ điều hành và phần cứng nên không diễn giải như phép cô lập tác động của GPU.

Các kiểm tra số học có dung sai float32, không đòi hỏi CPU/CUDA đồng nhất từng bit. Ở batch tám bản ghi, sai lệch đầu ra cực đại là 3,81 × 10⁻⁶, sai lệch gradient cực đại là 7,75 × 10⁻⁷ và sai lệch tham số sau bước Adam đầu tiên là 5,43 × 10⁻⁶. Cả hai batch đều đạt kiểm tra theo dung sai đã định nghĩa. Trung vị của GPU đã được tự tính lại từ đủ 15 lần đo cho cả phép đo resident và with-transfer.

ZIP kết quả nhận được có SHA-256:

`5f293aef8eddefc49c7460a57ff7bf5ca1c965a58a4b086eabee6651e3c2df41`

Bản sao ZIP và bảy tệp gốc được bảo toàn trong `runs/colab_t4_smoke_20261004`; báo cáo kiểm chứng nằm tại `verified_results/independent_verification.json`. Script tái kiểm chứng: `scripts/verify_colab_smoke_results.py`. Script không thực thi nội dung ZIP, không ghi đè tệp khác nội dung và không chạy huấn luyện.

## Bảo toàn chiến dịch hiện có

Không khởi chạy huấn luyện dài, không sửa mã mô hình/hàm huấn luyện đã đóng băng, không tiếp tục chiến dịch CPU đã chạm giới hạn và không chạy ADAST hoặc đánh giá SHHS.

Checkpoint dở dang fold 6/unweighted vẫn giữ nguyên SHA-256:

`5fa37275ed3dbc0050951a2ed125ef721410adaeec79960dc31457938ee0e17c`

Gói tải lên có SHA-256:

`63a2dc05d1888c412d99da39b24229a8fc7eae19afc63b16c4b838c5ef137f44`

Mã mô hình và training hiện tại khớp hash trong manifest gói. Không cập nhật số liệu bài báo từ phép kiểm tra thiết bị này.

## Phần cần hoàn tất

1. Trước chiến dịch GPU dài, chuẩn bị gói dữ liệu nguồn đầy đủ cho các fold còn thiếu, runner riêng có lưu checkpoint bền vững và phương án chuyển CPU → CUDA được kiểm chứng. Không coi việc chuyển backend là tiếp tục đồng nhất RNG; không sửa runner CPU đã đóng băng để che khác biệt môi trường.
2. Đo thời gian ít nhất một epoch thật gồm train và validation để ước tính ngân sách. Phép thử tám bản ghi hiện tại chưa đủ để dự báo thời gian một fold. Lượt kiểm chứng ZIP này không khởi chạy epoch hoặc chiến dịch mới.

Notebook đã chỉnh sửa nằm ở đường dẫn Colab người dùng cung cấp; có bản nguồn notebook tái chạy trong `runs/colab_t4_smoke_20261004`. Bản nguồn cục bộ không chứa đầu ra thực thi Colab.
