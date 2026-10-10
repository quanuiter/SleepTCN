# Khởi động chiến dịch ADAST full-source — 06-10-2026

Gói chạy đã chuẩn bị và kiểm tra xong tại máy. Đã mở được notebook hiện tại bằng tài khoản có quyền và thực thi ô kiểm tra môi trường: Tesla T4, 14,56 GiB VRAM, PyTorch 2.11.0+cu130. Thông tin này là kết quả kiểm tra của phiên ngày 06-10, không lấy từ đầu ra lưu của các lượt cũ.

## Phạm vi thực nghiệm

Huấn luyện 16 mô hình mới cho fold 2–9, gồm source-only và ADAST reference mỗi fold. Bốn mô hình full-source đã kiểm chứng của fold 0–1 được dùng lại tại máy sau khi nhận kết quả. Câu hỏi là tác động của ADAST thay đổi thế nào khi cả hai nhánh học hết train nguồn trong mỗi epoch, so với ngân sách nhỏ đã đo.

Mỗi mô hình có 30 epoch; hai nhánh cùng khởi tạo, thứ tự nguồn và ngân sách nguồn. Giữ lõi huấn luyện đã dùng cho fold 0–1. Ô mới chỉ gọi chương trình CUDA, không có cơ chế chuyển sang train CPU. Giới hạn chung là 18.000 giây; checkpoint được giữ sau từng epoch.

Đối chiếu chính dùng final epoch 30 ở cả hai ngân sách. Best theo validation nguồn dùng cho phân tích phụ. Kết quả SHHS sẽ được chấm ở máy sau khi đủ checkpoint và kiểm chứng.

## Dữ liệu và quyền tải

Gói gồm Sleep-EDF nguồn đã bỏ ID, các phân hoạch nguồn và EEG thích nghi không nhãn của đúng năm người đã được người dùng cho phép. Không có dữ liệu hoặc nhãn của 180 người SHHS kiểm tra.

Người dùng đã trực tiếp cho phép train/validation nguồn fold 0 và fold 1 lên notebook hiện tại trong hai lượt trước. Kiểm tra các chỉ mục cho thấy hợp của hai phần đã cho phép bao phủ đúng 195.469 epoch nguồn, bằng toàn bộ phần nguồn của gói mới; không có epoch nguồn mới ngoài hợp này. Audit trước đó đã đối chiếu từng phần tử và thứ tự của các mảng đã bỏ ID với nguồn gốc. Dữ liệu thích nghi và notebook đích vẫn như các lượt đã được cho phép; tiếp tục trong phạm vi này theo yêu cầu chạy GPU Colab của người dùng.

Gói dùng chung các bản ghi nguồn cho nhiều fold. Mỗi mô hình chỉ lấy phần train/validation của fold tương ứng; outer test của mô hình đó không tham gia huấn luyện hoặc chọn checkpoint.

## Hồ sơ gói và giao diện

- ZIP input: 2.243.778.226 byte, chia thành chín phần.
- SHA256 ZIP: 03182361c85375593ada052cab6f55380313e21337144e855ab886e25ebd58f9.
- SHA256 manifest: 3b4fe50bf8823df7e7915e531a86dbc1f25616a225524b71e24589ca07fd3ae8.
- Notebook: https://colab.research.google.com/drive/19wRkGwBOlqm4voA-z-6Al52gzIaRDW68.
- Ô mới: 16, execution 2, ID cell-l-Xf-x0_zTnN.
- Lần đọc khoảng 22:24: ô đã in GPU thực tế và đang chờ các phần upload. Chín phần đã gửi qua filechooser; chưa quan sát thông báo kiểm tra hash xong hoặc epoch train.

Khi các phần đủ kích thước và hash, ô tự ghép ZIP, kiểm tra manifest/toàn bộ payload rồi gọi runner CUDA. Giới hạn upload là 3.600 giây; giới hạn train năm giờ bắt đầu riêng khi runner hoạt động. Không chạy lại ô đang hoạt động hoặc gửi trùng phần đang tải.

## Chuẩn bị local đã đạt

- Kiểm tra tái sử dụng: bốn mô hình, tám tệp best/final, dữ liệu và phân hoạch khớp.
- 17 kiểm thử preflight đã đạt.
- 21 kiểm thử runner/launcher mới đã đạt; các kiểm thử mới chỉ kiểm tra dữ liệu, cấu hình và xử lý dừng, không huấn luyện mô hình CPU.
- ZIP kiểm tra đúng danh sách tệp và CRC; bản sao payload khớp hash dữ liệu/lõi huấn luyện lịch sử.
- Gói và chín phần lưu ở thư mục tạm ổ C; các con trỏ nhỏ lưu trong thư mục runs của dự án. Không xóa dữ liệu lịch sử hoặc lưu thêm gói lớn ở ổ D gần đầy.

Hồ sơ chi tiết: runs/adast_fullsource_completion_20261006/bundle_verification.json, upload_parts.json, colab_launch.py, colab_launch_record.json. Báo cáo này ghi trạng thái khởi động; không phải kết quả thực nghiệm mới.
