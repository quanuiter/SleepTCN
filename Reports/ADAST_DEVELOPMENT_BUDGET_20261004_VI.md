# Chẩn đoán ADAST theo ngân sách học — 04/10/2026

## Mục đích và trạng thái

Đã chuẩn bị, kiểm thử và khóa giao thức cho bốn mô hình mới trên fold 0, seed 123. Ô 13 của notebook Colab đã chạy một lần (execution 5), xác nhận Tesla T4 và PyTorch 2.11.0+cu130. Upload đã khớp hash và **cả bốn nhánh đã train đủ 30 epoch; kiểm chứng độc lập đã đạt**. Lượt train/validation mất 1.661,83 giây (27 phút 41,83 giây), không gồm upload. Kiểm chứng độc lập trên CPU mất 628,59 giây (10 phút 28,59 giây); đây là suy luận và tính lại chỉ số, không phải train lại. Không dùng kết quả cũ làm kết quả của lượt mới.

Mục đích: kiểm tra liệu ngân sách học nguồn quá nhỏ có góp phần làm ADAST mất khả năng nhận diện N1; ghi lại diễn biến qua từng epoch, thay vì chỉ xem mô hình cuối cùng.

## Bốn nhánh đã khóa

| Nhánh | Cách học | Cập nhật mỗi epoch | Epoch | Tổng cập nhật |
|---|---|---:|---:|---:|
| source_only_limited | Chỉ dữ liệu nguồn, ngân sách cũ | 38 | 30 | 1.140 |
| adast_limited | ADAST, ngân sách cũ | 38 | 30 | 1.140 |
| source_only_full_source | Chỉ dữ liệu nguồn, toàn bộ train mỗi epoch | 1.229 | 30 | 36.870 |
| adast_full_source | ADAST, toàn bộ train mỗi epoch | 1.229 | 30 | 36.870 |

Mỗi cặp dùng cùng khởi tạo, thứ tự dữ liệu nguồn, kiến trúc, tiền xử lý và quy tắc tối ưu. Train nguồn có 157.200 epoch tín hiệu; validation nguồn có 18.763 epoch. Chỉ ADAST dùng tín hiệu adaptation không nhãn của năm người SHHS đã được cho phép. Không tải dữ liệu hoặc nhãn của 180 người SHHS đánh giá, không đưa tín hiệu outer test nguồn vào gói.

Mức lớn hơn thay đổi cả số cập nhật, số lần xem tín hiệu adaptation và lịch học xét theo đơn vị cập nhật; batch cuối gồm 16 mẫu. Vì vậy, đây là so sánh **hai ngân sách huấn luyện**, không phải phép thử chỉ thay đổi duy nhất độ phủ dữ liệu nguồn.

## Theo dõi và chọn mô hình

- Ghi macro-F1, precision, recall, F1 và ma trận nhầm lẫn của cả năm lớp trên validation nguồn mỗi epoch; lưu logits để tính lại độc lập.
- Ghi riêng loss học nguồn, đối kháng, similarity và nhãn giả; ghi số mẫu/lớp đã học và độ phủ dữ liệu.
- Chọn checkpoint tốt nhất bằng macro-F1 validation nguồn theo đường source attention; nếu bằng nhau giữ epoch đầu tiên. Đường target attention trên validation nguồn chỉ phục vụ chẩn đoán.
- Giữ riêng checkpoint cuối epoch 30, không trộn với checkpoint tốt nhất. Không early stopping.
- Lưu model, optimizer, RNG CPU/CUDA và checkpoint mới nhất sau mỗi epoch.
- Toàn lượt có giới hạn 18.000 giây huấn luyện; không train CPU dự phòng, không tự gia hạn hoặc khởi chạy lại.

## Kiểm chứng trước và sau khi chạy

Đã đạt 14 test: so khớp phép cập nhật với adapter lịch sử, kiểm tra sampler/độ phủ, bảo đảm source-only không forward target khi train, kiểm tra checkpoint, tái dựng tổng loss từ các thành phần và chặn fallback CPU. Các test dùng dữ liệu giả nhỏ; không phải kết quả thực nghiệm. Sau khi khóa gói chỉ bổ sung kiểm tra phía máy local; không sửa runner/giao thức đã tải lên.

Gói đầu vào lossless gồm tám phần, tổng 2.025.338.867 byte. SHA-256 ZIP: `5a1368944a2f86b8c1900bc4cf8141fd7cbf2458e3e2becf7a65595697095b43`. Manifest SHA-256: `cbdf40d5800f238bf0ea5fbb2ac87bd11de7bfcc7eaab05c9e4c3cbd9ffc6b06`. Chỉ train sau khi toàn bộ phần tải lên và từng tệp trong gói khớp hash.

ZIP kết quả có SHA-256 quan sát trên Colab và khớp bản tải về: `312100c753f6db2d30ecc4808113c6f33cc7c8934f95841674354ffd2c2ba0bc`. Bộ kiểm chứng độc lập đã tính lại đủ 124 bộ chẩn đoán validation (khởi tạo + 30 epoch × bốn nhánh), tái tạo sampler/độ phủ/ngân sách, tái dựng loss và chạy suy luận CPU trên toàn bộ validation từ tám checkpoint best/final. Kiểm chứng trong gói và kiểm chứng độc lập đều đạt, cùng khớp SHA-256 aggregate `ff5ef23108fa3116ae4b8ebae3ebe945e2c6ee026f2c3ee615429ed8e52c9a99`.

Khởi tạo được tái dựng chính xác bằng phép uniform float32 có nhân–cộng hợp nhất. Khởi tạo native trên PyTorch CPU 2.5 không trùng bit với PyTorch Colab 2.11 (sai lệch tuyệt đối lớn nhất khoảng 1,49×10⁻⁸); không gọi hai môi trường native là đồng nhất bit. Trạng thái ghi trong gói và phép tái dựng chính xác đã khớp hash. Không dùng thời gian CUDA mới để thay benchmark vận hành lịch sử của paper.

## Kết quả đã kiểm chứng độc lập

Tất cả chỉ số bên dưới là **gộp epoch trên validation nguồn của một fold**, qua đường source attention, không phải subject-mean, outer-test, SHHS hoặc ensemble mười fold. Recall là tỷ lệ mẫu thuộc một giai đoạn được nhận diện đúng. Macro-F1 đánh giá cân bằng cả năm giai đoạn.

### Checkpoint tốt nhất theo tiêu chí đã khóa

| Nhánh | Epoch chọn | Macro-F1 | Recall N1 | Recall N3 |
|---|---:|---:|---:|---:|
| Source-only, ngân sách nhỏ | 15 | 0,6585 | 0,1956 | 0,7177 |
| ADAST, ngân sách nhỏ | 11 | 0,6470 | 0,1673 | 0,7184 |
| Source-only, toàn train nguồn | 9 | 0,7174 | 0,5380 | 0,8806 |
| ADAST, toàn train nguồn | 22 | 0,6934 | 0,2338 | 0,8261 |

### Checkpoint cuối epoch 30

| Nhánh | Macro-F1 | Recall N1 | Recall N3 |
|---|---:|---:|---:|
| Source-only, ngân sách nhỏ | 0,6316 | 0,1243 | 0,6059 |
| ADAST, ngân sách nhỏ | 0,5236 | 0,0000 | 0,3954 |
| Source-only, toàn train nguồn | 0,6911 | 0,2847 | 0,5783 |
| ADAST, toàn train nguồn | 0,6784 | 0,1591 | 0,8068 |

Trong validation nguồn có 2.301 mẫu N1. ADAST nhỏ nhận diện đúng 244 mẫu tại epoch 15, 22 mẫu tại epoch 16, một mẫu tại epoch 17 và không mẫu nào từ epoch 18. Từ epoch 19 đến 30 không còn dự đoán N1 qua source attention; cuối lượt cả source và target attention đều không dự đoán N1. Ở ADAST toàn nguồn, cuối lượt source attention nhận diện đúng 366/2.301 mẫu N1; target attention là 365/2.301.

Ngân sách lớn tăng macro-F1 ở cả hai cách học; ADAST lớn không còn mất hoàn toàn N1, nhưng vẫn kém đối chứng source-only về macro-F1 và recall N1. Ở checkpoint cuối của cặp lớn, ADAST tăng recall N3 từ 0,5783 lên 0,8068, đồng thời giảm recall N1 từ 0,2847 xuống 0,1591 và macro-F1 từ 0,6911 xuống 0,6784. Ở các checkpoint tốt nhất, ADAST có N3 F1 cao hơn (0,7880 so với 0,7684), nhưng recall N3 thấp hơn; không mô tả là cải thiện mọi chỉ số N3.

Mẫu kết quả này cho thấy ngân sách là một yếu tố cần kiểm soát, **chưa đủ chứng minh nguyên nhân duy nhất** của suy giảm ADAST. Một fold/một seed không cung cấp ước lượng biến thiên huấn luyện hoặc kiểm định tổng quát hóa. Không lấy các checkpoint mới thay vào kết quả mười fold lịch sử.

## Dấu hiệu cơ chế và bước tiếp theo

Tại epoch 16, hệ số source CE giảm từ 1 xuống 0,1, đồng thời hệ số target pseudo CE tăng từ 0 lên 0,01. ADAST nhỏ giảm recall N1 từ 0,1060 (epoch 15) xuống 0,0096 (epoch 16). Source-only cũng suy giảm khi giảm trọng số source CE, nhưng mức độ khác. Hai thay đổi của ADAST đồng thời xảy ra nên diễn biến này chỉ xác định một mốc đáng kiểm tra, không xác định nhân quả cho một loss riêng lẻ.

Ở cuối ADAST nhỏ, đóng góp số học trung bình của source CE vào tổng loss khoảng 0,1446, loss đối kháng khoảng 0,7307. Đây là trị số loss, **không phải phép đo trực tiếp độ lớn hoặc hướng gradient**. Nhãn giả N1 tại đầu vòng hai ít hơn ở ngân sách nhỏ (71/4.989 so với 226/4.989 ở ngân sách lớn); không có nhãn thật adaptation để chứng nhận các nhãn giả đó đúng.

Kế hoạch tách thành phần sau lượt ngân sách này đã được thực hiện: giữ source CE ở mức 1 trong vòng hai, tắt pseudo CE và tắt đối kháng, cùng đối chứng ADAST mới, đều đã train đủ 30 epoch trên ngân sách toàn nguồn. Giao thức và tiêu chí chọn checkpoint được khóa trước chạy. Xem `ADAST_LOSS_ABLATION_20261004_VI.md` cho trạng thái kiểm chứng và kết quả; không sửa số liệu lịch sử của lượt ngân sách này.

Chưa mở multi-seed hay mười fold ngân sách toàn nguồn và chưa đánh giá checkpoint phát triển mới trên SHHS. Phương pháp và kết quả phát triển đã được bổ sung riêng vào bản Anh, bản dịch Việt và phụ lục; không thay các kết quả SHHS mười fold lịch sử bằng điểm validation một fold.
