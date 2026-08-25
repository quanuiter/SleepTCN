# Kết quả Gate 6 — độ phức tạp, tốc độ và không gian đặc trưng

> Lưu ý trạng thái (2026-08-22): đây là bản đóng băng tại Gate 6. Sau đó dự án đã hoàn tất Gate 7--8,
> SHHS1 zero-shot và một lần lặp đủ 60 lượt với seed 123; xem `STATUS_V2.md`. Các số benchmark Gate 6
> vẫn chỉ thuộc checkpoint seed 42 và Tesla V100.

Ngày hoàn tất: **2026-08-14**.

## Trạng thái kiểm định

Gate 6 đã **ĐẠT** cả ba thành phần:

- số tham số từ checkpoint thật;
- latency, throughput và peak VRAM trên cùng GPU;
- Silhouette 10-fold và t-SNE mô tả trên cùng các epoch E1/E2.

Báo cáo tổng hợp `runs/v2/analysis/gate6_validation_report.json` trả `status: passed`.

Nguồn benchmark GPU:

- GPU: Tesla V100-PCIE-16GB;
- Python 3.10.13;
- PyTorch 2.5.1+cu124, CUDA 12.4, cuDNN 9.1;
- checkpoint fold 00, seed 42;
- input float32 `(1, 100, 1, 3000)`;
- ba vòng xáo thứ tự, mỗi vòng 20 lượt làm nóng và 100 lượt đo;
- tổng 300 forward pass có đồng bộ CUDA cho mỗi cấu hình;
- không gồm đọc đĩa, preprocessing, ghi cache hoặc training.

SHA-256:

- benchmark JSON:
  `e1cdba979d9f434bb19803ea5341052239f0e98c89f1bff95c21287e99d81cc7`;
- benchmark log:
  `94602c4cc18223bfac34ee3adafda0447b5899717be8e3221819bcf73d0eeb9d`.

## Kết quả benchmark

| Cấu hình | Tham số | Latency trung vị (ms/100 epoch) | p95 (ms) | Throughput (epoch/s) | Peak VRAM cấp phát (MiB) | Nhanh hơn E0 |
|---|---:|---:|---:|---:|---:|---:|
| E0 — 15CNN + BiLSTM | 248.630 | 13,4315 | 15,4400 | 7.445 | 58,81 | 1,00× |
| E1 — 15CNN + TCN | 640.950 | 12,5111 | 13,6892 | 7.993 | 60,31 | 1,07× |
| E2 — ResNet-1D + TCN | 1.085.578 | 3,5750 | 3,6531 | 27.972 | 75,54 | 3,76× |
| E3 — ResNet-1D + TCN | 1.085.578 | 3,5687 | 3,6813 | 28.021 | 75,54 | 3,76× |
| E4 — ResNet-1D + TCN | 1.085.578 | 3,5739 | 3,6427 | 27.980 | 75,54 | 3,76× |
| E6 — ResNet-1D + TCN | 1.085.578 | 3,5756 | 3,6426 | 27.967 | 75,54 | 3,76× |

E2/E3/E4/E6 dùng cùng kiến trúc nên các chênh lệch rất nhỏ giữa chúng là nhiễu đo, không phải tác động
của preprocessing. E2 nhanh hơn E1 khoảng **3,50×** trong phép đo này.

## Đánh đổi độ phức tạp

So với E0, ResNet-1D + TCN:

- giảm số mô hình thành phần từ 16 xuống 2;
- nhanh hơn khoảng **3,76×**;
- tăng số tham số khoảng **4,37×**;
- tăng peak VRAM cấp phát khoảng **28,4%**: 58,81 lên 75,54 MiB;
- tăng VRAM cấp phát bổ sung trong forward từ khoảng 24,62 lên 38,23 MiB.

Vì vậy thông điệp khoa học đúng là: kiến trúc ResNet-1D + TCN **đơn giản hơn về số mô hình/giai đoạn
vận hành và nhanh hơn khi suy luận**, nhưng **không tiết kiệm tham số hoặc bộ nhớ GPU**. Không dùng cụm
“mô hình nhẹ” hay “parameter-efficient”.

Con số tăng tốc thực nghiệm là khoảng **3,76×**, không phải 8,2×. Không dùng con số lịch sử 8,2× trong
abstract/kết luận nếu không có một giao thức khác được mô tả và kiểm chứng riêng.

## Không gian đặc trưng

Phân tích E1/E2 dùng cùng `paper_raw_v1`, cùng TCN và cùng epoch test. Mỗi fold lấy 200 epoch/lớp;
tổng 10.000 epoch, đủ 78 đối tượng. StandardScaler + PCA 20 chiều được fit riêng cho từng fold/biểu diễn.

Silhouette E2 thấp hơn E1 ở **10/10 fold**:

- chênh lệch E2−E1 trung bình: −0,107851;
- trung vị: −0,110025;
- khoảng quan sát: [−0,152622; −0,060811].

Kết quả này bác bỏ cách diễn đạt đơn giản rằng embedding ResNet “tách lớp tốt hơn” theo hình học cụm
Euclid đã khóa. Nó không chứng minh 15CNN dự đoán chuỗi tốt hơn: 15CNN softmax đã được tối ưu trực tiếp
theo lớp, còn TCN có thể khai thác thông tin thời gian trong embedding ResNet dù các cụm từng epoch kém
tách biệt hơn. Phân tích này được thực hiện sau Gate 5 nên chỉ là bằng chứng hỗ trợ, không phải giả thuyết
xác nhận.

## Tổng hợp với Gate 5

- E3 có Macro-F1 test cao nhất: 0,790443.
- E3−E6 là kết quả thống kê mạnh nhất: Δ Macro-F1 +0,021319, CI 95%
  [+0,012179; +0,030698], p Holm=0,001185.
- E2−E1 chỉ tăng Macro-F1 +0,003251 và chưa có ý nghĩa thống kê sau Holm.
- ResNet+TCN vẫn có đóng góp thực dụng rõ ràng: giảm số mô hình vận hành và tăng tốc suy luận khoảng
  3,76×, dù không chứng minh ưu thế thống kê về Macro-F1 so với 15CNN+TCN trong seed 42.

Thông điệp phù hợp cho khóa luận/bài báo:

> ResNet-1D + TCN đem lại quy trình ít mô hình thành phần hơn và suy luận nhanh hơn, đổi lại số tham số
> và peak VRAM cao hơn. Cải thiện chất lượng mạnh nhất trong thí nghiệm hiện tại đến từ lựa chọn xử lý
> biên độ E3 so với z-score E6, không phải từ bằng chứng rằng embedding ResNet tách lớp tốt hơn.

## Giới hạn

- Benchmark chỉ đo một GPU V100, một kích thước batch/chuỗi và forward pass; không đại diện cho mọi
  phần cứng hoặc toàn bộ thời gian hệ thống có I/O.
- Training chính thức chỉ có seed 42.
- Phân tích không gian đặc trưng thực hiện sau khi đã xem kết quả Gate 5.
- Chưa có bằng chứng SHHS, domain shift, zero-shot hoặc đa kênh.

## Bước tiếp theo

Gate 6 đã đóng. Bước tiếp theo là Gate 7: tạo bảng/hình xuất bản, viết chương phương pháp–kết quả–thảo
luận, lập ma trận bằng chứng cho từng tuyên bố và quyết định có chạy thêm seed 123/2025 như nghiên cứu
độ ổn định riêng hay không. Không thay đổi kết luận E0–E6 dựa trên test đã mở.
