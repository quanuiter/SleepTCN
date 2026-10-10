# Huấn luyện bổ sung trên Colab T4 — 04/10/2026

Đã hoàn tất bốn cặp unweighted/weighted ở fold 6–9. Tái sử dụng sáu cặp CPU fold 0–5 đã kiểm chứng; tổng cộng đủ 20 checkpoint được chọn bằng validation nguồn. Đây là can thiệp loss trên đặc trưng của encoder E3 đã khóa, không phải huấn luyện lại toàn bộ encoder hoặc một lượt ADAST.

| Fold | Nhánh | Epoch đã chạy | Epoch chọn | Train + validation (giây) |
|---|---|---:|---:|---:|
| 6 | Unweighted | 67 | 37 | 135,83 |
| 6 | Weighted | 49 | 19 | 102,96 |
| 7 | Unweighted | 50 | 20 | 105,92 |
| 7 | Weighted | 48 | 18 | 101,66 |
| 8 | Unweighted | 67 | 37 | 147,79 |
| 8 | Weighted | 64 | 34 | 140,98 |
| 9 | Unweighted | 51 | 21 | 115,32 |
| 9 | Weighted | 53 | 23 | 118,79 |

Tổng thời gian train và validation của tám mô hình mới: **969,25 giây (~16 phút 9 giây)**. Không bao gồm chuẩn bị đặc trưng, upload/download, kiểm chứng hoặc suy luận SHHS; không dùng số này để thay thế benchmark vận hành của paper.

## Giao thức và kiểm chứng

- Giữ recipe: seed 123, batch 8, Adam learning rate 0,0005, tối đa 300 epoch, patience 30, gradient clipping 1.
- Các cặp fold 6–9 khởi tạo mới trên CUDA, không tiếp tục checkpoint CPU dở dang của fold 6. Checkpoint CPU cũ được giữ nguyên.
- Mỗi cặp có cùng khởi tạo và thứ tự minibatch. Không đưa SHHS lên Colab và không dùng kết quả SHHS để chọn checkpoint.
- Cả bốn gói kết quả qua kiểm chứng SHA-256, nguồn code/model, checkpoint chọn, class weights và stopping rule. Replay validation và outer test nguồn độc lập trên CPU cho **cùng ma trận nhầm lẫn và số liệu** với báo cáo CUDA.
- Hash ZIP tổng khớp giá trị in tại Colab: `fd4c16d3a566428ce2e48dea8418fe72588f36b018fe25e1f697d02eb81398b8`.
- Đã khóa đủ 20 checkpoint trước lượt suy luận SHHS mới. Backend được ghi rõ: fold 0–5 CPU/PyTorch 2.5.1, fold 6–9 CUDA/PyTorch 2.11.0; không mô tả đây là một chiến dịch đồng nhất backend.

## Đánh giá SHHS đã hoàn tất

Lượt suy luận và chấm tổ hợp mười fold trên 180 người SHHS bắt đầu lúc 08:57:57 (giờ Việt Nam), chạy cục bộ trên CPU; không phải CPU training. Hoàn tất trong **2.484,95 giây (~41 phút 25 giây)**, gồm suy luận, chấm điểm và kiểm chứng trong lượt; dưới giới hạn 18.000 giây. Chấm 169.012 epoch có nhãn sau khi suy luận toàn bản ghi. Hai bản kiểm chứng đều đạt và cùng trỏ đến SHA-256 aggregate `44211a4ccf92a1d2b6d5d46bc02a2b9bba140f1d2a132a75e03c8a534c9006ef`, đã đối chiếu với tệp thực tế. Đã kiểm tra 20 checkpoint nguồn, 180 tệp dự đoán và 3.600 mảng xác suất fold/nhánh; tái tính ma trận nhầm lẫn và bootstrap bắt cặp.

| Chỉ số SHHS | Không trọng số | Có trọng số |
|---|---:|---:|
| Macro-F1 trung bình theo người | 0,5530 | 0,5514 |
| Macro-F1 gộp | 0,5930 | 0,5928 |
| Accuracy | 0,6892 | 0,6663 |
| N3 precision | 0,9462 | 0,9342 |
| N3 recall | 0,2345 | 0,3225 |
| N3 F1 | 0,3759 | 0,4795 |
| N2 recall | 0,7260 | 0,6728 |
| N3 bị nhầm thành N2 | 74,97% | 65,49% |
| N2 bị nhầm thành N3 | 0,38% | 0,63% |

Bootstrap 10.000 lần bắt cặp theo người, chiều **có trọng số trừ không trọng số**:

- N3 recall +0,0880; CI 95% [0,0805; 0,0954]. N3 F1 +0,1036; CI [0,0938; 0,1137].
- N3 precision −0,0120; CI [−0,0187; −0,0050]. N2 recall −0,0531; CI [−0,0577; −0,0488].
- Macro-F1 trung bình theo người −0,0016; CI [−0,0049; 0,0017]. Macro-F1 gộp −0,0002; CI [−0,0032; 0,0027].

**Diễn giải:** can thiệp tăng nhận diện N3 trong so sánh bắt cặp nhưng không khắc phục toàn bộ thiếu hụt. Điểm tổng thể không có lợi ích được xác lập; accuracy, N2 recall và N3 precision giảm. CI chứa không không chứng minh tương đương hoặc không thua kém. Trên outer test nguồn đủ 78 người, macro-F1 gộp giảm 0,7878 → 0,7840; N3 recall tăng 0,8059 → 0,8724 nhưng N3 precision giảm 0,8121 → 0,7404 và F1 giảm 0,8090 → 0,8010.

Hai tổ hợp dùng cùng số fold và encoder tương ứng, không so một mô hình weighted với E3 lịch sử mười fold. Fold 0 tái sử dụng cặp pilot đã kiểm chứng, nên pilot không phải bằng chứng độc lập. Đây là phân tích bổ sung trên quần thể đã được xem; bootstrap điều kiện trên checkpoint đã khớp, không bao gồm biến thiên seed hoặc backend. Không đưa thời gian mới vào benchmark V100 lịch sử của bài.

ADAST chưa train trên Colab tại thời điểm cập nhật: người dùng đã cho phép dữ liệu nguồn có nhãn và năm người adaptation không nhãn đã bỏ ID; gói đang upload. Không dùng dữ liệu 180 người đánh giá để adaptation hoặc đưa chúng lên cloud. Kết quả pilot một fold, ensemble lịch sử và chiến dịch bổ sung này được giữ tách biệt.
