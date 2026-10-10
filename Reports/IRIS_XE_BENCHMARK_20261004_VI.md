# Thử nghiệm Iris Xe cho lượt TCN đang còn dở — 04/10/2026

## Kết luận

**Không nên chuyển chiến dịch class-weighted hiện tại sang Iris Xe–DirectML.** Thiết bị được nhận đúng và có thể thực hiện các bước huấn luyện sau khi bổ sung bộ điều chỉnh bố trí tensor trong mã thử nghiệm. Tuy nhiên, bài đo này không cho thấy lợi ích tốc độ: batch 8 bản ghi chạy trên Iris Xe mất khoảng 0,46–0,47 giây/bước, so với khoảng 0,27–0,30 giây trên môi trường CPU hiện tại. Đây không phải kết luận về mọi tác vụ hoặc mọi GPU Intel.

Chiến dịch mười fold vẫn ở trạng thái dừng trước đó. Không tiếp tục fold 6, không thay đổi checkpoint, không suy luận/chấm thêm SHHS và không cập nhật số liệu bài báo từ bài đo thiết bị này.

## Thiết bị và môi trường

- GPU được chọn theo tên: Intel Iris Xe Graphics, không chọn bộ điều hợp hiển thị ảo. Driver đọc được: 32.0.101.7076.
- Môi trường nghiên cứu hiện tại giữ nguyên PyTorch 2.5.1+cpu.
- Môi trường thử riêng `.venv-iris-benchmark`: Python 3.11.9, PyTorch 2.4.1+cpu, torch-directml 0.2.5.dev240914, NumPy 1.26.4. Hậu tố `+cpu` của gói torch không có nghĩa bài đo DirectML chạy hoàn toàn trên CPU: tensor/mô hình được đưa lên thiết bị DirectML đã xác định là Iris Xe.
- PyTorch 2.4.1 là phiên bản gói DirectML yêu cầu khi cài đặt; không hạ phiên bản môi trường nghiên cứu. Kiểm tra phụ thuộc riêng bằng `pip check` thành công.

## Cách đo

Sử dụng nguyên lớp `SleepTCN(input_dim=128)`, bộ ghép chuỗi và hàm loss của dự án, với feature cache nguồn của fold 6; không đọc dữ liệu đích hay nạp checkpoint mô hình chiến dịch. Dùng cùng trọng số khởi tạo, cùng feature/nhãn và class weight lấy từ toàn bộ tập huấn luyện nguồn của fold này.

Hai trường hợp: batch ngắn chẩn đoán `(2, 256, 128)` và batch 8 bản ghi huấn luyện thật, ghép đầy đủ thành `(8, 1236, 128)`. Batch thật không bị cắt thành 256 epoch. Đây là một batch cố định, không đại diện đầy đủ cho phân bố thời gian cả fold.

- CPU sử dụng 4 luồng; mọi trường hợp dùng float32.
- Kiểm tra đầu ra, weighted cross-entropy, gradient từng tham số và một bước Adam đối chiếu CPU cùng phiên bản. Dropout tắt **chỉ trong phép đối chiếu số học** để tránh so sánh các mặt nạ ngẫu nhiên khác nhau.
- Đo ở chế độ huấn luyện với dropout 0,2, Adam learning rate 0,0005, clip gradient norm 1,0. Mỗi loạt có 2 bước làm nóng và 7 bước đo.
- Thời gian bao gồm forward, loss, backward, clip, Adam và đọc tensor sau cập nhật về CPU để chờ GPU hoàn tất. Có cả loạt tensor đã ở thiết bị và loạt đưa batch từ CPU lên thiết bị mỗi bước. Bảng dưới dùng loạt có chuyển batch.
- Các đối chiếu chính được chạy lại; các lượt CPU/GPU đo tuần tự. Mỗi tiến trình benchmark có giới hạn 300 giây, không chạy huấn luyện nhiều giờ.

## Kết quả tốc độ

Khoảng dưới đây là **hai trung vị của hai lượt đo**, không phải khoảng tin cậy thống kê.

| Cấu hình | Batch 8 bản ghi: giây/bước |
|---|---:|
| CPU hiện tại, PyTorch 2.5.1, mã mô hình nguyên bản | 0,268–0,302 |
| CPU đối chứng, PyTorch 2.4.1, mã nguyên bản | 0,272 |
| CPU 2.4.1, bộ điều chỉnh tensor giống phép thử GPU | 0,243–0,246 |
| Iris Xe–DirectML 2.4.1, bộ điều chỉnh tensor | 0,461–0,469 |
| Iris Xe, bộ điều chỉnh tensor và Adam `foreach=False` | 0,461 |

Như vậy, Iris Xe mất khoảng 1,5–1,8 lần thời gian CPU hiện tại trong các lượt đo này; so với CPU cùng phiên bản và cùng bộ điều chỉnh tensor, khoảng 1,9 lần. Tắt `foreach` không loại bỏ được cảnh báo fallback của Adam và không tạo ra lợi thế tốc độ. Không dùng các bước đo ngắn này để dự báo chắc chắn thời gian một fold hoặc một lượt ADAST đầy đủ.

## Vấn đề số học và xử lý trong phép thử

1. Đưa mô hình nguyên bản trực tiếp lên DirectML cho đầu ra gần khớp CPU (sai khác tuyệt đối cực đại khoảng `3,34e-6`) và loss khớp, **nhưng gradient không khớp**. Trong batch ngắn, sai khác gradient cực đại toàn mô hình khoảng `1,0054`; 40/42 tensor tham số không vượt qua tiêu chí so sánh. Đã dừng nhánh này trước phép đo Adam/tốc độ, không dùng nó để tạo checkpoint nghiên cứu.
2. Chẩn đoán riêng cho Linear, LayerNorm, GELU, Conv1d ở sáu dilation, LayerNorm với transpose, masked residual và loss cho thấy các kiểm tra đơn lẻ vượt qua tiêu chí, trong khi gradient của tổ hợp `TemporalBlock` không khớp. Điều này thu hẹp nơi cần kiểm tra, **chưa xác định chắc chắn kernel hoặc driver nào là nguyên nhân**.
3. Bộ điều chỉnh cục bộ đưa đầu vào của các module lá về tensor contiguous đã giúp phép đối chiếu toàn mô hình vượt qua trên cả hai batch. Không thay đổi kiến trúc, kích thước, weights hay công thức loss; chỉ thay đổi cách vật chất hóa các view trong bộ nhớ. Không đưa bộ điều chỉnh này vào mã huấn luyện đã đóng băng.
4. Với batch thật sau điều chỉnh: sai khác đầu ra cực đại `3,34e-6`, gradient `1,15e-6`, tham số sau bước Adam `2,88e-6`. Đây là kiểm chứng một bước ở hai batch, không phải bảo đảm tương đương toàn bộ quá trình huấn luyện, RNG/dropout hoặc khả năng resume xuyên backend.
5. Adam trên DirectML báo `aten::lerp.Scalar_out` chưa hỗ trợ và fallback về CPU, kể cả khi `foreach=False`. Log lưu được cảnh báo này. Không khẳng định fallback là nguyên nhân duy nhất làm GPU chậm hơn.

## Tệp tái lập và bảo toàn

- Script chính: `scripts/benchmark_iris_tcn.py`; chẩn đoán phép toán: `scripts/diagnose_iris_gradients.py`.
- Kết quả máy đọc, các lần lặp và log: `runs/iris_xe_benchmark_20261004/`. Không chứa ID người tham gia trong bản báo cáo này.
- Môi trường thử được giữ lại và thêm vào danh sách Git ignore. Không cài đè dependency nghiên cứu, không sửa driver hoặc thiết lập nguồn điện.
- Checkpoint đã giữ ở fold 6 vẫn khớp SHA-256 đã kiểm chứng trước phép thử; trạng thái chiến dịch không đổi. Mã mô hình và huấn luyện dùng chung không bị chỉnh sửa.

Lệnh tái lập phép thử GPU có điều chỉnh (chạy ngắn, không chạy chiến dịch):

```powershell
.venv-iris-benchmark/Scripts/python.exe scripts/benchmark_iris_tcn.py --backend dml --layout-contiguous --output runs/iris_xe_benchmark_20261004/recheck_dml.json --bounded-child
```

**Hướng tiếp theo:** giữ nguyên checkpoint và hoàn tất phần TCN còn lại bằng CPU khi người dùng cho phép chạy tiếp; nếu có GPU khác, đo và kiểm tra tương thích riêng trước khi chuyển backend. Bài đo Iris Xe này không tự cấp thêm ngân sách huấn luyện.
