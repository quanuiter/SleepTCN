# Gate 6 — kết quả phần CPU và trạng thái benchmark GPU

Ngày thực hiện: **2026-08-14**. Commit giao thức/mã: `bfe0ba3`.

## Trạng thái

| Thành phần | Trạng thái |
|---|---|
| Khóa giao thức Gate 6 | ĐẠT |
| Kiểm thử tự động | ĐẠT — 66/66 |
| Đếm tham số từ checkpoint thật | ĐẠT |
| Manifest mẫu đặc trưng 10-fold | ĐẠT — 10.000 epoch, 78 đối tượng |
| Silhouette 10-fold và t-SNE fold 00 | ĐẠT, tái lập bit-for-bit |
| Latency/throughput/peak VRAM | CHƯA CHẠY — cần cùng một GPU CUDA |

## Độ phức tạp tham số

| Cấu hình | Số mô hình thành phần | Tham số | So với E0 |
|---|---:|---:|---:|
| E0 — 15CNN + BiLSTM | 16 | 248.630 | 1,000× |
| E1 — 15CNN + TCN | 16 | 640.950 | 2,578× |
| E2/E3/E4/E6 — ResNet-1D + TCN | 2 | 1.085.578 | 4,366× |

Kết quả xác nhận góp ý của giảng viên: E2–E6 giảm số mô hình/giai đoạn vận hành từ 16 xuống 2 nhưng
không tiết kiệm tham số. Tổng tham số lớn hơn E0 khoảng **4,37 lần**. Vì vậy luận điểm đúng là “đơn giản
hóa cấu trúc vận hành”, không phải “mô hình nhẹ” hay “parameter-efficient”.

Con số E0 đo từ chính mã hiện tại là 248.630, lệch 130 tham số so với con số 248.500 trong nhận xét.
Báo cáo/khóa luận phải dùng số được tính tự động từ checkpoint và giải thích nếu đối chiếu với nguồn khác.

## Không gian đặc trưng E1 so với E2

Thiết kế: E1 và E2 cùng dữ liệu `paper_raw_v1`, cùng TCN và cùng epoch test; khác bộ trích đặc trưng.
Mỗi fold lấy 200 epoch/lớp, tổng 1.000 epoch/fold. StandardScaler + PCA 20 chiều được fit riêng theo
biểu diễn/fold, sau đó tính Silhouette. Đây là phân tích hỗ trợ thực hiện sau Gate 5.

| Fold | E1 — 15CNN softmax | E2 — ResNet-1D | E2−E1 |
|---:|---:|---:|---:|
| 00 | 0,220369 | 0,103393 | −0,116976 |
| 01 | 0,199117 | 0,104526 | −0,094591 |
| 02 | 0,275121 | 0,122500 | −0,152622 |
| 03 | 0,212542 | 0,101732 | −0,110809 |
| 04 | 0,155970 | 0,095159 | −0,060811 |
| 05 | 0,243576 | 0,144450 | −0,099126 |
| 06 | 0,194911 | 0,090127 | −0,104784 |
| 07 | 0,165756 | 0,056515 | −0,109241 |
| 08 | 0,229768 | 0,112052 | −0,117716 |
| 09 | 0,228892 | 0,117055 | −0,111837 |

Tóm tắt chênh lệch E2−E1:

- trung bình: **−0,107851**;
- trung vị: **−0,110025**;
- khoảng quan sát: **[−0,152622; −0,060811]**;
- E2 cao hơn E1: **0/10 fold**.

Kết quả này **không hỗ trợ** giả thuyết rằng embedding ResNet-1D của E2 tự động tách năm lớp tốt hơn
75 xác suất softmax của E1 dưới phép đo đã khóa. Nó cũng không mâu thuẫn với Macro-F1 E2 cao hơn E1
nhẹ ở Gate 5: Silhouette đo hình học cụm từng epoch, còn TCN sử dụng cấu trúc chuỗi; không gian có cụm
gọn hơn không đồng nghĩa chắc chắn với phân loại chuỗi tốt hơn.

Không được dùng kết quả này để khẳng định 15CNN “giàu thông tin hơn” một cách tổng quát. Softmax 15CNN
đã được tối ưu trực tiếp theo nhãn lớp và có cấu trúc 15×5 xác suất, nên Silhouette theo nhãn có thể thuận
lợi hơn; ResNet embedding có thể mã hóa thông tin hữu ích cho chuỗi nhưng không tạo cụm Euclid gọn.

## Tính tái lập

Ba artifact phân tích đặc trưng được chạy lại và khớp SHA-256 bit-for-bit:

- `feature_space_report.json`:
  `bddcdb5415c291db7ad3f0fb61765f5458643027f2d5dd0640b911c3ea5e267f`;
- `tsne_points.csv`:
  `7a9fdac625a9e64439b0dce507d959b678e9faccaabf1957e76ccd082feb1c1a`;
- `tsne_E1_vs_E2.png`:
  `1eb21c4b4e69904ec28fc9be0d9560878bbd9ddcdcfc6559777d3a0dce437c17`.

Manifest mẫu và CSV điểm t-SNE chứa khóa đối tượng/bản ghi/epoch nên chỉ giữ cục bộ, không đẩy lên
GitHub. Chúng có thể tái tạo bit-for-bit bằng seed và runbook đã khóa. Báo cáo tổng hợp cùng hình không
định danh được phép lưu Git.

## Việc còn lại để đóng Gate 6

Chỉ còn benchmark CUDA theo `GATE6_PROTOCOL_AND_RUNBOOK.md`: ba vòng × sáu mô hình, sau đó chạy
`validate_gate6_artifacts.py`. Không cần huấn luyện và không cần mở test lại.
