# Rà ngân sách ADAST trên CPU — 03-10-2026

Phân tích lại lượt ADAST/source-only fold 0 seed 123 đã kiểm chứng; không train
lại, không đọc nhãn adaptation hoặc điều chỉnh theo điểm SHHS. Chương trình:
`scripts/analyze_adast_cpu_budget.py`. Kết quả tổng hợp nằm trong
`runs/teacher_revision_cpu_20261003/adast_budget_diagnostic.json`.

## Số mẫu nguồn thực sự được trình bày

- Tập train nguồn: 157.200 epoch có nhãn.
- Mỗi epoch tối ưu hóa: 38 batch × 128 = 4.864 mẫu nguồn.
- Hai round × 15 epoch: tổng 145.920 lượt trình bày mẫu, tương đương **0,9282
  lượt qua toàn bộ nguồn**, không phải 30 lượt qua nguồn.
- Do permutation độc lập mỗi epoch, 95.823 mẫu nguồn khác nhau được thấy,
  tương ứng **60,96%** tập train. Đây là số đo tái dựng từ các source order đã
  ghi hash; source-only và ADAST có cùng thứ tự/mẫu nguồn.

Sự cân bằng ngân sách giữa hai nhánh đã được giữ. Tuy nhiên, số update nhỏ khiến
kết quả khó đại diện một baseline ADAST được huấn luyện đủ trên dữ liệu nguồn.
Không thể suy từ số mẫu này rằng tăng ngân sách chắc chắn cải thiện kết quả.

## Pseudo-label trên 4.989 epoch adaptation không nhãn

| Round | W | N1 | N2 | N3 | REM | Trọng số giám sát pseudo-label |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4.872 | 0 | 117 | 0 | 0 | 0 |
| 1 | 1.746 | 124 | 1.781 | 222 | 1.116 | 0,01 |

Round 0 được sinh từ mô hình ban đầu chưa huấn luyện, và **không góp loss giám
sát đích vì trọng số bằng 0**. Không quy việc suy giảm cho histogram round 0.
Round 1 có đủ năm lớp nhưng N1 ít; không có nhãn adaptation để xem đây là lỗi
hay tần suất sinh lý thực. Histogram không thay thế việc đo chất lượng pseudo-label.

Trọng số source CE giảm từ 1 xuống 0,1 ở round 1, trong khi trọng số adversarial
giữ 1. Đây là lịch đã triển khai từ upstream; chưa có ablation để quy nguyên nhân
suy giảm cho lịch loss, pseudo-label hoặc attention. Hiệu năng nguồn của ADAST
cũng thấp, nên không nên giải thích toàn bộ bằng chuyển quần thể.

## Tác động đến kế hoạch CPU

Lượt nhỏ đã chạy mất khoảng 121 giây source-only + 251 giây ADAST cho mỗi cặp.
Nếu đổi giao thức để mỗi epoch xem hết nguồn, số batch nguồn đầy đủ sẽ khoảng
1.228 thay vì 38. Ngoại suy tuyến tính theo số update cho khoảng **3,3 giờ train
mỗi cặp fold**, hoặc **33 giờ cho mười cặp**, chưa tính chuẩn bị/suy luận/kiểm
chứng. Đây là dự phóng từ lượt nhỏ, không phải benchmark chiến dịch thực tế;
việc lặp target loader còn là một thay đổi giao thức cần ghi nhận.

Vì người dùng giới hạn năm giờ, **chưa khởi chạy chiến dịch ADAST mười fold với
ngân sách nguồn đầy đủ**. Cần chốt ngân sách có căn cứ và đo một lượt phù hợp
trước mở rộng. Không tự giảm ngân sách để làm cho baseline chạy nhanh rồi gọi
là đã đáp ứng yêu cầu benchmark ngang hàng.
