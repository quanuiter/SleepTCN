# Lượt xác nhận nhỏ ADAST — 04/10/2026

## Mục tiêu và phạm vi đã chốt trước khi chạy

Thử lại ba cách học trên **fold 1, seed 123**: source-only, ADAST đối chứng và
ADAST giữ trọng số học nguồn ở mức 1 trong vòng hai. Chỉ thay fold so với lượt
phát triển trước; cùng khởi tạo, thứ tự mẫu và ngân sách giữa ba mô hình mới.
Đây là kiểm tra trên cách chia dữ liệu khác, không phải mười fold hay nhiều seed.

Mỗi mô hình học đủ 152.825 đoạn train nguồn trong mỗi epoch, gồm 1.194 update
(batch 128, giữ batch cuối 121) × 30 epoch = 35.820 update. Validation có 23.881
đoạn nguồn. Chỉ dùng lại đúng năm người SHHS thích nghi không nhãn (4.989 đoạn).
Không đóng gói/truy cập test nguồn fold 1 hoặc 180 người SHHS test.

Người dùng trực tiếp đồng ý tải train/validation Sleep-EDF fold 1 đã bỏ ID và
dữ liệu không nhãn của năm người SHHS đến notebook mới `19wRkGw…` trong cuộc
trò chuyện này. Phiên Colab đã được kết nối lại và giao diện xác nhận T4.
Gói lớn được tạo trong thư mục tạm ổ C, vì ổ D chỉ còn khoảng 0,48 GB. Không
xóa hay ghi đè dữ liệu, checkpoint hoặc giao thức cũ.

## Quy tắc đánh giá và dừng

Chọn checkpoint bằng macro-F1 validation nguồn qua source attention, giữ epoch
đầu khi bằng điểm; báo riêng epoch 30. Chấm đủ precision/recall/F1 năm lớp qua
cả hai đường attention trên cùng validation nguồn. Target attention không chọn
checkpoint; điểm này không phải hiệu năng SHHS.

Ứng viên giữ học nguồn được xem là đáng cân nhắc mở rộng nếu điểm được chọn
vượt ADAST đối chứng và source-only về macro-F1, tăng F1 N1 so với ADAST đối
chứng, không giảm quá 0,01 F1 ở W/N2/N3/REM; đồng thời đường target attention
không giảm quá 0,01 macro-F1 hoặc F1 N1/N3 so với đối chứng. Các ngưỡng này là
quy tắc quyết định phát triển mô tả, không phải kiểm định tương đương hoặc
không thua kém lâm sàng. Báo kết quả dù đạt hay không, cùng checkpoint cuối.

Fold 0 trước đây vẫn giữ nguyên, gồm đánh đổi REM đã quan sát. Một kết quả đẹp
ở fold 1 không xóa đánh đổi fold 0 hoặc xác nhận cải thiện mọi quần thể. Không
tự mở rộng seed/fold, không chọn lại tiêu chí sau khi thấy điểm.

Giới hạn chung ba mô hình là 18.000 giây. Lưu model/optimizer/RNG sau từng epoch;
không train CPU thay thế, không tự restart hoặc tiếp tục sau timeout. Dự kiến
khoảng 45–60 phút train và validation trên T4, dựa trên lượt trước; không gồm
thời gian đóng gói, upload hay kiểm chứng độc lập.

## Trạng thái

**Cả ba mô hình đã chạy đủ 30 epoch trên Tesla T4**, mỗi mô hình 35.820 update.
Thời gian training và validation là **2.975,78 giây (49 phút 35,78 giây)**,
không gồm upload và kiểm chứng; không dùng thời gian T4 này thay benchmark cũ.
ZIP tải về khớp SHA thực sự xuất trong log Colab. Kiểm chứng trong gói và độc
lập cùng đạt, cùng khớp hash aggregate. Kiểm chứng độc lập mất 1.166,22 giây,
tính lại 93 bộ chẩn đoán và chấm CPU toàn validation từ sáu checkpoint
best/final qua cả hai attention. Khởi tạo, sampling, coverage, hệ số loss,
ngân sách và logits hai ADAST trong vòng một đều khớp giao thức. Kiểm thử
runner, launcher, cổng kiểm chứng và quy tắc đọc kết quả đạt **17/17**.
Script phân tích
`scripts/analyze_adast_small_confirmation_results.py` chỉ nhận kết quả đã
được kiểm chứng độc lập, báo riêng checkpoint chọn/cuối và cả hai attention;
đã chạy trên kết quả được kiểm chứng, xuất summary và CSV đầy đủ precision,
recall, F1, support, ma trận nhầm lẫn và diễn biến mỗi epoch.
Lịch theo dõi hiện có đã cập nhật cho đúng notebook và lượt ba cấu hình này,
giữ im lặng khi bình thường và không tự mở rộng thử nghiệm. Giao thức ở
`configs/adast_small_confirmation_v1_20261004.json`; các hồ sơ cục bộ ở
`runs/adast_small_confirmation_20261004/` chỉ giữ con trỏ/hash của gói tạm.

## Kết quả dễ đọc

Các số dưới đây là trên **validation nguồn fold 1**, không phải SHHS test.
Macro-F1 là điểm cân bằng giữa năm giai đoạn; cao hơn tốt hơn. F1 từng lớp
kết hợp khả năng tìm đúng và tránh dự đoán nhầm sang lớp đó.

| Cấu hình | Epoch chọn | Macro-F1 | F1 N1 | F1 N3 | F1 REM |
|---|---:|---:|---:|---:|---:|
| Không thích nghi | 10 | 0,7370 | 0,3200 | 0,8568 | 0,7224 |
| ADAST đối chứng | 27 | 0,6907 | 0,2175 | 0,8096 | 0,6852 |
| ADAST giữ trọng số học nguồn | 27 | 0,6892 | 0,2302 | 0,8161 | 0,6680 |

So với ADAST đối chứng, bản sửa tăng F1 N1 **0,0127** và F1 N3 **0,0065**,
nhưng giảm F1 REM **0,0172**; macro-F1 giảm **0,0015**. N1 recall tăng
0,1502 → 0,1627, REM recall giảm 0,7376 → 0,6980. Đây là lợi ích nhỏ ở N1
đi kèm đánh đổi REM, cùng hướng với checkpoint được chọn ở fold 0 trước đó.
Ở fold 1, macro-F1 của đối chứng không thích nghi vẫn cao nhất.

N3 recall cao không đồng nghĩa F1 N3 cao: ADAST đối chứng tăng N3 recall so
với source-only (0,8816 → 0,9644), nhưng precision giảm (0,8333 → 0,6975),
khiến F1 giảm (0,8568 → 0,8096). Vì vậy không chọn mô hình chỉ bằng N3 recall.

Qua **target attention trên cùng validation nguồn**, macro-F1 của ADAST đối
chứng/bản sửa là 0,6817/0,6727; F1 N3 là 0,8003/0,7864. Bản sửa tăng recall
N3 ở đường này nhưng giảm precision, nên F1 giảm. Đây là kiểm tra đường xử
lý của mô hình trên dữ liệu nguồn, không phải kết quả chuyển quần thể mới.

| Cấu hình, tại epoch 30 | Macro-F1 | F1 N1 | F1 N3 | F1 REM |
|---|---:|---:|---:|---:|
| Không thích nghi | 0,7176 | 0,2415 | 0,8503 | 0,7184 |
| ADAST đối chứng | 0,6857 | 0,2175 | 0,8077 | 0,6672 |
| ADAST giữ trọng số học nguồn | 0,6814 | 0,2158 | 0,7999 | 0,6653 |

Tại epoch cuối, bản sửa không còn lợi ích N1 so với ADAST đối chứng. Không
đánh đồng kết quả checkpoint được chọn với diễn biến cuối lượt.

## Quyết định theo quy tắc đã khóa

Bản sửa **không đạt điều kiện mở rộng**: macro-F1 không vượt hai đối chứng;
F1 REM qua source attention giảm hơn 0,01; F1 N3 qua target attention giảm
hơn 0,01. Điều này giúp chốt rằng chỉ giữ trọng số học nguồn chưa tạo được
cấu hình cải tiến đồng đều. Không tự train thêm fold/seed hoặc chấm SHHS cho
bản sửa này. Giữ riêng fold 0 và kết quả mười fold lịch sử. Lượt xác nhận
đã làm xong; kiểm chứng kỹ thuật đạt, còn gate phát triển không đạt là kết
quả nghiên cứu, không phải lỗi chương trình.

## Hồ sơ kiểm chứng

- ZIP kết quả SHA-256: `6200fef5a7eab470fe8c054e76661bffac2fe96fd8e6fe815ed49b9793e0b762`.
- Aggregate SHA-256: `579ce087700074764044f13823829803003f75ab1986cc58dd0689b67785bf53`.
- Phân tích: `Reports/analysis/adast_small_confirmation_20261004/summary.json`,
  `all_class_metrics.csv` và `manifest.json`.
- Con trỏ kiểm chứng: `runs/adast_small_confirmation_20261004/verified_results_pointer.json`.
- Bản Anh 16 trang, bản Việt 17 trang và phụ lục 12 trang đã bổ sung kết quả,
  dựng lại và rà trực quan; thêm bảng S28–S29, giữ nguyên bảng lịch sử.
  Bàn giao PDF/gói nguồn được xác nhận trong
  `tmp/adast_confirmation_revision_20261004/delivery_verification.json`.
