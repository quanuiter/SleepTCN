# Phản hồi góp ý thầy Trí — cập nhật 07-10-2026

Đối tượng góp ý ban đầu là bản `Sleep_Stage.pdf`. Dưới đây là những thay đổi đã
thực hiện trong bản hiện tại, để tác giả rà trước khi gửi thầy.

## Kết quả bổ sung mới nhất: đối chiếu ADAST đủ nguồn, đủ mười fold

Đánh giá phụ dùng checkpoint tốt nhất trên validation nguồn cũng đã hoàn tất:
macro-F1 trung bình theo người trên SHHS của source-only/ADAST là 0,5073/0,5396,
chênh +0,0322, CI [0,0234; 0,0412]; recall N3 là 0,2828/0,6095. F1 N1/N2/REM
và hiệu suất outer test nguồn giảm. Cùng hướng lợi ích và chi phí xuất hiện ở cả
checkpoint được chọn và epoch 30; đây là hai phân tích trên cùng người/mô hình,
không phải hai quần thể xác nhận độc lập. Bảng S32-S33 trình bày kết quả tổng hợp
và đủ năm lớp; epoch 30 vẫn là đối chiếu ngân sách chính. Không train lại hoặc
chọn lại cấu hình theo SHHS. Xem `ADAST_FULLSOURCE_BEST_COMPLETION_20261007_VI.md`.

Đã chấm và kiểm chứng 20 mô hình source-only/ADAST học toàn nguồn trong 30 epoch,
trên cùng 180 người SHHS và toàn bộ outer test Sleep-EDF. So sánh chính dùng
epoch 30 ở cả hai mức ngân sách; không chọn mô hình theo điểm SHHS.

| SHHS: chỉ số | Source-only toàn nguồn | ADAST toàn nguồn |
|---|---:|---:|
| Macro-F1 trung bình theo người | 0,5010 | 0,5455 |
| Macro-F1 gộp | 0,5478 | 0,5918 |
| Recall N3 | 0,2432 | 0,6111 |
| F1 N3 | 0,3869 | 0,7221 |
| Recall N1 | 0,1568 | 0,1057 |
| F1 N2 | 0,7544 | 0,7132 |
| F1 REM | 0,6231 | 0,5744 |
| Accuracy | 0,6898 | 0,6873 |

ADAST tăng macro-F1 trung bình **0,0445** (4,45 điểm phần trăm; CI 95%
[0,0352; 0,0539]). Ở ngân sách nhỏ, hiệu ứng là −0,0452. Chênh lệch giữa hai
hiệu ứng là **+0,0897**, CI [0,0767; 0,1023], tính bằng cùng 10.000 lần lấy
mẫu lại theo người cho cả bốn hệ thống. Đây là bằng chứng hiệu ứng ADAST thay
đổi theo ngân sách thực nghiệm, không phải chỉ theo kiến trúc.

N3 bị nhầm thành N2 giảm từ 75,26% xuống 36,89%. Tuy nhiên, F1 N1/N2/REM giảm;
REM recall tăng nhưng precision giảm nên F1 vẫn thấp hơn. Trên outer test
Sleep-EDF, macro-F1 gộp giảm 0,7211 → 0,6831. Kết quả giải quyết câu hỏi liệu
ADAST vẫn giảm điểm tổng thể khi được học đủ nguồn: **không, điểm tổng thể SHHS
tăng ở ngân sách lớn**, nhưng lợi ích không đồng đều giữa các lớp và quần thể.
Ngân sách lớn đồng thời tăng số update, mức tiếp xúc dữ liệu adaptation và
thay đổi thời điểm scheduler theo update; không quy toàn bộ hiệu ứng cho riêng
độ phủ nguồn. Các pilot và thử nghiệm thành phần loss được giữ riêng.

Phương pháp, kết quả chính, thảo luận và kết luận bản Anh/Việt đã được đồng bộ;
phụ lục bổ sung đủ precision/recall/F1 năm lớp và đối chiếu ngân sách bắt cặp.
Hồ sơ cập nhật: `ADAST_FULLSOURCE_PAPER_UPDATE_20261007_VI.md`.

## 1. Nghịch lý preprocessing E3–E4

**Đã sửa lập luận và bổ sung bằng chứng.** Bài hiện không gọi E3 là quy trình
tiền xử lý tốt nhất cho transfer. E3 giữ vai trò tham chiếu của thiết kế chính;
E4 đạt subject-mean macro-F1 cao hơn trong cả hai đối chiếu cùng seed trên SHHS:

- Seed 42: E4 0,5811, E3 0,5680; chênh +0,0131, CI 95% [0,0111; 0,0153].
- Seed 123: E4 0,5732, E3 0,5634; chênh +0,0098, CI [0,0073; 0,0125].

Phần seed 42 mới được chạy từ đủ mười checkpoint E4 lịch sử đã khôi phục và
kiểm chứng. Kết quả đã cập nhật vào tóm tắt, phương pháp, kết quả, thảo luận,
kết luận của bản Anh/Việt; Bảng S11 đặt hai seed cạnh nhau. Phân tích chính và
họ Holm lịch sử được giữ nguyên.

**Đã chỉnh diễn giải cơ chế.** Do clipping không kích hoạt, chênh lệch đầu vào
E3–E4 thực tế nằm ở thang biên độ. Phép nhân hằng số không loại bỏ riêng sóng
chậm; vì vậy đã bỏ hướng biện luận thiếu bằng chứng về “mất tín hiệu N3 do
chia 100”. So sánh các mô hình được huấn luyện riêng chưa tách được cơ chế
nhân quả tối ưu/BatchNorm. Bản hiện tại nêu đúng giới hạn này thay vì dựng lời
giải thích vật lý từ điểm số.

**Còn lại:** nếu cần một tuyên bố cơ chế hoặc chọn lại pipeline triển khai cuối,
cần thiết kế đo/can thiệp riêng và quy tắc lựa chọn trên dữ liệu phù hợp. Kết quả
hiện đủ để sửa khẳng định E3 tối ưu, chưa đủ để gọi E4 tốt hơn ở mọi quần thể.

## 2. Can thiệp cho thiếu hụt N3

**Đã thực hiện hai can thiệp.** Phần kết quả không còn chỉ dừng ở chẩn đoán lỗi
và đề xuất nghiên cứu tương lai.

1. **Calibration + prior correction, đủ mười fold:** temperature được khớp theo
   validation nguồn của từng fold; prior EM dùng năm người adaptation cố định
   không nhãn. Cả mười bộ tham số được đóng băng trước suy luận 180 người test.
   Temperature thay đổi subject-mean macro-F1 0,5590 → 0,5595; thêm EM làm giảm
   còn 0,4899 và không còn dự đoán N3. Kết quả âm được báo cáo ở mục 4.6 và Bảng S19.
2. **Cost-sensitive TCN, đủ mười fold:** đã kiểm chứng đủ 20 checkpoint nguồn,
   đóng băng trước khi chấm hai tổ hợp mười fold trên 180 người SHHS. Trong mỗi
   fold, hai nhánh có cùng encoder E3 đóng băng, khởi tạo và thứ tự minibatch;
   chỉ bộ phân loại chuỗi được huấn luyện lại. Fold 0–5 dùng CPU, fold 6–9 dùng
   CUDA; không mô tả chiến dịch là đồng nhất backend. N3 recall tăng 0,2345 →
   0,3225 (chênh +0,0880; CI [0,0805; 0,0954]), N3 F1 tăng 0,3759 → 0,4795.
   Tuy nhiên N2 recall giảm 0,7260 → 0,6728 và accuracy giảm 0,6892 → 0,6663.
   Subject-mean macro-F1 0,5530 → 0,5514; chênh −0,0016, CI [−0,0049; 0,0017].
   Bảng S20 trình bày tổ hợp mười fold; pilot một fold giữ riêng ở Bảng S21.

**Còn lại:** đã hoàn thành phần mở rộng weighted/unweighted đủ fold; chưa tìm
được giải pháp tăng N3 mà không làm giảm các giai đoạn khác. Không diễn giải CI
chứa không là bằng chứng tương đương. Bootstrap bắt cặp theo người điều kiện
trên các mô hình đã khớp, không bao gồm biến thiên huấn luyện hoặc backend.

## 3. Baseline domain adaptation ngoài

**Đã triển khai ADAST và đối chứng source-only tương ứng.** Mã upstream được
ghim revision; giữ model classes/similarity penalty, đồng nhất preprocessing với
dự án và bỏ nhãn target khỏi luồng huấn luyện. Hai nhánh có cùng khởi tạo, mẫu
nguồn và ngân sách 1.140 update. Phụ lục S2 nêu rõ năm người adaptation khiến
loader upstream giới hạn 38 batch mỗi epoch; đây không phải một lượt qua hết
tập nguồn.

**Pilot một fold trước đây:** subject-mean macro-F1 source-only/ADAST là 0,4755/0,4214; N3 recall
0,2657/0,2622. N1 recall ADAST chỉ 0,0077, và hiệu năng outer test nguồn cũng
giảm. Bảng S23 trình bày kết quả pilot âm cùng đối chứng phù hợp. Không dùng kết quả
này để tuyên bố E3 vượt ADAST, hoặc UDA nói chung không hiệu quả.

**Đã hoàn tất mở rộng đủ mười fold:** đã kiểm chứng độc lập 20 checkpoint mới
trên CUDA, khóa trước suy luận cục bộ trên đủ 180 người SHHS. Mỗi nhánh có cùng
ngân sách 1.140 update/fold; không nhập checkpoint CPU từ pilot. Bảng S22 trình
bày kết quả tổ hợp mười fold. N3 recall tăng 0,3195 → 0,4269 (chênh +0,1074;
CI 95% [0,0714; 0,1422]), N3 F1 tăng 0,4752 → 0,5837 và tỷ lệ N3→N2 giảm
0,6747 → 0,5588. Tuy nhiên, ADAST không dự đoán N1 nào trên SHHS (recall của
đối chứng là 0,1974), REM recall giảm 0,4343 → 0,2938. Subject-mean macro-F1
giảm 0,4949 → 0,4497 (chênh −0,0452; CI [−0,0558; −0,0347]); macro-F1 gộp
giảm 0,5539 → 0,5072. Không mô tả đây là kết quả hoàn toàn âm về N3, cũng không
gọi là cải thiện tổng thể.

Đánh giá ngoài fold trên 78 người Sleep-EDF, dùng source attention cho cả hai
nhánh, cho macro-F1 gộp 0,6665 → 0,5276; vì vậy suy giảm không chỉ xuất hiện ở
miền đích. Hai bộ kiểm chứng cùng khớp hash aggregate; đã tính lại mọi ma trận
nhầm lẫn/bootstrap và chạy lại 40 cặp mô hình–bản ghi đích, 20 cặp mô hình–batch
nguồn. Xem `COLAB_ADAST_RESULTS_20261004_VI.md` cho thời gian và giao thức.

**Đã bổ sung chẩn đoán ngân sách trên một fold, một seed:** bốn mô hình mới
(source-only/ADAST × ngân sách nhỏ/toàn bộ train nguồn) đã chạy đủ 30 epoch trên
T4 và đạt kiểm chứng độc lập. Hai mức ngân sách lần lượt là 1.140 và 36.870 update
mỗi mô hình. Chọn checkpoint bằng macro-F1 validation nguồn theo quy tắc khóa
trước; ghi đủ chỉ số từng lớp qua mỗi epoch. Đây là validation nguồn gộp epoch,
không phải kết quả SHHS mới hoặc thay thế tổ hợp mười fold nêu trên.

Ở checkpoint tốt nhất, ngân sách lớn tăng macro-F1 source-only từ 0,6585 lên
0,7174 và ADAST từ 0,6470 lên 0,6934. ADAST ngân sách nhỏ cuối lượt không còn
dự đoán N1; ADAST toàn nguồn giữ recall N1 0,1591, nhưng vẫn thấp hơn đối chứng
0,2847. Ở checkpoint cuối của cặp lớn, recall N3 tăng 0,5783 → 0,8068 trong khi
macro-F1 giảm 0,6911 → 0,6784. Vì vậy, tăng ngân sách khắc phục tình trạng mất
hoàn toàn N1 trong lượt phát triển này, chưa khắc phục toàn bộ đánh đổi.

Suy giảm N1 của ADAST nhỏ bắt đầu rõ ở epoch 16, khi hệ số học nguồn giảm và
loss nhãn giả được bật đồng thời. Chưa tách được tác động nhân quả của từng
thành phần. Xem `ADAST_DEVELOPMENT_BUDGET_20261004_VI.md` cho bảng best/final,
giao thức và hồ sơ kiểm chứng. Số liệu phát triển này đã được bổ sung vào bản thảo.

**Đã train xong can thiệp tách thành phần loss:** bốn mô hình mới gồm ADAST
đối chứng và ba can thiệp riêng (giữ source CE, tắt pseudo CE, tắt đối kháng)
đã chạy đủ 30 epoch trên Tesla T4, mỗi mô hình 36.870 update; mất 4.139,75 giây
không gồm upload. Fold 0, seed 123, cùng dữ liệu và quy tắc chọn checkpoint;
không đưa outer test nguồn hoặc 180 người SHHS đánh giá vào lượt phát triển.
Gói kết quả tải về khớp SHA trên Colab; **kiểm chứng trong gói và độc lập đều
đạt**, cùng khớp hash aggregate. Đã tính lại 124 bộ chẩn đoán và suy luận CPU
trên toàn bộ validation từ tám checkpoint best/final qua cả hai đường attention.

Kết quả ở checkpoint được chọn theo macro-F1 validation nguồn:

| Cấu hình | Epoch | Macro-F1 | Recall N1 | Recall N3 | Recall REM |
|---|---:|---:|---:|---:|---:|
| ADAST đối chứng | 22 | 0,6934 | 0,2338 | 0,8261 | 0,7139 |
| Giữ source CE | 22 | 0,6941 | 0,2664 | 0,8323 | 0,6490 |
| Tắt pseudo CE | 7 | 0,6891 | 0,4902 | 0,8834 | 0,5486 |
| Tắt đối kháng | 29 | 0,6943 | 0,2416 | 0,7115 | 0,6865 |

Giữ source CE tăng khả năng nhận diện N1 nhưng giảm REM; N3 recall tăng nhẹ
trong khi N3 precision/F1 giảm. Nhánh tắt đối kháng đứng đầu macro-F1 theo
tiêu chí chính đã khóa, nhưng recall N3 giảm 11,46 điểm phần trăm ở checkpoint
được chọn và 20,43 điểm phần trăm tại epoch 30. Hai mức tăng macro-F1 được
chọn so với ADAST đối chứng đều dưới 0,001. Không gọi giữ source CE là cấu
hình thắng chỉ vì N1/N3 recall đẹp hơn.

Nhánh tắt nhãn giả chọn epoch 7, trước khi bản gốc bật pseudo CE ở epoch 16.
Recall N1 0,4902 tại đó cũng xuất hiện ở đối chứng tại cùng epoch, không phải
bằng chứng tắt pseudo CE đã sửa N1. Ở cùng epoch 30, tắt nhãn giả tăng nhẹ
recall N1/N3 nhưng giảm macro-F1 và REM. Đối chứng toàn nguồn không sụp N1
tại epoch 16; không chuyển diễn biến của ngân sách nhỏ thành cơ chế chung.

Source-only toàn nguồn trong lượt trước đạt macro-F1 được chọn 0,7174, vẫn
cao hơn các cấu hình ADAST phát triển mới. Các bảng đầy đủ gồm năm lớp,
checkpoint được chọn/cuối và diễn biến 15→16→30 nằm trong
`ADAST_LOSS_ABLATION_20261004_VI.md`.

Đường target attention được kiểm tra trên cùng validation nguồn, không dùng
để chọn lại checkpoint. Tắt đối kháng làm macro-F1 của đường này giảm
0,6780 → 0,5227 và recall N3 giảm 0,8302 → 0,0773 ở checkpoint được chọn.
Vì vậy, xếp hạng tốt hơn qua source attention không xác nhận khả năng thích
nghi của đường target attention; chưa chọn nhánh này cho chuyển quần thể.

**Đã hoàn tất lượt xác nhận trên fold nguồn khác:** ba mô hình mới trên fold 1,
seed 123 (source-only, ADAST đối chứng, giữ source CE) học đủ nguồn trong 30
epoch, 35.820 update/nhánh. Thời gian T4 là 2.975,78 giây, không gồm upload.
ZIP khớp SHA Colab; kiểm chứng trong gói và độc lập đạt cùng aggregate hash,
tính lại 93 bộ chẩn đoán và suy luận CPU toàn validation từ sáu checkpoint
best/final. Mọi chọn mô hình dùng validation nguồn; không truy cập test nguồn
hoặc 180 SHHS test. Xem `ADAST_SMALL_CONFIRMATION_20261004_VI.md`.

Macro-F1 được chọn của source-only/ADAST đối chứng/giữ source CE lần lượt
0,7370/0,6907/0,6892. Giữ source CE tăng F1 N1 0,2175 → 0,2302 và F1 N3
0,8096 → 0,8161 qua source attention, nhưng giảm F1 REM 0,6852 → 0,6680.
Qua target attention trên cùng validation nguồn, F1 N3 giảm 0,8003 → 0,7864;
không gọi chỉ số này là hiệu năng SHHS. Tại epoch 30, bản sửa có macro-F1 và
F1 N1 thấp hơn ADAST đối chứng. Hướng cải thiện N1 nhỏ tại checkpoint được
chọn và đánh đổi REM lặp lại ở hai fold cùng seed; không phải cải thiện đều
mọi checkpoint/đường attention. Bản sửa không đạt quy tắc phát triển đã khóa,
do điểm tổng hợp và các đánh đổi nêu trên; chưa chọn cấu hình này để mở rộng.

**Kết luận sau đối chiếu đủ nguồn:** baseline ADAST nay có đối chứng mười fold
ở cả ngân sách nhỏ và lớn, cùng epoch cuối và cùng 180 người SHHS. Điều này
hoàn thành kiểm tra ngân sách và bổ sung hệ quy chiếu ngoài cho góp ý của thầy.
Can thiệp giữ source CE không được nhập vào tổ hợp mới, vì đánh đổi trong
phát triển không đáp ứng tiêu chí lựa chọn. Không cần chạy thêm cấu hình chỉ
để tăng số thí nghiệm. Nếu nghiên cứu tiếp về ổn định qua khởi tạo hoặc một
biện pháp giữ được mọi lớp, đó là câu hỏi mới cần thiết kế riêng. Các kết quả
hiện tại định lượng rõ lợi ích N3, điểm tổng thể và tổn thất N1/N2/REM; không
gọi ADAST là giải pháp cải thiện tất cả các lớp hay một benchmark UDA toàn diện.
Benchmark vận hành lịch sử và mọi kết quả ngân sách nhỏ được giữ nguyên.

## 4. Tác giả và bản thảo

Đã thêm Tri Nguyen Ho Duy (`trinhd@uit.edu.vn`) và Tri Nguyen (`tringuyen@uit.edu.vn`)
sau hai tác giả hiện tại, cùng khoa Hệ thống Thông tin, UIT–ĐHQG TP.HCM; giữ tác giả
liên hệ hiện tại. Vai trò CRediT thực tế của Tri Nguyen còn cần xác nhận.

Các mục bổ sung dùng tên **Thí nghiệm bổ sung**. Phạm vi một fold/mười fold,
quần thể đã được xem và giao thức riêng được mô tả trong phương pháp; không
thay lịch sử nghiên cứu bằng lời khẳng định “đánh giá lần đầu”. Các câu nói chưa
chạy can thiệp/UDA đã được cập nhật. Tuyên bố hỗ trợ AI cũng được sửa để phản ánh
việc hỗ trợ triển khai và chạy mã tính toán, đồng thời giữ trách nhiệm kiểm chứng
của tác giả.

Bản ngày 04-10 gồm ngân sách, bốn can thiệp loss và lượt xác nhận fold 1
(các bảng S24–S29). Lượt loss fold 0 được kiểm chứng đủ 124 bộ chẩn đoán và
tám checkpoint; lượt fold 1 đủ 93 bộ chẩn đoán và sáu checkpoint. Bản ngày
07-10 bổ sung kết quả chuyển quần thể toàn nguồn; hồ sơ PDF/source và kiểm tra
bố cục hiện hành nằm trong báo cáo cập nhật mới nhất.
Đây là bản
làm việc có thực nghiệm bổ sung, không phải xác nhận đã xử lý trọn mọi yêu cầu
benchmark hoặc được bốn tác giả duyệt nộp.
