# Tách thành phần ADAST — 04/10/2026

## Trạng thái

Đã hoàn thiện mã và khóa gói. Notebook cũ bị chặn hạn mức GPU; giữ hồ sơ đó trong `colab_launch_record.json`. Người dùng chuyển tài khoản, cung cấp notebook mới và xác nhận trực tiếp cho phép tải lại đúng EEG nguồn fold 0 và năm người adaptation đã bỏ ID đến phiên mới.

Ngày 04/10 lúc khoảng 18:14 (+07), đã gửi tám phần đầu vào và thực thi ô 14 tại notebook mới `19wRkGwBOlqm4voA-z-6Al52gzIaRDW68`. Upload và toàn bộ hash đầu vào/mã đã đạt kiểm tra; **cả bốn mô hình đã train đủ 30 epoch trên Tesla T4, PyTorch 2.11.0+cu130**, kết thúc khoảng 20:03 (+07). Train cùng validation mất 4.139,75 giây (1 giờ 8 phút 59,75 giây), không tính upload. Không chạm giới hạn năm giờ, không train CPU thay thế.

ZIP nằm trong Downloads của máy, không phải ổ D gốc. Hash bản tải về khớp giá trị quan sát trên Colab. **Kiểm chứng trong gói và kiểm chứng độc lập đều đạt**, cùng khớp hash aggregate. Kiểm chứng độc lập trên CPU mất 1.249,94 giây (20 phút 49,94 giây), chạy lại toàn bộ validation từ tám checkpoint best/final qua hai đường attention; tính lại 124 bộ chẩn đoán và kiểm tra sampler/độ phủ, ngân sách, loss, khởi tạo, discriminator đã tắt và phần lịch sử chung trước epoch 16. Đây là suy luận, không train lại. Hồ sơ phiên mới giữ riêng với hồ sơ hạn mức GPU cũ.

Đã đạt 18 kiểm thử cho lượt mới và hồi quy: đối chứng khớp phép cập nhật cũ, mỗi can thiệp chỉ thay một yếu tố, nhánh tắt đối kháng không forward/cập nhật discriminator, loss tái dựng được, checkpoint best/final và RNG được lưu, kiểm chứng sampler/chỉ số và khởi tạo độc lập. Test dùng dữ liệu giả nhỏ trên CPU, không phải một lượt train nghiên cứu.

## Bốn mô hình đã chạy

| Mô hình | Thay đổi duy nhất so với ADAST gốc |
|---|---|
| ADAST đối chứng mới | Không thay đổi |
| Giữ học nguồn | Giữ hệ số source CE = 1 ở vòng hai, thay vì giảm xuống 0,1 |
| Không học nhãn giả | Đặt hệ số pseudo CE = 0 ở cả hai vòng; giữ forward và lịch sinh nhãn giả để đối chiếu |
| Không đối kháng | Loại forward/backward/cập nhật discriminator và loss đối kháng; giữ target forward, nhãn giả và các loss khác |

Tất cả dùng fold 0, seed 123, cùng dữ liệu, kiến trúc, preprocessing, khởi tạo, thứ tự mẫu và lịch learning rate. Mỗi mô hình học toàn bộ 157.200 mẫu nguồn mỗi epoch: 1.229 update/epoch × 30 epoch = 36.870 update. Chọn checkpoint theo macro-F1 validation nguồn, giữ epoch đầu nếu bằng điểm; mô hình cuối epoch 30 được báo riêng. Ghi đủ năm lớp và hai attention paths mỗi epoch, source attention là đường chọn checkpoint.

Đây là bước phát triển cấu hình trên validation nguồn; dùng lại đúng năm người adaptation không nhãn đã cho phép, không dùng dữ liệu outer test nguồn hoặc 180 người SHHS test.

## Quy tắc đọc kết quả

1. Trình bày bảng checkpoint được chọn: macro-F1, N1/N3 precision–recall–F1 và các giai đoạn còn lại.
2. Trình bày checkpoint cuối và diễn biến epoch 15→16→30 để xác định sự suy giảm trong quá trình học.
3. Đối chiếu từng can thiệp với ADAST đối chứng chạy mới; dùng source-only toàn nguồn đã kiểm chứng trước đây làm mốc bổ sung, không thay thế đối chứng mới.
4. Đọc lợi ích và đánh đổi từ số liệu, không mặc định mọi can thiệp sẽ cải thiện. Chỉ quyết định mở rộng sau khi kiểm chứng kết quả và đóng băng cấu hình được chọn.

## Gói và giới hạn chạy

- Gói mã nhỏ: `SleepTCN_ADAST_Loss_Ablation_Patch_20261004.zip`, 25.297 byte; SHA-256 `c03a44428a08b075c2ae37f7d59e9e715d87fc394a91d190461b165aedffac22`.
- Manifest mới: `5a18f41a60adf29767b1530f93b3c61207dc5f13eba621852643a13dd691ea91`.
- Dữ liệu tái sử dụng khớp từng tệp với manifest phát triển cũ `cbdf40d5800f238bf0ea5fbb2ac87bd11de7bfcc7eaab05c9e4c3cbd9ffc6b06`.
- Local dùng hardlink chỉ đọc cho EEG để không nhân đôi dữ liệu trên ổ D gần đầy. Không sửa các tệp này.
- Ô 14 (`cell-WIjid1PYjr3y`) chứa launcher nhúng gói mã; nội dung 40.457 ký tự đã được đọc lại qua clipboard của giao diện notebook mới và khớp toàn bộ với tệp local; thực thi số 1 đã hoàn tất.
- Khi GPU sẵn sàng, launcher kiểm tra hash trước train. Nếu VM còn dữ liệu, tái sử dụng ngay; nếu VM đã mất dữ liệu, cần tải lại đúng tám phần EEG cũ vào cùng phiên trước khi chạy.
- Giới hạn chung cho cả bốn mô hình: 18.000 giây. Giữ checkpoint sau từng epoch; dừng khi hết thời gian, không tự restart, không fallback train CPU, không tự mở rộng seed/mười fold.
- ZIP đầu ra: `SleepTCN_ADAST_Loss_Ablation_Results_20261004.zip`. Đối chiếu SHA hiển thị trên Colab rồi dùng bộ kiểm chứng độc lập để tính lại đủ lịch sử và suy luận CPU từ tám checkpoint best/final.

Lệnh kiểm chứng của lượt này (đã hoàn tất, không ghi đè thư mục kiểm chứng):

```powershell
.venv/Scripts/python.exe scripts/verify_adast_development_results.py --input-base runs/adast_loss_ablation_20261004 --archive C:/Users/ADMIN/Downloads/SleepTCN_ADAST_Loss_Ablation_Results_20261004.zip --expected-sha256 3d9c12751d412e74b79bacff74b9abc207e06bd0f74f5cae902ebb421151b034 --output runs/adast_loss_ablation_20261004/verified_results --max-seconds 1800
```

## Kết quả: mô hình được chọn và mô hình cuối lượt

Các số bên dưới là chỉ số gộp trên 18.763 epoch tín hiệu validation Sleep-EDF của fold 0, seed 123, qua source attention. Số mẫu W/N1/N2/N3/REM lần lượt là 5.336/2.301/7.010/1.449/2.667. Đường target attention cũng được kiểm tra trên chính validation nguồn, không phải đánh giá SHHS. Không thay các tổ hợp mười fold lịch sử bằng kết quả phát triển này.

### Checkpoint được chọn theo macro-F1 validation nguồn

| Cấu hình | Epoch chọn | Macro-F1 | Accuracy | Recall N1 | Recall N3 | Recall REM |
|---|---:|---:|---:|---:|---:|---:|
| ADAST đối chứng | 22 | 0,6934 | 0,7627 | 0,2338 | 0,8261 | 0,7139 |
| Giữ học nguồn | 22 | 0,6941 | 0,7605 | 0,2664 | 0,8323 | 0,6490 |
| Tắt nhãn giả | 7 | 0,6891 | 0,7266 | 0,4902 | 0,8834 | 0,5486 |
| Tắt đối kháng | 29 | 0,6943 | 0,7677 | 0,2416 | 0,7115 | 0,6865 |

Theo tiêu chí chính đã khóa, thứ tự macro-F1 là: tắt đối kháng → giữ học nguồn → đối chứng → tắt nhãn giả. Mức tăng so với đối chứng của hai cấu hình đầu chỉ là +0,000887 và +0,000673. Không đổi tiêu chí chính thành recall N1 hoặc N3 sau khi thấy bảng này.

### Checkpoint cuối cố định, epoch 30

| Cấu hình | Macro-F1 | Accuracy | Recall N1 | Recall N3 | Recall REM |
|---|---:|---:|---:|---:|---:|
| ADAST đối chứng | 0,6784 | 0,7584 | 0,1591 | 0,8068 | 0,7544 |
| Giữ học nguồn | 0,6809 | 0,7538 | 0,2064 | 0,8095 | 0,6742 |
| Tắt nhãn giả | 0,6717 | 0,7493 | 0,1769 | 0,8192 | 0,6430 |
| Tắt đối kháng | 0,6777 | 0,7565 | 0,2425 | 0,6025 | 0,6764 |

### F1 đủ năm lớp

| Cấu hình | Checkpoint | W | N1 | N2 | N3 | REM |
|---|---|---:|---:|---:|---:|---:|
| Đối chứng | Được chọn | 0,8771 | 0,2988 | 0,8151 | 0,7880 | 0,6880 |
| Giữ học nguồn | Được chọn | 0,8831 | 0,3238 | 0,8100 | 0,7834 | 0,6700 |
| Tắt nhãn giả | Được chọn | 0,8818 | 0,4055 | 0,7692 | 0,7776 | 0,6116 |
| Tắt đối kháng | Được chọn | 0,8804 | 0,3111 | 0,8205 | 0,7660 | 0,6934 |
| Đối chứng | Epoch 30 | 0,8812 | 0,2295 | 0,8073 | 0,7944 | 0,6796 |
| Giữ học nguồn | Epoch 30 | 0,8835 | 0,2722 | 0,8014 | 0,7880 | 0,6596 |
| Tắt nhãn giả | Epoch 30 | 0,8797 | 0,2465 | 0,7925 | 0,7848 | 0,6548 |
| Tắt đối kháng | Epoch 30 | 0,8772 | 0,3074 | 0,8108 | 0,7173 | 0,6757 |

## Can thiệp thực sự thay đổi điều gì?

### 1. Giữ học nguồn: N1 tốt hơn, đổi lại REM giảm

Hai checkpoint được chọn đều ở epoch 22, cho phép đối chiếu cùng thời điểm. Recall N1 tăng 0,2338 → 0,2664 (+3,26 điểm phần trăm), F1 N1 tăng 0,2988 → 0,3238. Precision N1 gần như giữ nguyên (0,4138 → 0,4128). Ở epoch 30, recall N1 tăng 0,1591 → 0,2064 (+4,74 điểm phần trăm).

Recall N3 tăng nhẹ, nhưng precision N3 giảm 0,7533 → 0,7399 và F1 N3 giảm 0,7880 → 0,7834 ở checkpoint được chọn. Recall REM giảm 0,7139 → 0,6490; F1 REM giảm 0,6880 → 0,6700, accuracy giảm 0,7627 → 0,7605. Vì vậy, kết quả hỗ trợ giữ mức học nguồn như một can thiệp có ích cho N1 trong lượt này, không phải cải thiện đồng đều mọi lớp hoặc đã giải quyết thiếu hụt N3.

### 2. Tắt nhãn giả: phải phân biệt chọn mô hình sớm với tác động của can thiệp

Trong 15 epoch đầu, đối chứng, giữ học nguồn và tắt nhãn giả có cùng phép cập nhật vì hệ số học nhãn giả của bản gốc cũng bằng 0. Nhánh tắt nhãn giả chọn epoch 7: recall N1 0,4902 ở đó cũng tồn tại trong đối chứng tại cùng epoch. Không quy chỉ số này cho lợi ích của tắt nhãn giả, cũng không dùng nó để đổi checkpoint của đối chứng vốn được chọn theo macro-F1.

Sau khi hai cấu hình thực sự khác nhau, tại epoch 30, tắt nhãn giả tăng recall N1 0,1591 → 0,1769 và recall N3 0,8068 → 0,8192, nhưng giảm macro-F1 0,6784 → 0,6717, F1 N3 0,7944 → 0,7848 và recall REM 0,7544 → 0,6430. Tắt nhãn giả một mình không sửa được đánh đổi tổng thể.

### 3. Tắt đối kháng: điểm tổng thể cao nhất, nhưng khả năng bắt N3 giảm

Ở checkpoint được chọn, macro-F1 tăng 0,6934 → 0,6943, N1 F1 tăng 0,2988 → 0,3111; recall N3 giảm 0,8261 → 0,7115 (−11,46 điểm phần trăm), N3 F1 giảm 0,7880 → 0,7660. Precision N3 tăng 0,7533 → 0,8294: mô hình dự đoán N3 thận trọng hơn, đồng thời bỏ sót nhiều hơn.

Ở cùng epoch 30, recall N1 tăng 0,1591 → 0,2425 nhưng recall N3 giảm 0,8068 → 0,6025; macro-F1 giảm nhẹ 0,6784 → 0,6777. Phải báo kết quả theo đúng từng quy tắc checkpoint. Việc bỏ thành phần đối kháng còn làm thay đổi đường thực thi và số phép tính; không diễn giải số liệu như phép đo riêng độ lớn gradient đối kháng.

## Diễn biến tại mốc chuyển vòng học

Mỗi ô dưới đây là recall N1 / recall N3; số liệu đều qua source attention.

| Cấu hình | Epoch 15 | Epoch 16 | Epoch 30 |
|---|---:|---:|---:|
| Đối chứng | 0,1660 / 0,8461 | 0,1934 / 0,7226 | 0,1591 / 0,8068 |
| Giữ học nguồn | 0,1660 / 0,8461 | 0,2351 / 0,7923 | 0,2064 / 0,8095 |
| Tắt nhãn giả | 0,1660 / 0,8461 | 0,2060 / 0,7585 | 0,1769 / 0,8192 |
| Tắt đối kháng | 0,2412 / 0,7640 | 0,2277 / 0,6349 | 0,2425 / 0,6025 |

Trong đối chứng toàn nguồn, N1 không sụp tại epoch 16: recall tăng 0,1660 → 0,1934. Không gán diễn biến sụp N1 ở ngân sách nhỏ cho mọi mức ngân sách. Recall N1 trung bình mô tả qua 15 epoch vòng hai lần lượt là 0,1521 / 0,1856 / 0,1709 / 0,2261; các epoch phụ thuộc nhau, không dùng chúng như 15 lần chạy độc lập để tính kiểm định hoặc khoảng tin cậy huấn luyện.

## Kiểm tra riêng đường target attention

Bảng này vẫn đánh giá trên validation Sleep-EDF, chỉ thay đường attention của cùng checkpoint. Đây không phải đánh giá SHHS; đường target attention không được dùng để chọn lại checkpoint.

| Cấu hình | Checkpoint | Macro-F1 | Recall N1 | Recall N3 |
|---|---|---:|---:|---:|
| Đối chứng | Được chọn | 0,6780 | 0,2225 | 0,8302 |
| Giữ học nguồn | Được chọn | 0,6789 | 0,2599 | 0,8599 |
| Tắt nhãn giả | Được chọn | 0,6803 | 0,4259 | 0,8551 |
| Tắt đối kháng | Được chọn | 0,5227 | 0,1521 | 0,0773 |
| Đối chứng | Epoch 30 | 0,6674 | 0,1586 | 0,8116 |
| Giữ học nguồn | Epoch 30 | 0,6628 | 0,1899 | 0,8682 |
| Tắt nhãn giả | Epoch 30 | 0,6546 | 0,1517 | 0,8116 |
| Tắt đối kháng | Epoch 30 | 0,5026 | 0,1352 | 0,0628 |

Nhánh tắt đối kháng có điểm source-attention được chọn cao nhất nhưng đường target attention yếu rõ, đặc biệt N3. Trong mã của nhánh này, source CE đi qua source attention, similarity chỉ phạt hai classifier heads; target pseudo CE có hệ số 0 ở vòng đầu, và loss đối kháng bị bỏ. Vì vậy, đường target attention ở vòng đầu không còn hai đóng góp loss huấn luyện đặc thù đó, dù target forward và optimizer/weight decay được giữ. Đây là hệ quả của can thiệp đã định, không tự sửa mã hay đổi đường suy luận sau khi xem điểm.

Phát hiện này cần được xem cùng bảng chính trước khi cân nhắc dùng nhánh tắt đối kháng cho chuyển quần thể. Điểm source-attention cao không đủ xác nhận đường target attention đã học biểu diễn thích nghi tốt. Giữ học nguồn còn tăng recall N3 qua target attention, nhưng macro-F1 cuối lượt vẫn giảm 0,6674 → 0,6628; không mô tả là cải thiện toàn diện cả hai đường.

## Đối chiếu nguồn-only và kết luận cho bước tiếp theo

Source-only toàn nguồn đã kiểm chứng trong lượt trước đạt macro-F1 được chọn 0,7174 (epoch 9), cao hơn mọi cấu hình ADAST vừa chạy. Recall N1 của source-only là 0,5380; recall N3 là 0,8806, F1 N3 là 0,7684. ADAST đối chứng có F1 N3 cao hơn source-only (0,7880), nhưng recall N3 thấp hơn. Dùng nguồn-only làm mốc bổ sung phù hợp ngân sách, không thay thế đối chứng ADAST mới của ablation.

Kết quả hiện trả lời được ba can thiệp riêng và chỉ ra đánh đổi từng lớp. Chưa có cấu hình nào đồng thời hơn đối chứng nguồn-only về macro-F1, nhận diện N1 và mọi chỉ số N3. Không tuyên bố một loss là nguyên nhân duy nhất của suy giảm hoặc một can thiệp đã sửa hoàn toàn chuyển quần thể.

Bước tiếp theo cần phê duyệt riêng là kiểm tra độ lặp lại và đường target attention trên seed/fold khác với giao thức khóa trước. Theo xếp hạng chính, tắt đối kháng đứng đầu nhưng có cảnh báo mạnh ở đường target attention; giữ học nguồn là ứng viên bổ sung có cân bằng recall N1/N3 khác. Không tự chọn giữ học nguồn làm cấu hình thắng, không tự ghép các can thiệp thành một cấu hình mới, không tự mở mười fold hoặc dùng nhãn SHHS để chọn cấu hình. Bản Anh/Việt và phụ lục được bổ sung kết quả phát triển; các kết quả SHHS lịch sử không thay đổi.

## Hồ sơ kết quả

- ZIP tải về: SHA-256 `3d9c12751d412e74b79bacff74b9abc207e06bd0f74f5cae902ebb421151b034`.
- Aggregate kết quả: SHA-256 `e54577df6235ec381aa3996d8b59f9efae008bae351131110a6cbbb07e10122b`; kiểm chứng trong gói khớp giá trị này.
- Script phân tích không train: `scripts/analyze_adast_loss_ablation_results.py`; sáu kiểm thử đạt, kiểm tra tách selected/final, xếp hạng, mốc kích hoạt, chặn đối chiếu lịch sử sai và ràng buộc cổng kiểm chứng với hash aggregate.
- Phân tích đã xuất tại `Reports/analysis/adast_loss_ablation_20261004/` sau khi kiểm chứng độc lập đạt. `summary.json` giữ cả hai attention paths, toàn bộ diễn biến và chênh lệch; `all_class_metrics.csv` chứa precision/recall/F1 đủ năm lớp qua source attention, không chứa ID hoặc dự đoán cá thể. `manifest.json` ràng buộc hash của hai tệp phân tích.
- Bản Anh và bản dịch Việt đều 16 trang, phụ lục 12 trang, đã dựng lại và rà trực quan toàn bộ trang; các bảng phát triển S24–S27 không tràn lề, tham chiếu Anh/Việt khớp nhau và các bảng kết quả lịch sử được giữ. Hồ sơ QA và đóng gói của lượt sửa này ở `tmp/adast_loss_revision_20261004/`; gói PDF/source được phát hành chỉ sau khi các cổng kiểm chứng và hash đều đạt.
