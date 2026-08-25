# Bản thảo nghiên cứu — Gate 8

> Bản thảo được biên soạn từ các artifact đã khóa. Nội dung trình bày kết quả theo chuẩn báo cáo khoa học; các bảng và hình đi kèm là nguồn số liệu định lượng.

## Tiêu đề đề xuất

**Đánh giá bắt cặp các kiến trúc học sâu cho phân giai đoạn giấc ngủ một kênh trên Sleep-EDF Expanded**

Tiêu đề thay thế:

**Đánh giá hiệu năng, độ bền ngoài miền và độ phức tạp của ResNet-1D–TCN trong phân giai đoạn giấc ngủ**

## Tóm tắt

Nghiên cứu đánh giá một cách có kiểm soát các mô hình học sâu cho phân giai đoạn giấc ngủ từ EEG một kênh. Thí nghiệm được thực hiện trên phân tập Sleep Cassette của Sleep-EDF Expanded với 78 đối tượng và 153 bản ghi. Sáu cấu hình được so sánh trên cùng phép chia theo đối tượng, cùng quy tắc chọn mô hình và cùng tập kiểm tra ngoài fold; chỉ số chính là Macro-F1.

Trong chiến dịch chính, cấu hình kết hợp ResNet-1D, mạng tích chập theo thời gian và chế độ xử lý biên độ đã đạt Macro-F1 bằng 0,7904, cao nhất trong sáu cấu hình. So với chế độ z-score theo bản ghi, cấu hình này tăng 0,0213 điểm, với khoảng tin cậy 95\% từ 0,0122 đến 0,0307 và $p$ sau hiệu chỉnh Holm bằng 0,0012. Một lần lặp sau giao thức vẫn cho hiệu ứng cùng chiều. Hệ thống ResNet-1D–TCN có thời gian suy luận nhanh hơn khoảng 3,76 lần, đổi lại số tham số cao hơn 4,37 lần và bộ nhớ GPU cực đại cao hơn 28,4\%.

Trong chiến dịch seed 123, E2 hoàn tất 10 fold trong 2 giờ 44 phút 57 giây, so với 33 giờ 35 phút 36 giây của E0; toàn bộ sáu cấu hình cần 45 giờ 12 phút 10 giây khi chạy tuần tự. Đây là thời gian huấn luyện, validation và hoàn thiện artifact trên GPU V100, được báo cáo riêng với độ trễ suy luận.

Đánh giá zero-shot trên 180 đối tượng SHHS1 đã khóa tiếp tục cho thấy cấu hình này cao hơn các mốc đối chứng. Kết quả hỗ trợ giá trị của toàn bộ quy trình trong các giao thức đã thực hiện, đồng thời cho thấy lợi ích quan sát được không nên quy trực tiếp cho một thành phần riêng lẻ. Phân tích ablation nhóm ngữ cảnh không tìm thấy lợi ích tăng thêm có ý nghĩa tại vùng chuyển pha. Kết luận hiện tại vì vậy tập trung vào hiệu quả của quy trình đã đánh giá, không mở rộng thành tuyên bố về tính tối ưu phổ quát, tương đương mô hình hoặc giá trị lâm sàng.

**Từ khóa:** phân giai đoạn giấc ngủ; EEG một kênh; ResNet-1D; mạng tích chập theo thời gian; Sleep-EDF Expanded; SHHS1; đánh giá bắt cặp.

## 1. Đặt vấn đề

Phân giai đoạn giấc ngủ tự động có ý nghĩa đối với phân tích đa ký giấc ngủ ở quy mô lớn. Tuy nhiên, sự khác biệt giữa các mô hình có thể bị chi phối bởi cách chia đối tượng, quy tắc lựa chọn mô hình và chế độ tiền xử lý. Một đánh giá đáng tin cậy cần bảo đảm rằng các cấu hình được so sánh trên cùng đối tượng, cùng dữ liệu đầu vào và cùng tiêu chí đánh giá.

Nghiên cứu này tập trung vào ba vấn đề: hiệu quả của mô hình chuỗi so với mốc BiLSTM, giá trị của bộ trích đặc trưng ResNet-1D trong cùng quy trình chuỗi, và ảnh hưởng của cách xử lý biên độ. Bên cạnh hiệu năng trong miền, nghiên cứu đánh giá khả năng vận hành và độ bền ngoài miền trên SHHS1.

Đóng góp chính gồm: thiết kế bắt cặp theo đối tượng; quy trình lựa chọn và khóa tập kiểm tra rõ ràng; đánh giá đồng thời hiệu năng, độ phức tạp và khả năng ngoài miền; cùng phân tích ablation có kiểm soát đối với ngữ cảnh thời gian. Các kết luận được giới hạn theo đúng giao thức và mẫu đã đánh giá.

## 2. Phương pháp

### 2.1. Dữ liệu và thiết kế đánh giá

Nghiên cứu sử dụng Sleep-EDF Expanded, phân tập Sleep Cassette, gồm 78 đối tượng và 153 bản ghi. Tín hiệu EEG một kênh được phân thành các epoch 30 giây và gán vào năm lớp W, N1, N2, N3 và REM. Các epoch không có nhãn hợp lệ được giữ lại để bảo toàn trục thời gian nhưng không tham gia tính loss hoặc chỉ số.

Phép chia dữ liệu được thực hiện theo đối tượng với 10 fold. Hai bản ghi của cùng một đối tượng luôn thuộc cùng vai trò; mỗi đối tượng xuất hiện đúng một lần trong tập kiểm tra ngoài fold. Tất cả cấu hình dùng cùng phép chia. Mô hình được lựa chọn trên validation, còn test được giữ kín cho đến khi hoàn tất toàn bộ chiến dịch validation.

### 2.2. Các cấu hình so sánh

Các nhãn E0–E6 chỉ là ký hiệu của các điều kiện thí nghiệm trong bài; diễn giải khoa học dựa trên thành phần được thay đổi:

| Điều kiện | Thành phần chính | Mục đích so sánh |
| --- | --- | --- |
| E0 | Mốc CNN–BiLSTM | Đối chứng nội bộ |
| E1 | Giữ bộ trích đặc trưng, thay mô hình chuỗi bằng TCN | Đánh giá mô hình chuỗi |
| E2 | Thay bộ trích đặc trưng bằng ResNet-1D | Đánh giá bộ trích đặc trưng |
| E3 | ResNet-1D–TCN với chế độ xử lý biên độ chính | Đánh giá cấu hình đề xuất |
| E4 | ResNet-1D–TCN với lọc dải | Đối chiếu tiền xử lý |
| E6 | ResNet-1D–TCN với z-score theo bản ghi | Đối chiếu đổi thang |

Một biến thể tiền xử lý được xác định là trùng dữ liệu với E4 nên không được đưa vào so sánh thống kê riêng.

### 2.3. Chỉ số và suy luận thống kê

Macro-F1 là chỉ số chính; Accuracy, Cohen’s kappa và F1 theo lớp được dùng để mô tả bổ sung. Khoảng tin cậy 95\% của chênh lệch được ước lượng bằng bootstrap bắt cặp theo cụm đối tượng. Kiểm định Wilcoxon hai phía được thực hiện trên kết quả theo đối tượng và hiệu chỉnh Holm cho bốn đối chiếu đã đăng ký trước.

Một chiến dịch đầy đủ được lặp lại với seed thứ hai sau khi kết quả chính đã được quan sát. Hai seed được phân tích riêng, không gộp $p$-value và không được xem là mẫu đại diện cho mọi khởi tạo.

### 2.4. Đánh giá vận hành và ngoài miền

Độ trễ được đo trên cùng phần cứng và cùng quy trình suy luận cho các cấu hình. Phép đo phản ánh forward inference, không bao gồm thời gian đọc dữ liệu, tiền xử lý hoặc huấn luyện.

Thời gian wall-clock của chiến dịch seed 123 được ghi nhận riêng trên cùng GPU V100. Sáu cấu hình được chạy tuần tự trên 10 fold; thời gian bao gồm huấn luyện, validation và hoàn thiện artifact, còn test chưa được mở. Đây là chi phí vận hành quan sát được trong giao thức hiện tại, không phải phép đo độc lập với phần cứng.

Đánh giá zero-shot được thực hiện trên mẫu SHHS1 gồm 180 đối tượng test đã khóa. Trọng số được giữ nguyên từ Sleep-EDF; không cập nhật mô hình bằng dữ liệu SHHS. Kết quả ngoài miền được báo cáo như một chiến dịch riêng, không nhập ngược vào các kiểm định chính trên Sleep-EDF.

### 2.6. Ablation nhóm đặc trưng C/P/N

Phân tích Gate 8 khảo sát ba nhóm ngữ cảnh: epoch hiện tại, epoch liền trước và epoch liền sau. Điều kiện đầy đủ sử dụng cả ba nhóm; các điều kiện ablation thay nhóm bị loại bằng giá trị trung bình được ước lượng từ dữ liệu huấn luyện của từng fold. Tiêu chí chính là Macro-F1 tại vùng chuyển pha. Phân tích này đo hiệu ứng dự báo có điều kiện trong một quy trình cụ thể; không phải phép đo phần trăm thông tin, không xác định quan hệ nhân quả và không thiết lập tương đương.

## 3. Kết quả

### 3.1. Hiệu năng trên Sleep-EDF

E3 đạt Macro-F1 bằng 0,7904, cao nhất trong sáu cấu hình. So với E6, chênh lệch là 0,0213 với khoảng tin cậy 95\% [0,0122; 0,0307] và $p$ Holm bằng 0,0012. Kết quả này cho thấy cách xử lý biên độ là yếu tố có liên hệ rõ nhất với hiệu năng trong giao thức hiện tại.

Các so sánh thay đổi mô hình chuỗi và bộ trích đặc trưng cho mức tăng mô tả nhỏ hơn, nhưng chưa đủ bằng chứng sau hiệu chỉnh Holm để khẳng định ưu thế riêng của TCN so với BiLSTM hoặc ResNet-1D so với bộ trích đặc trưng ban đầu. Lần lặp sau giao thức giữ hướng dương của các hiệu ứng chính; riêng mức ý nghĩa thống kê thay đổi theo seed.

### 3.2. Độ phức tạp và tốc độ

ResNet-1D–TCN sử dụng ít mô hình thành phần hơn và suy luận nhanh hơn khoảng 3,76 lần so với mốc E0. Đổi lại, hệ thống có 4,37 lần số tham số và bộ nhớ GPU cực đại cao hơn 28,4\%. Vì vậy, ưu điểm được xác định là sự đơn giản hóa trong vận hành và tốc độ suy luận, không phải giảm tài nguyên mô hình.

Thời gian chạy huấn luyện và validation của seed 123 được tổng hợp như sau:

| Cấu hình | Tổng 10 fold | Trung bình mỗi fold |
| --- | ---: | ---: |
| E0 — CNN đối chứng + BiLSTM | 33 giờ 35 phút 36 giây | 3 giờ 21 phút 34 giây |
| E1 — CNN đối chứng + TCN | 14 phút 17 giây | 1 phút 26 giây |
| E2 — ResNet-1D + TCN | 2 giờ 44 phút 57 giây | 16 phút 30 giây |
| E3 — ResNet-1D + TCN, xử lý chính | 3 giờ 16 phút 40 giây | 19 phút 40 giây |
| E4 — ResNet-1D + TCN, lọc dải | 2 giờ 55 phút 56 giây | 17 phút 36 giây |
| E6 — ResNet-1D + TCN, z-score | 2 giờ 24 phút 44 giây | 14 phút 28 giây |
| **Toàn bộ sáu cấu hình** | **45 giờ 12 phút 10 giây** | **4 giờ 31 phút 13 giây/fold** |

Trong chiến dịch này, E2 có thời gian chạy huấn luyện và validation tổng cộng thấp hơn E0 khoảng 12,2 lần; E3 thấp hơn khoảng 10,2 lần. Đây là lợi ích wall-clock của giao thức cụ thể, chịu ảnh hưởng của số epoch thực tế, dừng sớm, điều phối lượt chạy và phần cứng; không nên diễn giải thành ưu thế tốc độ phổ quát.

### 3.3. Kết quả zero-shot trên SHHS1

Trên mẫu SHHS1 đã khóa, E3 cao hơn E0 0,0412 điểm Macro-F1 trung bình theo đối tượng, với khoảng tin cậy 95\% [0,0314; 0,0512] và $p$ Holm bằng $2,65\times10^{-13}$. So với E6, mức tăng là 0,0274, với khoảng tin cậy 95\% [0,0182; 0,0370] và $p$ Holm bằng $1,87\times10^{-8}$. Hai khoảng tin cậy đều hoàn toàn dương và hiệu ứng được quan sát trên phần lớn đối tượng.

Kết quả này cung cấp bằng chứng ngoài miền đáng chú ý cho toàn bộ quy trình E3 trong mẫu SHHS1 đã đánh giá. Do kiến trúc và tiền xử lý được thay đổi đồng thời giữa một số đối chiếu, kết quả không được dùng để quy kết nguyên nhân cho một thao tác riêng.

### 3.4. Ablation nhóm đặc trưng C/P/N

Macro-F1 tại vùng chuyển pha của điều kiện đầy đủ là 0,6201. Các điều kiện loại nhóm ngữ cảnh đạt lần lượt 0,6191, 0,6177 và 0,6189. Chênh lệch so với điều kiện đầy đủ lần lượt là 0,0010, 0,0024 và 0,0012; cả ba khoảng tin cậy đều chứa 0 và $p$ Holm đều bằng 1,000.

Kết quả cho thấy trong thiết kế hiện tại chưa quan sát thấy lợi ích tăng thêm có ý nghĩa của các nhóm ngữ cảnh liền trước/liền sau tại vùng chuyển pha. Kết quả không cho phép kết luận P/N không có thông tin, cũng không cho phép khẳng định các điều kiện tương đương.

## 4. Thảo luận

Phát hiện nổi bật nhất là tính nhất quán của E3 so với E6: E3 đạt kết quả cao hơn trong miền Sleep-EDF và tiếp tục có lợi thế trong đánh giá SHHS1. Mẫu hình này cho thấy lựa chọn chế độ xử lý biên độ có vai trò thực nghiệm quan trọng trong dữ liệu đã khảo sát.

Giá trị của ResNet-1D–TCN thể hiện đồng thời ở hiệu năng cạnh tranh, tốc độ suy luận, thời gian chạy chiến dịch và số lượng thành phần vận hành. Trong phép đo chuẩn hóa, cấu hình này nhanh hơn 3.76 lần khi suy luận; trong chiến dịch seed 123, E2 có thời gian huấn luyện và validation thấp hơn E0 khoảng 12,2 lần. Đổi lại, hệ thống có 4.37 lần số tham số và bộ nhớ đỉnh cao hơn. Đây là một lợi thế thực dụng khi triển khai hoặc kiểm toán nhiều fold, dù phải chấp nhận chi phí tài nguyên cao hơn và giới hạn diễn giải vào phần cứng đã đo.

Các kết quả ablation và phân tích không gian đặc trưng cho thấy hiệu năng dự đoán không thể được suy ra chỉ từ độ tách cụm hoặc từ một cách diễn giải tầm quan trọng đặc trưng. Cách trình bày phù hợp là báo cáo trực tiếp chênh lệch dự báo, độ bất định và phạm vi áp dụng.

### 4.1. Ý nghĩa của kết quả Gate 8

Gate 8 không ủng hộ cách quy đổi đóng góp của P/N thành một tỷ lệ phần trăm thông tin. Đây là phân tích bổ sung với một training seed. Phân tích cho thấy hiệu ứng tăng thêm tại vùng chuyển pha nhỏ và chưa đủ bằng chứng thống kê. Đây là kết luận có điều kiện trong quy trình đã đánh giá; không phải bằng chứng rằng P/N hoàn toàn không có vai trò.

## 5. Hạn chế

Kết luận được xây dựng từ một chiến dịch chính và một lần lặp độ nhạy sau giao thức; vì vậy chưa mô tả toàn bộ biến thiên do khởi tạo. Đánh giá ngoài miền chỉ áp dụng cho mẫu SHHS1 đã khóa và một kênh EEG. Riêng Gate 8 chưa có SHHS; các kết quả ngoài miền được trình bày ở chiến dịch riêng. Các phép đo vận hành được thực hiện trên một điều kiện phần cứng cố định và không bao gồm toàn bộ chi phí của hệ thống.

Những giới hạn này không làm thay đổi các chênh lệch đã quan sát trong mẫu nghiên cứu, nhưng xác định phạm vi cần giữ khi diễn giải hoặc mở rộng kết luận.

## 6. Kết luận

Trong giao thức bắt cặp theo đối tượng, E3 là cấu hình có hiệu năng cao nhất trên Sleep-EDF và cho lợi thế rõ ràng trong đánh giá zero-shot SHHS1. ResNet-1D–TCN đồng thời mang lại tốc độ suy luận cao hơn, thời gian chạy chiến dịch thấp hơn ở E2 so với E0 và quy trình vận hành gọn hơn, với đánh đổi về số tham số và bộ nhớ GPU. Gate 8 không tìm thấy lợi ích tăng thêm có ý nghĩa của các nhóm ngữ cảnh liền trước/liền sau tại vùng chuyển pha.

Các kết quả ủng hộ việc ưu tiên đánh giá đồng thời tiền xử lý, hiệu năng và chi phí vận hành thay vì diễn giải một chỉ số đơn lẻ. Kết luận chỉ áp dụng cho dữ liệu, mẫu và giao thức đã khóa; không mở rộng thành tuyên bố tối ưu phổ quát, tương đương mô hình hoặc giá trị lâm sàng.

## 7. Hồ sơ bảng và hình

Các bảng định lượng, hình hiệu ứng, hình đánh đổi và ma trận bằng chứng được lưu cùng gói công bố Gate 8. Mọi chỉnh sửa phần Tóm tắt hoặc Kết luận cần được đối chiếu với `CLAIM_EVIDENCE_MATRIX.md`.
