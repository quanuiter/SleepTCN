# Chẩn đoán và bước cải thiện tiếp theo — 04/10/2026

## Những việc đã thực sự chạy

Đã dùng CPU đọc lại 20 mô hình ADAST/source-only trên **tập xác thực nguồn của cả 10 fold**, và trên dữ liệu thích nghi không nhãn của 5 người đã sử dụng trước đó. Không train lại, không đọc 180 người SHHS đánh giá, không dùng nhãn SHHS để chọn mô hình. Thời gian: 136,55 giây, chưa tính lượt kiểm chứng riêng.

Trong bảng dưới, macro-F1 là điểm cân bằng giữa năm giai đoạn; recall là phần trăm đoạn thực sự thuộc một giai đoạn mà mô hình nhận ra đúng. Đây là điểm **xác thực phục vụ chẩn đoán**, không phải kết quả kiểm tra độc lập hay ensemble 10 fold.

| Đường suy luận, gộp hai đầu ra bằng maximum | Macro-F1 | Nhận đúng N1 | Nhận đúng N3 | Nhận đúng REM |
| --- | ---: | ---: | ---: | ---: |
| Source-only, attention nguồn | 0,6649 | 14,15% | 77,53% | 60,76% |
| ADAST, attention nguồn | 0,5282 | 0,153% | 62,87% | 23,22% |
| ADAST, attention đích | 0,5275 | 0,200% | 60,86% | 25,37% |

Đã kiểm tra riêng từng đầu phân loại và cách lấy trung bình hai đầu ra. N1 vẫn gần như biến mất ở ADAST trong cả hai đường attention. Vì vậy, **chỉ đổi cách gộp đầu ra hoặc đổi attention không giải quyết được vấn đề**. Đây cũng không phải lỗi chỉ xuất hiện khi chuyển sang SHHS: nó đã có trên tập xác thực Sleep-EDF.

Kết quả này chưa chứng minh nhãn giả, học đối kháng hoặc ngân sách huấn luyện là nguyên nhân cụ thể. Muốn xác định nguyên nhân cần train các đối chứng chỉ thay một yếu tố. Các mô hình lịch sử chỉ giữ trạng thái cuối, nên không thể dùng chúng để truy ngược chính xác epoch bắt đầu suy giảm.

## Một hạn chế đã định lượng được của lượt ADAST cũ

Mỗi mô hình có 1.140 lần cập nhật, tương ứng 145.920 lượt trình bày đoạn nguồn: chỉ khoảng 0,91–0,96 lần đi qua toàn bộ nguồn nếu quy đổi theo số đoạn. Vì có lấy mẫu lặp, đây không có nghĩa mô hình đã nhìn thấy toàn bộ nguồn. Riêng fold 0, khoảng 60,96% đoạn nguồn khác nhau đã xuất hiện ít nhất một lần. Chưa thể kết luận tăng thời gian sẽ khắc phục N1, nhưng cần kiểm tra khả năng học nguồn trước khi mở rộng đánh giá chuyển quần thể.

## Thử trọng số mềm hơn

Đã chuẩn bị và kiểm chứng gói đặc trưng Sleep-EDF của **train/validation fold 0**, giữ nguyên bộ trích xuất đặc trưng E3 đã kiểm chứng. Không có tập kiểm tra ngoài, SHHS, ID hay EEG thô trong gói. Bốn bộ phân loại TCN mới train trên CUDA T4, cùng khởi tạo và thứ tự dữ liệu:

- Không tăng trọng số: mức 0.
- Tăng nhẹ: mức 0,25.
- Tăng vừa: mức 0,5.
- Tăng như lượt đã thử trước: mức 1.

Ví dụ trọng số N3 lần lượt khoảng 1; 1,34; 1,80; 3,25. Mục đích là kiểm tra mức ưu tiên vừa phải có giữ được cân bằng giữa các lớp hơn không. Chọn epoch bằng macro-F1 xác thực nguồn, tối đa 300 epoch, dừng sớm sau 30 epoch không cải thiện; cả lượt tối đa 5 giờ. Không tự chuyển sang CPU nếu GPU mất kết nối.

**Đã hoàn tất bốn bộ phân loại mới trên GPU T4 và kiểm chứng kết quả tải về.** Lượt training/xác thực và kiểm tra trong runner mất 371,17 giây (khoảng 6 phút 11 giây), không gồm upload, download hay kiểm chứng tại máy. Tổng thời gian vòng train/validation của bốn nhánh là 364,03 giây. Không dùng thời gian này thay benchmark vận hành lịch sử của paper vì đây là train bộ phân loại trên đặc trưng có sẵn, không phải toàn pipeline.

Kết quả dưới đây chỉ dùng **18.763 đoạn có nhãn của tập validation nguồn fold 0**. Mỗi nhánh chọn epoch tốt nhất bằng macro-F1 trên chính tập này; do đó đây là số liệu phát triển/chọn cấu hình, không phải ước lượng kiểm tra độc lập.

| Mức ưu tiên lớp ít gặp | Macro-F1 | Nhận đúng N1 | Nhận đúng N2 | Nhận đúng N3 | Độ chính xác của dự đoán N3 | F1 của N3 | Epoch chọn / đã chạy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Không tăng — 0 | 0,7798 | 60,41% | 83,92% | 66,67% | 86,79% | 0,7541 | 12 / 42 |
| Nhẹ — 0,25 | 0,7845 | 61,93% | 85,18% | 73,29% | 82,90% | 0,7780 | 6 / 36 |
| Vừa — 0,5 | 0,7875 | 63,45% | 83,04% | 80,33% | 78,65% | 0,7948 | 6 / 36 |
| Mạnh — 1 | 0,7858 | 59,71% | 83,01% | 77,92% | 80,41% | 0,7914 | 23 / 53 |

“Nhận đúng N3” là recall: trong các đoạn thực sự là N3, bao nhiêu đoạn được nhận ra. “Độ chính xác của dự đoán N3” là precision: trong các đoạn mô hình gọi là N3, bao nhiêu đoạn thực sự đúng. Vì vậy không thể chỉ nhìn recall tăng mà nói đã giải quyết vấn đề.

Mức 0,5 là **ứng viên phát triển theo tiêu chí macro-F1 đã khóa**: hơn mức 0 khoảng 0,00774 và hơn mức 1 khoảng 0,00178. So với không tăng, recall N3 tăng 13,67 điểm phần trăm, nhưng precision N3 giảm 8,14 điểm phần trăm và recall N2 giảm 0,88 điểm phần trăm; accuracy tổng giảm từ 81,64% xuống 81,57%. Mức 0,25 có recall N2 tốt hơn mức 0,5 trong lượt này. Chưa có kiểm chứng độ ổn định qua seed/fold, kiểm định cho lựa chọn mới hay đánh giá SHHS mới. Không gọi mức 0,5 là cấu hình tốt nhất tổng quát hoặc đã khắc phục lỗi chuyển quần thể.

### Kiểm chứng và sự cố được giữ lại

Lần khởi động đầu gặp lỗi kiểu đường dẫn ở bước ghi hash runner, trước khi có epoch train; không tạo checkpoint. Đã sửa lỗi, thêm kiểm thử bước khởi động, giữ gói và ô lỗi nguyên trạng rồi tạo gói Fix1 riêng. Không thay code/config của lượt đang chạy.

Đã kiểm tra SHA của ZIP tải về theo log Colab, allowlist và hash từng thành viên, giao thức/đặc trưng/bộ trích xuất, cả bốn checkpoint tốt nhất và checkpoint cuối, quy tắc chọn epoch, trọng số và thứ tự dữ liệu. Tiến trình độc lập tại máy dùng checkpoint để suy luận lại validation trên CPU: **cả bốn ma trận nhầm lẫn khớp hoàn toàn với CUDA**.

Lượt kiểm chứng ban đầu dừng khi tái tạo khởi tạo tại máy PyTorch 2.5.1 không khớp từng bit với Colab PyTorch 2.11.0. Đã kiểm tra riêng trên đúng phiên training: khởi tạo CPU và sau chuyển sang CUDA cho cùng hash, khớp hash ghi ở cả bốn nhánh. Đã tải bằng chứng tensor khởi tạo về và xác minh digest độc lập. Chênh lệch tuyệt đối lớn nhất giữa hai môi trường là khoảng 7,45 × 10⁻⁹; không giả định hai môi trường sẽ train lại cho kết quả hoàn toàn giống nhau. Sự khác biệt này không ngăn việc tái dựng đúng các ma trận nhầm lẫn từ checkpoint đã train.

Kiểm thử mới: **18 ca đạt**, gồm trọng số, vai trò train/validation, chẩn đoán đầu ra, khởi động runner, từ chối fallback CPU, ZIP và ràng buộc bằng chứng khởi tạo; 20 kiểm thử hồi quy liên quan cũng đã đạt. Chẩn đoán ADAST có tiến trình kiểm chứng độc lập tái dựng tất cả dự đoán/ma trận nhầm lẫn từ logits đã lưu và đối chiếu hash đầu vào/checkpoint; đạt.

## Thứ tự xử lý ADAST tiếp theo

1. Khóa một phép thử phát triển fold 0 trước khi chạy: giữ dữ liệu và kiến trúc hiện có, chưa mở thêm người thích nghi.
2. So sánh source-only và ADAST với cùng ngân sách; lưu điểm xác thực từng lớp theo epoch để biết N1 bắt đầu giảm khi nào.
3. Tách các yếu tố bằng đối chứng: ngân sách nhìn nguồn, trọng số giám sát nguồn, thành phần thích nghi/nhãn giả. Không đồng thời đổi tất cả rồi gán cải thiện cho một yếu tố.
4. Chỉ sau khi có cấu hình phát triển ổn định mới khóa giao thức mở rộng các fold. Không chọn bằng điểm trên 180 SHHS đã xem.

Phần train ADAST mới này **chưa chạy**; không nhầm với chẩn đoán 10 fold hoặc bốn bộ phân loại TCN đã hoàn tất. Đã lập bản nháp giao thức đối chứng ngân sách nguồn và các thành phần loss. Khi tăng ngân sách, mức tiếp xúc với đích và tiến trình lịch học cũng có thể thay đổi: phải ghi nhận các yếu tố này, không gọi đó là phép thử chỉ thay độ bao phủ nguồn. Chưa mở chiến dịch ADAST/mười fold mới hoặc tải thêm dữ liệu đích.

## Bản thảo và lựa chọn E3/E4

Giữ E3 làm cấu hình tham chiếu lịch sử; không đổi tên hoặc viết lại lịch sử thí nghiệm. E4 tốt hơn trên phép đánh giá SHHS đã thực hiện phải được trình bày đúng, nhưng không xếp hạng trực tiếp với các can thiệp dùng cửa sổ đánh giá khác. Chưa có căn cứ gọi bất kỳ cách mới nào là đã khắc phục triệt để N3.

Các kết quả phát triển trong báo cáo này giữ riêng khỏi paper nộp hiện tại. Chưa thay số liệu PDF/ZIP nộp bằng điểm xác thực của một fold, dù số liệu phát triển đã được kiểm chứng.
