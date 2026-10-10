# Kế hoạch hoàn thiện phản hồi thầy Trí — 05-10-2026

Trạng thái: kế hoạch, chưa khởi chạy thêm huấn luyện, suy luận SHHS hoặc upload.

## 1. Đích đến và phạm vi

Hoàn thiện một bài đánh giá lựa chọn mô hình và tiền xử lý khi chuyển quần thể, có:

1. Kết luận E3/E4 nhất quán với đối chiếu cùng seed.
2. Thực nghiệm can thiệp N3 có đối chứng và báo cáo đầy đủ đánh đổi.
3. Baseline ADAST/source-only có ngân sách học nguồn phù hợp, giải thích được vai trò của ngân sách thay vì chỉ trình bày một kết quả ADAST thấp.

Không đặt mục tiêu buộc ADAST phải thắng, tìm cấu hình thắng trên SHHS, hay mở một nghiên cứu phương pháp thích nghi mới.
Hoàn thành có nghĩa là trả lời được câu hỏi đặt ra, dù kết quả thuận lợi, bất lợi hoặc cho thấy đánh đổi.

## 2. Những nhánh đã đủ thông tin để đóng

| Nhánh | Thông tin đã có | Quyết định |
|---|---|---|
| E3/E4 | E4 hơn E3 trên SHHS ở cả seed 42 và 123; clipping không kích hoạt | Không train lại hoặc tìm thêm seed cho đối chiếu này. Trình bày E4 tốt hơn trong điều kiện đã đo; E3 giữ kết quả lịch sử và lợi ích so với E0. |
| Cơ chế chia 100 | Phép chia giữ hình dạng sóng; hai mô hình được huấn luyện riêng ở hai thang đầu vào | Không chạy một nghiên cứu BatchNorm/gradient chỉ để tạo lời giải thích vật lý. Không quy kết mất sóng chậm. |
| Trọng số lớp | Đủ mười fold; N3 tăng, N2/accuracy giảm | Không quét thêm trọng số theo điểm SHHS. Kết quả trả lời trực tiếp yêu cầu can thiệp của thầy. |
| Temperature/EM | Đủ mười fold; EM loại bỏ dự đoán N3 và giảm macro-F1 | Dừng biến thể đã thử; không đổi prior/floor liên tục để tìm điểm đẹp. |
| Giữ source CE, bỏ pseudo CE, bỏ đối kháng | Đã có ablation fold 0 và xác nhận giữ source CE ở fold 1; không có lợi ích đồng đều | Không mở rộng ba nhánh sửa loss lên mười fold. |
| Baseline UDA thứ hai | Chưa có câu hỏi riêng mà phương pháp thứ hai cần trả lời | Không thêm phương pháp chỉ để tăng số baseline. Xem lại chỉ khi thầy yêu cầu một họ phương pháp cụ thể hoặc kết quả hiện có chỉ ra thiếu sót rõ ràng. |

## 3. Vấn đề duy nhất có thể cần thêm training

Câu hỏi: Khi cả source-only và ADAST học đủ dữ liệu nguồn, tác động của thích nghi lên macro-F1 và N1/N3/REM trên SHHS thay đổi thế nào so với ngân sách nhỏ?

Lý do: lượt mười fold cũ có 1.140 update/mô hình. Full-source fold 0 có 36.870 update, fold 1 có 35.820 update. Các lượt phát triển cho thấy tăng ngân sách nâng điểm validation của cả source-only và ADAST. Vì vậy cần phân biệt tác động của ngân sách với tác động của thích nghi.

Điểm phải kiểm soát: lượt mười fold cũ dùng checkpoint epoch 30; lượt phát triển còn báo checkpoint tốt nhất theo validation nguồn. Không lấy best của ngân sách lớn so với final của ngân sách nhỏ rồi quy toàn bộ khác biệt cho ngân sách.

## 4. Bước A — kiểm kê và kiểm tra thiết kế, không GPU

Đầu ra: một bảng đối chiếu giao thức và một danh sách checkpoint có thể dùng lại.

### A1. Kiểm tra tính tương thích

Đối chiếu hai ngân sách và hai nhánh về:

- Chuyển đổi đơn vị tín hiệu, lọc/đổi thang, tạo epoch, loại nhãn và phân hoạch theo người.
- Backbone, hai classifier, attention, khởi tạo, seed, source batch order.
- Optimizer, lịch learning rate, source CE, similarity, adversarial và pseudo CE.
- Sinh pseudo-label, ghép logits khi inference, chọn checkpoint.
- Source-only không dùng dữ liệu adaptation khi huấn luyện; ADAST không dùng nhãn thật adaptation/test.
- Đường inference đã dùng: source attention cho source-only; target attention cho ADAST trên SHHS.

Phải nêu chính xác: full-source tăng số update, số lần xem target adaptation và thời điểm scheduler theo đơn vị update. Đây là thay đổi ngân sách huấn luyện, không phải phép cô lập riêng độ phủ nguồn.

Nếu phát hiện khác biệt ngoài ngân sách, kiểm tra xem nó chỉ là ghi log hay thay đổi phép học. Khác biệt làm thay đổi phép học phải được giải quyết hoặc mô tả thành so sánh giao thức rộng hơn trước khi quyết định chạy.

### A2. Tái sử dụng có điều kiện

- Ưu tiên cặp source-only_full_source/adast_full_source của lượt ngân sách fold 0, không chọn giữa các lượt ADAST lặp lại theo điểm cao nhất.
- Fold 1 dùng source_only_full_source/adast_reference_full_source của lượt xác nhận; không dùng nhánh giữ source CE.
- Dùng lại chỉ khi dữ liệu, phân hoạch, phép học, khởi tạo, số cập nhật, preprocessing, checkpoint và kết quả kiểm chứng tương thích với chiến dịch dự định.
- Nếu hai cặp đạt, cần thêm tám fold × hai nhánh = 16 mô hình, không train lại 20 mô hình.
- Nếu không đạt, ghi rõ lý do từng cặp và tính lại chi phí; không tự động train lại.

### A3. Kiểm tra kỹ thuật và nguồn lực

Chỉ chạy các test nhỏ còn thiếu: khớp phép cập nhật giữa runner mới/cũ, lặp target, batch cuối, đếm mẫu/update, chọn best, resume/RNG và không rò nhãn. Test đã đạt và mã không đổi thì dùng lại kết quả.

Ổ D hiện còn khoảng 0,11 GB, không dùng để chứa thêm payload/checkpoint. Ổ C còn khoảng 50 GB tại lần đọc này. Lập dự toán từ kích thước manifest và bản sao ZIP/giải nén trước khi tạo gói; dùng vị trí lưu được phép, không xóa lịch sử.

Kiểm tra quyền upload theo tài khoản/notebook hiện tại. Sự đồng ý cho fold 1 không tự mở rộng thành đồng ý cho toàn bộ các fold trên tài khoản đó. Chỉ upload source cần thiết và đúng năm người adaptation không nhãn sau khi phạm vi được xác nhận; 180 người SHHS test vẫn ở local.

Điều kiện sang bước B: câu hỏi và đối chiếu rõ, checkpoint tái sử dụng rõ, kiểm tra kỹ thuật đạt, đủ dung lượng và ngân sách được người dùng đồng ý.
Nếu không có nguồn lực hoặc quyết định giữ bài ở phạm vi hiện tại: dừng training, dùng baseline ngân sách nhỏ với mô tả đúng giao thức và hoàn thiện bản thảo.

## 5. Bước B — một chiến dịch full-source, hai nhánh

Không mở các nhánh sửa loss mới.

| Thành phần | Thiết kế |
|---|---|
| Nhánh A | Source-only tương ứng của ADAST |
| Nhánh B | ADAST reference, giữ công thức loss đang được đánh giá |
| Phân hoạch | Mười fold nguồn đã dùng; seed 123 |
| Ngân sách | 30 epoch, mỗi epoch học hết train nguồn của fold; số bước bằng ceil(N_train/128) |
| Đối chứng | Cùng khởi tạo và source order trong mỗi cặp; cùng ngân sách nguồn và quy tắc tối ưu |
| Adaptation | Cùng năm người, không nhãn; lặp target để đi hết source |
| Lưu | best theo source-validation macro-F1, final epoch 30, latest, optimizer, RNG và lịch sử |
| Theo dõi | Năm lớp, source/target attention trên validation nguồn, loss thành phần, số mẫu/update, thời gian |
| Phần cứng | GPU; không chuyển sang train CPU khi mất GPU |

Không chọn thêm epoch, loss, seed hoặc fold dựa vào SHHS. Không bỏ một fold chỉ vì điểm validation thấp. ADAST có thể đánh đổi điểm nguồn để thích nghi miền đích, nên thua source-only trên validation nguồn không phải lý do tự loại baseline.

Ước lượng lập kế hoạch: lượt fold 1 ba mô hình mất 49,6 phút gồm train và validation; 16 mô hình tương đương khoảng 4,4 giờ nếu tốc độ tương tự. Fold và runtime khác nhau nên phải lấy thời gian từng nhánh trong log để dự toán lại; không hứa hoàn tất dưới năm giờ từ phép nhân này.

Giữ giới hạn 18.000 giây của lượt chạy, có checkpoint mỗi epoch; không tự tăng giới hạn hoặc chia thành nhiều lượt để vượt ngân sách. Nếu dự toán cộng phần dự phòng vượt năm giờ, báo trước khi chạy. Khi đã chạy, cập nhật ETA từ epoch đầu của fold còn thiếu, không train một pilot riêng chỉ để đo tốc độ.

Dừng khi lỗi dữ liệu/NaN, mất GPU, hết dung lượng hoặc chạm giới hạn; lưu trạng thái và báo fold/nhánh/epoch. Không dừng vì điểm thấp, không tự restart.

## 6. Bước C — đánh giá đúng câu hỏi, không training

### C1. Đối chiếu chính về ngân sách

Dùng final epoch 30 cho cả ngân sách nhỏ và lớn, cho cả source-only và ADAST.

| | Ngân sách nhỏ | Full-source |
|---|---|---|
| Source-only | Kết quả mười fold hiện có | Chấm tổ hợp mười fold full-source |
| ADAST | Kết quả mười fold hiện có | Chấm tổ hợp mười fold full-source |

Tính ba đại lượng trên cùng người và cửa sổ chấm:

1. ADAST − source-only tại ngân sách nhỏ.
2. ADAST − source-only tại ngân sách full-source.
3. Chênh lệch giữa hai hiệu ứng trên.

Đại lượng thứ ba trả lời liệu tác động thích nghi có thay đổi theo ngân sách; không chỉ nhìn ADAST lớn cao hơn ADAST nhỏ trong khi source-only cũng tăng.

### C2. Vai trò của chọn checkpoint

Từ cùng lượt train full-source, chấm thêm cặp best theo validation nguồn. Đây chỉ là inference, không train thêm. Báo best/final riêng, không chọn kết quả đẹp hơn trên SHHS làm kết quả chính sau khi xem điểm.

Nếu best trùng final thì dùng lại dự đoán sau khi kiểm tra hash. Không mở chấm mọi epoch trên SHHS.

### C3. Chỉ số và kiểm chứng

- Chỉ số chính: macro-F1 trung bình theo người trên 180 SHHS.
- Phân tích từng lớp: precision, recall, F1 cả năm lớp; tập trung N3→N2, N1 và REM.
- Accuracy và macro-F1 gộp là chỉ số hỗ trợ.
- Sleep-EDF outer test: mỗi người chấm một lần bởi fold tương ứng; không gọi là ensemble.
- SHHS: trung bình xác suất đủ mười fold; dùng đúng đường attention/ghép logits và scoring mask đã nêu.
- Paired bootstrap theo người, 10.000 mẫu; phép so sánh hiệu ứng ngân sách dùng cùng chỉ số lấy mẫu người ở cả bốn ô.
- Không coi 10 fold, 30 epoch hoặc các mô hình chia sẻ dữ liệu là các lần lặp độc lập để tăng cỡ mẫu thống kê.
- Kiểm tra số đếm, chỉ số, hash và replay chọn mẫu bằng tiến trình riêng. Chỉ xuất tổng hợp vào báo cáo.

Lượt đánh giá nguồn/đích và kiểm chứng mười fold trước mất khoảng 20,7 phút local; lần mới có thêm best/final nên dùng số này để dự toán chứ không xem là cam kết. Kiểm chứng toàn validation từng checkpoint là phần thời gian riêng.

## 7. Quyết định sau kết quả — không chạy nối để tìm điểm đẹp

| Kết quả | Kết luận và hành động |
|---|---|
| Full-source ADAST cải thiện điểm tổng hợp, N3 và các lớp còn lại | Báo hiệu quả của baseline trong giao thức này; kết thúc chiến dịch. Không tự mở thêm phương pháp/seed. |
| N3 tăng nhưng N1/REM hoặc điểm tổng hợp giảm | Báo đánh đổi còn tồn tại sau tăng ngân sách; kết thúc. Không quét loss trên SHHS. |
| Cả hai nhánh đều tăng gần tương tự | Lợi ích chủ yếu gắn với ngân sách học; không gán phần tăng chung cho thích nghi. |
| Hiệu ứng ADAST đổi giữa ngân sách nhỏ/lớn | Báo tương tác quan sát được giữa ngân sách và thích nghi, cùng khoảng chênh lệch bắt cặp. |
| Best tốt hơn final, hoặc thứ hạng thay đổi | Báo độ nhạy theo quy tắc chọn checkpoint. Không đổi tên best thành kết quả final. |
| Khoảng chênh lệch chứa không | Trình bày mức chênh và khoảng; không suy tương đương, không tự chạy thêm chỉ để đạt p nhỏ. |
| Lỗi kỹ thuật | Kết quả lỗi không dùng để kết luận về ADAST; xác định lỗi trước khi đề xuất sửa/chạy lại. |

Không có yêu cầu ADAST phải vượt E3/E4 mới được xem là thí nghiệm thành công.
So sánh ADAST với source-only là đối chiếu tác động thích nghi. E3/E4 khác kiến trúc và recipe, nên nếu đặt cùng bảng thì mô tả hiệu năng các hệ thống, không quy chênh lệch đó cho riêng thích nghi.

## 8. Bước D — hoàn thiện bài và phản hồi

- E3/E4: giữ đối chiếu cùng seed và kết luận E4 cao hơn trên SHHS; không cần thí nghiệm mới.
- N3: dùng bảng can thiệp đã có; nêu tăng recall/F1 và tổn thất tương ứng.
- ADAST: một bảng chính cho câu hỏi ngân sách; best/final và diễn biến lớp chi tiết đặt ở phụ lục.
- Lượt ngân sách nhỏ và các ablation cũ vẫn được giữ để giải thích quá trình nghiên cứu, không trộn vào ensemble full-source.
- Không thêm biểu đồ trang trí. Nếu cần hình, ưu tiên chênh lệch bắt cặp với khoảng 95% để nhìn rõ ngân sách làm đổi tác động thích nghi; đưa bản xem trước cho người dùng trước khi chèn.
- Đồng bộ Anh/Việt/phụ lục/phản hồi thầy, kiểm tra từng claim với nguồn, dựng PDF/source ZIP và rà bố cục.
- Xác nhận đóng góp thực tế của tác giả và các mục nộp bài còn thiếu; không tự gán CRediT hoặc nộp bài.

Tiêu chí kết thúc: mỗi góp ý có câu trả lời, bảng bằng chứng và thay đổi tương ứng trong bài; không còn thí nghiệm nào bắt buộc chỉ để làm số đẹp hơn.

## 9. Hồ sơ đã dùng để lập kế hoạch

Reports/ADAST_DEVELOPMENT_BUDGET_20261004_VI.md;
Reports/ADAST_LOSS_ABLATION_20261004_VI.md;
Reports/ADAST_SMALL_CONFIRMATION_20261004_VI.md;
Reports/COLAB_ADAST_RESULTS_20261004_VI.md;
Reports/CLAIM_REVIEW_20261005_VI.md;
configs/adast_development_budget_v1_20261004.json;
configs/adast_small_confirmation_v1_20261004.json;
docs/ADAST_LOCAL_EVALUATION_20261004.md;
scripts/run_colab_adast_training.py;
scripts/run_adast_development_cuda.py.
