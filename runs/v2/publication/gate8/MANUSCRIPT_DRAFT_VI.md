# Bản nháp bài viết/khóa luận — hoàn tất Gate 8

> Trạng thái: bản nháp khoa học dựa trên artifact đã khóa. Cần bổ sung trích dẫn thư mục tài liệu tham
> khảo, thông tin hội đồng/tác giả và định dạng theo nơi nộp. Không thay các con số bằng kết quả chạy tay.

## Tiêu đề đề xuất

**Đánh giá bắt cặp ResNet-1D và mạng tích chập theo thời gian cho phân giai đoạn giấc ngủ một kênh trên
Sleep-EDF Expanded**

Tiêu đề thay thế ngắn hơn:

**Đơn giản hóa pipeline phân giai đoạn giấc ngủ một kênh: đánh đổi giữa hiệu năng, tốc độ và độ phức tạp**

## Tóm tắt

Phân giai đoạn giấc ngủ tự động từ điện não đồ một kênh có tiềm năng giảm chi phí xử lý đa ký giấc ngủ,
nhưng các so sánh mô hình thường bị ảnh hưởng bởi cách chia đối tượng, lựa chọn checkpoint và khác biệt
tiền xử lý. Nghiên cứu này tái triển khai một baseline 15CNN–BiLSTM và đánh giá tuần tự việc thay BiLSTM
bằng TCN, thay 15CNN bằng ResNet-1D, cùng các biến thể tiền xử lý trên Sleep-EDF Expanded Sleep Cassette.
Tất cả cấu hình dùng cùng split 10-fold theo đối tượng, seed huấn luyện 42 và checkpoint được chọn chỉ từ
validation; mỗi đối tượng xuất hiện đúng một lần trong test out-of-fold. Chỉ số chính là Macro-F1. Độ bất
định được ước lượng bằng bootstrap bắt cặp theo cụm đối tượng 10.000 lần; Wilcoxon signed-rank theo đối
tượng được hiệu chỉnh Holm trên bốn so sánh chính. Trong 78 đối tượng, 153 bản ghi và 195.469 epoch hợp
lệ, cấu hình E3 đạt Macro-F1 0.7904, cao nhất trong sáu cấu hình. So với z-score
theo bản ghi (E6), E3 cải thiện 0.0213, CI 95%
[0.0122; 0.0307], p Holm=0.0012. Thay BiLSTM bằng TCN
và thay 15CNN bằng ResNet-1D chỉ tạo mức tăng mô tả nhỏ, chưa có ý nghĩa sau Holm. Trên Tesla V100,
ResNet-1D–TCN suy luận nhanh hơn baseline khoảng 3.76 lần, nhưng có
4.37 lần số tham số và peak VRAM cao hơn
28.4%. Kết quả cho thấy lợi ích chính đến từ
lựa chọn xử lý biên độ, trong khi pipeline ResNet-1D–TCN mang lại sự đơn giản hóa vận hành và tăng tốc
suy luận với đánh đổi về tham số và bộ nhớ. Kết luận hiện chỉ áp dụng in-domain trên Sleep-EDF và một
training seed.

**Từ khóa:** phân giai đoạn giấc ngủ; EEG một kênh; ResNet-1D; TCN; Sleep-EDF; thiết kế thực nghiệm bắt cặp.

## 1. Đặt vấn đề

Phân giai đoạn giấc ngủ là bước nền tảng trong phân tích đa ký giấc ngủ. Việc chấm thủ công đòi hỏi thời
gian và chuyên môn, tạo động lực cho các phương pháp học sâu tự động. Tuy nhiên, đánh giá trong tín hiệu
y sinh phải kiểm soát biến thiên giữa người bệnh: nếu các mô hình dùng split khác nhau, chênh lệch có thể
phản ánh thành phần đối tượng thay vì kiến trúc. Ngoài ra, tuyên bố “đơn giản” cần phân biệt số mô hình
thành phần, thời gian suy luận, số tham số và bộ nhớ.

Nghiên cứu này giải quyết ba câu hỏi. Thứ nhất, TCN có cải thiện baseline BiLSTM khi giữ nguyên 15CNN
hay không? Thứ hai, ResNet-1D có thay thế 15CNN hiệu quả khi dùng chung TCN hay không? Thứ ba, các lựa
chọn lọc và biến đổi biên độ ảnh hưởng thế nào đến hiệu năng? Điểm trọng tâm là một giao thức bắt cặp
theo đối tượng, khóa test cho đến khi hoàn tất lựa chọn checkpoint và công bố đầy đủ đánh đổi tính toán.

Các đóng góp chính gồm: (1) tái triển khai pipeline baseline và pipeline ResNet-1D–TCN trong cùng giao
thức; (2) ablation tuần tự tách thay đổi mô hình chuỗi, bộ trích đặc trưng và tiền xử lý; (3) phân tích
thống kê bắt cặp ở mức đối tượng; và (4) benchmark có kiểm soát về latency, throughput, tham số và VRAM.

## 2. Phương pháp

### 2.1. Dữ liệu và nhãn

Nghiên cứu sử dụng Sleep-EDF Expanded, phân tập Sleep Cassette, gồm 78 đối tượng và 153 bản ghi. Kênh
EEG Fpz-Cz được lấy mẫu ở 100 Hz và chia thành epoch 30 giây. Năm lớp đánh giá là W, N1, N2, N3 và REM.
Movement/Unknown được giữ trong chuỗi với nhãn −1 để bảo toàn vị trí thời gian nhưng bị mask khỏi loss và
metrics. Tổng cộng có 195.469 epoch hợp lệ.

### 2.2. Chia dữ liệu và phòng tránh rò rỉ

Split 10-fold được tạo theo đối tượng với seed 42; hai đêm của cùng một người luôn cùng vai trò. Trong
mỗi outer fold, tập test là một fold đối tượng, validation là fold kế tiếp theo modulo 10 và phần còn lại
là train. Cùng một split được dùng cho mọi cấu hình. Mỗi đối tượng xuất hiện đúng một lần trong test
out-of-fold. Checkpoint tốt nhất được chọn bằng validation Macro-F1; test bị khóa trong suốt huấn luyện
và chỉ được mở một lần sau khi đủ 60 run validation-only.

### 2.3. Cấu hình thí nghiệm

- E0: 15CNN tạo 75 xác suất từ epoch hiện tại/liền trước/liền sau, sau đó BiLSTM.
- E1: giữ 15CNN của E0 và thay BiLSTM bằng TCN chung.
- E2: thay 15CNN bằng ResNet-1D tạo embedding 128 chiều, giữ TCN và dữ liệu raw.
- E3: ResNet-1D–TCN với gói lọc và chia hằng số 100.
- E4: ResNet-1D–TCN chỉ với lọc dải.
- E6: ResNet-1D–TCN với lọc và z-score theo bản ghi.

E5 bị loại vì dữ liệu của biến thể clipping trùng bitwise với E4 (`clip_fraction=0`). Vì vậy không tạo
p-value E5−E4.

### 2.4. Chỉ số và thống kê

Macro-F1 gộp trên toàn bộ dự đoán test out-of-fold là chỉ số chính; Accuracy, Cohen's kappa và F1 từng
lớp là chỉ số hỗ trợ. CI 95% cho chênh lệch Macro-F1 được tính bằng bootstrap bắt cặp theo cụm đối tượng
10.000 lần, giữ toàn bộ epoch của đối tượng được lấy mẫu. Wilcoxon signed-rank hai phía dùng Macro-F1
từng đối tượng. Holm được áp dụng chỉ cho bốn so sánh định trước: E1−E0, E2−E1, E3−E2 và E3−E6.
E4−E2 là phân tích cơ chế thứ cấp.

### 2.5. Benchmark và phân tích đặc trưng

Benchmark dùng checkpoint thật fold 00 trên Tesla V100 16 GB, input `(1,100,1,3000)`, ba vòng xáo
thứ tự; mỗi mô hình có 20 lượt làm nóng và 100 lượt đo/vòng với đồng bộ CUDA. Phép đo không gồm I/O,
preprocessing, cache và training. Phân tích không gian đặc trưng so sánh E1/E2 trên cùng epoch test:
200 epoch/lớp/fold, StandardScaler, PCA 20 chiều và Silhouette Score; t-SNE fold 00 chỉ dùng mô tả.


### 2.6. Ablation nhóm đặc trưng C/P/N

Gate 8 đánh giá đóng góp dự báo có điều kiện của các nhóm C (epoch hiện tại), P (epoch liền trước) và
N (epoch liền sau) trong E1. Full CPN tái sử dụng prediction E1; CP, CN và C huấn luyện lại TCN trên
10 fold. Đầu vào luôn giữ 75 chiều và cùng kiến trúc TCN. Nhóm bị loại được thay bằng trung bình từng
chiều chỉ tính từ epoch train hợp lệ trong fold, sau đó áp dụng nguyên vector cho train, validation và
test. Tiêu chí chính là Macro-F1 tại vùng ±1 epoch quanh chuyển pha nhãn hợp lệ liên tiếp. Ba so sánh
Full CPN−C, Full CPN−CP và Full CPN−CN dùng bootstrap cụm bắt cặp 10.000 lần, Wilcoxon theo đối tượng
và hiệu chỉnh Holm. Gate 8 là phân tích cơ chế bổ sung với một training seed. Đây không phải phép đo phần trăm thông tin hay kiểm định tương đương.

## 3. Kết quả

### 3.1. Hiệu năng test

E0, E1, E2, E3, E4 và E6 lần lượt đạt Macro-F1
0.7754, 0.7802, 0.7835,
0.7904, 0.7891 và 0.7691. E3 có
giá trị mô tả cao nhất; E6 thấp nhất trong nhóm ResNet-1D–TCN.

E1−E0 tăng 0.0048, nhưng CI chứa 0 và p Holm=
0.1022. E2−E1 tăng 0.0033, CI chứa 0 và
p Holm=0.1036. Do đó chưa đủ bằng chứng kết luận TCN tốt hơn BiLSTM hoặc
ResNet-1D tốt hơn 15CNN về Macro-F1 với seed hiện tại.

E3−E2 tăng Macro-F1 gộp 0.0070 với CI vừa vượt 0, nhưng Wilcoxon
không có ý nghĩa và số đối tượng thắng/thua là 37/41.
Điều này cho thấy lợi ích gộp không đồng đều theo đối tượng. E3−E6 là kết quả nhất quán nhất, với chênh
lệch 0.0213, CI hoàn toàn dương và p Holm=0.0012.

### 3.2. Độ phức tạp và tốc độ

E0 có 248,630 tham số; E1 có 640,950; E2–E6 có
1,085,578. ResNet-1D–TCN giảm từ 16 xuống 2 mô hình thành phần và có latency trung
vị khoảng 3.575 ms/100 epoch, so với
13.431 ms của E0. Tuy nhiên, số tham số tăng
4.37 lần và peak VRAM tăng
28.4%.

### 3.3. Không gian đặc trưng

Silhouette E2 thấp hơn E1 trong 10/10 fold; chênh lệch E2−E1 trung bình là -0.1079.
Do đó phép đo không hỗ trợ giả thuyết embedding ResNet tự động tạo cụm lớp tốt hơn softmax 15CNN.
Kết quả này chỉ mang tính hỗ trợ và không thay thế đánh giá dự đoán chuỗi.


### 3.4. Ablation nhóm đặc trưng C/P/N

Macro-F1 toàn bộ của Full CPN, C, CP và CN lần lượt là
0.780230, 0.777477,
0.777733 và 0.781585. Tại vùng chuyển
pha ±1, các giá trị tương ứng là 0.620055,
0.619102, 0.617686
và 0.618895.

Full CPN−C tại vùng chuyển pha là 0.000953, CI 95%
[-0.004588; 0.006568]. Full CPN−CP
và Full CPN−CN lần lượt là 0.002369 và
0.001160; cả ba p Holm đều bằng 1,000. Vì vậy
chưa có bằng chứng thống kê rằng P/N tạo lợi ích tăng thêm cho Macro-F1 vùng chuyển pha trong thiết kế
hiện tại. Kết quả không thiết lập tương đương và không cho phép kết luận P/N không có thông tin.

## 4. Thảo luận

Kết quả cho thấy cần tách hai loại đóng góp. Về chất lượng dự đoán, thay đổi kiến trúc chuỗi và bộ trích
đặc trưng đem lại mức tăng mô tả nhỏ nhưng chưa vượt qua ngưỡng suy luận sau hiệu chỉnh đa kiểm định.
Ngược lại, cách xử lý biên độ E3 so với z-score E6 tạo hiệu ứng lớn hơn và nhất quán ở cả bootstrap lẫn
Wilcoxon. Điều này gợi ý việc bảo toàn quan hệ biên độ có liên quan đến hiệu năng trong dữ liệu hiện tại,
nhưng chưa chứng minh quan hệ nhân quả hoặc khả năng khái quát sang cơ sở dữ liệu khác.

Về vận hành, ResNet-1D–TCN thay 15 mô hình CNN bằng một bộ trích đặc trưng, giúp pipeline ít thành phần
hơn và nhanh hơn khoảng 3.76 lần. Đổi lại, mô hình có nhiều tham số và sử
dụng peak VRAM cao hơn. Do đó “đơn giản hóa” nên được dùng theo nghĩa kiến trúc vận hành, không đồng nhất
với tiết kiệm tài nguyên.

Sự khác biệt giữa Silhouette và Macro-F1 nhấn mạnh rằng biểu đồ t-SNE hoặc độ gọn cụm không thể thay thế
đánh giá tác vụ. Softmax 15CNN được tối ưu trực tiếp theo nhãn và có cấu trúc 15×5 xác suất, trong khi
embedding ResNet có thể mã hóa thông tin phục vụ TCN mà không tạo cụm Euclid gọn.


### 4.1. Ý nghĩa của kết quả Gate 8

Ablation theo nhóm không xác nhận cách diễn giải cũ rằng các CNN P/N chỉ đóng góp một tỷ lệ thông tin
cố định. Full CPN có một số lợi thế mô tả ở F1 N1, nhưng hiệu ứng vùng chuyển pha nhỏ, CI chứa 0 và
không nhất quán theo đối tượng. CN thậm chí có Macro-F1 toàn bộ mô tả cao hơn Full CPN. Cách trình bày
đúng là báo cáo hiệu ứng tăng thêm có điều kiện và độ bất định, không quy đổi thành phần trăm thông tin.
Việc không bác bỏ giả thuyết không cũng không chứng minh C/CP/CN tương đương Full CPN.

## 5. Hạn chế

Nghiên cứu mới dùng một training seed 42, nên chưa định lượng độ ổn định theo khởi tạo. Benchmark chỉ
thực hiện trên một Tesla V100, một batch và một độ dài chuỗi, không gồm I/O hay preprocessing. Dữ liệu
chỉ là Sleep-EDF Expanded in-domain; chưa có SHHS, domain shift, zero-shot, đa kênh hoặc xác nhận lâm
sàng. Phân tích không gian đặc trưng được thực hiện sau Gate 5 và chỉ là bằng chứng hỗ trợ. Gate 8 được thiết kế sau khi đã xem E0–E6 và chỉ dùng một seed; do đó đây là phân tích cơ chế bổ sung, không phải xác nhận độc lập. Không có kiểm định tương đương hoặc không thua kém. N1 vẫn là lớp
khó và EEG một kênh không chứa đầy đủ thông tin chuyển động mắt.

## 6. Kết luận

Trong giao thức bắt cặp theo đối tượng, E3 đạt hiệu năng mô tả cao nhất và tốt hơn E6 một cách nhất quán,
cho thấy lựa chọn biến đổi biên độ là yếu tố đáng chú ý nhất. ResNet-1D–TCN mang lại pipeline ít mô hình
thành phần hơn và suy luận nhanh hơn, nhưng không tiết kiệm tham số hoặc VRAM. Các kết luận chỉ áp dụng
cho Sleep-EDF Expanded và seed 42; bước xác nhận tiếp theo nên đánh giá thêm seed và dữ liệu ngoài miền
theo một giao thức đăng ký trước riêng biệt.

Gate 8 không tìm thấy lợi ích tăng thêm có ý nghĩa của P/N tại vùng chuyển pha, nhưng cũng không chứng minh các ablation tương đương hoặc P/N vô dụng.

## 7. Hướng dẫn sử dụng bảng và hình

- Bảng hiệu năng: `TABLES.md`, phần “Hiệu năng test out-of-fold”.
- Hình hiệu ứng: `figure_primary_effects.png`.
- Hình đánh đổi: `figure_performance_speed_tradeoff.png`.
- Hình đặc trưng: `figure_feature_silhouette.png`; t-SNE gốc nằm ở artifact Gate 6.
- Trước khi sửa Abstract/Kết luận, đối chiếu `CLAIM_EVIDENCE_MATRIX.md`.

- Bảng ablation: `table_context_ablation.csv` và `table_context_ablation_comparisons.csv`.
- Hình Gate 8: `figure_context_ablation_effects.png`.
