# Gate 8 — kiểm định đóng góp của nhóm đặc trưng C/P/N

## 1. Câu hỏi nghiên cứu

Trong pipeline `15CNN + TCN` đã khóa ở E1, hai nhóm đặc trưng từ epoch liền trước (`P`) và epoch
liền sau (`N`) có mang lại giá trị dự báo tăng thêm so với chỉ dùng nhóm epoch hiện tại (`C`) hay
không, đặc biệt tại các vùng chuyển pha và đối với N1?

Gate 8 **không** ước lượng “phần trăm thông tin”, tầm quan trọng nhân quả hay tầm quan trọng toàn cục
của từng CNN. Kết luận hợp lệ chỉ là đóng góp dự báo có điều kiện của một nhóm đặc trưng khi giữ cố
định tập dữ liệu, split, TCN và các nhóm còn lại.

Gate này được thiết kế sau khi đã xem kết quả E0–E6 nên được ghi rõ là phân tích cơ chế bổ sung,
không phải kiểm định xác nhận độc lập hoàn toàn.

## 2. Các điều kiện

| Điều kiện | Nhóm còn thông tin | Nhóm được thay bằng hằng số | Số run mới |
|---|---|---|---:|
| Full CPN | C, P, N | không | 0 — tái sử dụng E1 đã khóa |
| CP | C, P | N | 10 |
| CN | C, N | P | 10 |
| C | C | P, N | 10 |

Mỗi nhóm có 25 chiều theo thứ tự đã tồn tại trong E1: `C[0:25]`, `P[25:50]`, `N[50:75]`.

Khi loại một nhóm, đầu vào vẫn giữ **75 chiều**. Mỗi chiều bị loại được thay bằng trung bình của chính
chiều đó, chỉ tính trên các epoch hợp lệ của tập huấn luyện trong fold. Cùng vector thay thế được áp
dụng cho train, validation và test. Không được tính lại trung bình bằng validation/test.

Thiết kế này giữ nguyên số tham số, kiến trúc, siêu tham số và seed khởi tạo TCN giữa các điều kiện;
khác biệt duy nhất có chủ đích là nhóm thông tin được cung cấp.

## 3. Dữ liệu và huấn luyện bị khóa

- Sleep-EDF Expanded Sleep Cassette, biến thể `paper_raw_v1`.
- Split 10-fold theo subject: `sleepedf_sc_10fold_seed42_v2.json`.
- Seed huấn luyện: 42 cho tất cả điều kiện.
- 15 CNN: chỉ nạp checkpoint E0 tương ứng từng fold, không huấn luyện lại.
- Full CPN: dùng nguyên prediction E1, không huấn luyện lại.
- CP/CN/C: huấn luyện TCN mới từ đầu; chọn checkpoint bằng Macro-F1 validation.
- Không tải, trích xuất đặc trưng hoặc suy luận trên test trong chiến dịch validation.
- Chỉ mở test một lần sau khi đủ 30/30 run validation, checkpoint và vector thay thế đã khóa.

Gate 8 không chạy thêm seed vì giới hạn tài nguyên. Do đó, độ bất định phản ánh biến thiên giữa
subject trong bộ dữ liệu, không phản ánh đầy đủ biến thiên do khởi tạo/huấn luyện.

## 4. Định nghĩa chuyển pha

Trong từng bản ghi, một anchor chuyển pha là epoch đầu tiên của pha mới khi:

1. epoch đó và epoch trước đều có nhãn 0–4;
2. chỉ số epoch gốc liên tiếp nhau;
3. hai nhãn khác nhau.

Unknown/Movement hoặc một khoảng trống chỉ số làm đứt chuỗi, không tạo anchor bắc qua khoảng trống.
Vùng chuyển pha chính gồm các epoch cách anchor không quá ±1 epoch trong cùng đoạn liên tục. Phân
tích độ nhạy dùng ±2 epoch.

Các cặp chuyển pha hỗ trợ được đăng ký trước: W↔N1, N1↔N2, N1↔REM và N2↔N3.

## 5. Tiêu chí và thống kê

So sánh luôn bắt cặp trên cùng subject, bản ghi, epoch gốc và nhãn thật.

- So sánh chính: `Full CPN − C`, Macro-F1 vùng chuyển pha ±1.
- Hai so sánh phụ then chốt: `Full CPN − CP` (đóng góp N) và `Full CPN − CN` (đóng góp P), cùng
  tiêu chí vùng chuyển pha ±1.
- Khoảng tin cậy 95%: bootstrap cụm bắt cặp theo subject, 10.000 lần, seed 2028.
- Kiểm định hỗ trợ: Wilcoxon signed-rank hai phía trên giá trị theo subject.
- Hiệu chỉnh Holm trên đúng ba so sánh vùng chuyển pha ±1 nêu trên.
- Báo cáo bổ sung: Macro-F1/accuracy toàn bộ, F1 và recall N1, recall N1 ở vùng chuyển pha/ổn định,
  Macro-F1 vùng ±2 và theo bốn loại chuyển pha.

Không suy ra tương đương/không thua kém từ việc `p > 0,05`. Gate 8 cũng không được dùng cụm từ
“P/N chứa x% thông tin”.

## 6. Cổng an toàn

1. `prepare`: kiểm tra đúng commit nguồn, Git sạch, cấu hình/split, 150 checkpoint CNN E0, 10 run E1,
   dữ liệu `paper_raw_v1`, và bảo đảm chưa có test artifact Gate 8.
2. `validation campaign`: chạy 30 run CP/CN/C; hỗ trợ resume, lưu SHA-256 của checkpoint, vector
   thay thế, prediction và metrics.
3. `validation audit`: bắt buộc đủ 30/30 và kiểm tra độc lập trước khi mở test.
4. `locked test`: yêu cầu câu xác nhận `OPEN-GATE8-LOCKED-TEST-ONCE`; có journal để resume; không
   huấn luyện lại và không chọn checkpoint lại.
5. `analysis`: chỉ chạy sau khi đủ 30/30 test prediction và bắt cặp hoàn toàn với E1.

## 7. Phạm vi kết luận

Nếu Full CPN tốt hơn C/CP/CN ở vùng chuyển pha, bằng chứng ủng hộ P/N có giá trị dự báo tăng thêm
trong pipeline cụ thể này. Nếu không, kết luận chỉ là chưa quan sát thấy lợi ích trong thiết kế và
độ mạnh thống kê hiện có; không được kết luận các CNN P/N “không có thông tin” trong mọi mô hình hay
mọi tình huống.
