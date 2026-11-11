# Phụ lục A3: hiệu chỉnh N2/N3 theo biên độ và đoàn hệ đích thứ hai

Tệp cấu hình khóa: `configs/amendment_a3.json`. Giao thức gốc:
`docs/SHHS_HOLDOUT_CONFIRMATORY_PROTOCOL_V1.md`; các phụ lục trước: `docs/SHHS_HOLDOUT_AMENDMENT_A1.md`,
`docs/SHHS_HOLDOUT_AMENDMENT_A2.md`. Ngày khóa: 11/11/2026 (mốc thời gian là commit chứa tệp này trên
GitHub).

## 1. Mục đích

A3 trả lời hai điểm yếu của bản thảo:

| Phân tích | Điểm yếu | Cách trả lời |
|---|---|---|
| **F** Hiệu chỉnh N2/N3 theo biên độ | Lỗi N3→N2 được chẩn đoán nhưng chưa có biện pháp nào giảm được mà không cần dữ liệu đích | Một quy tắc hậu xử lý lấy toàn bộ tham số từ Sleep-EDF, không dùng dữ liệu hay nhãn của miền đích |
| **G** Đoàn hệ đích thứ hai | Mọi kết luận chuyển quần thể chỉ dựa trên SHHS1 | Chấm các mô hình đã khóa trên Sleep-EDF Telemetry; ISRUC nhóm I khi lấy được dữ liệu |

A3 **không** thay đổi, kiểm định lại hay diễn giải lại H1–H3, phân tích gốc, A1 hay A2. A3 là phân tích
**mô tả, sau xác nhận**, không có kiểm định giả thuyết.

## 2. Những gì đã biết trước khi khóa

- **SHHS1 holdout:** nhãn đã dùng trong phân tích gốc, A1, A2 và mô tả hậu nghiệm theo giai đoạn (Mục bổ
  sung S8 của bản thảo, commit `4f1ebee`). Từ mô tả đó, các tác giả đã biết ngưỡng độ lệch chuẩn chọn trên
  Sleep-EDF (16,51 µV) đặt 84,0% epoch N3 tham chiếu của holdout ở phía trên và 82,2% epoch N2 ở phía
  dưới. Đây là động cơ của phân tích F. Tác dụng của quy tắc khi kết hợp với dự đoán của mô hình (độ thu
  hồi, độ chính xác, macro-F1 sau hiệu chỉnh) **chưa được tính** ở bất kỳ đâu trước khi khóa.
- **Sleep-EDF Telemetry và ISRUC:** chưa từng được tải, xử lý hay xem. Vì vậy đây là phép thử chưa bị ảnh
  hưởng bởi bất kỳ lựa chọn nào của nghiên cứu, kể cả với quy tắc F.

## 3. Phân tích F: quy tắc hiệu chỉnh (khóa)

- **Đặc trưng:** độ lệch chuẩn (µV) của epoch 30 giây trên đầu vào `bandpass_v2` (lọc thông dải 0,5–30 Hz,
  không cắt, không đổi thang). Đặc trưng này dùng cho mọi arm, kể cả arm có đầu vào khác.
- **Dữ liệu ước lượng:** chỉ các epoch N2 và N3 tham chiếu của Sleep-EDF. Không dùng dữ liệu đích, không
  dùng dự đoán của mô hình, không có tham số nào được chỉnh theo hiệu năng.
- **R1 (quy tắc cứng, biến thể chính):** ngưỡng là giá trị, trong 999 phân vị gộp (0,001–0,999) của các
  epoch N2 và N3 nguồn, cho tổng (tỷ lệ N3 từ ngưỡng trở lên + tỷ lệ N2 dưới ngưỡng) lớn nhất. Epoch được
  dự đoán là N2 và có đặc trưng từ ngưỡng trở lên được đổi thành N3. Không thay đổi gì khác.
- **R2 (quy tắc mềm, biến thể phụ):** hồi quy logistic N3 so với N2 trên ln(đặc trưng) (scikit-learn,
  `C=1e6`, `lbfgs`). Tỷ số hợp lý = odds khớp được chia cho tỷ số số epoch N3:N2 của nguồn. Với mỗi epoch,
  khối xác suất p(N2)+p(N3) được chia lại theo odds p(N3)/p(N2) nhân với tỷ số hợp lý; các lớp khác giữ
  nguyên; sau đó lấy argmax.
- **Áp dụng trên miền đích:** ngưỡng và hồi quy ước lượng trên toàn bộ 153 bản ghi Sleep-EDF, áp dụng lên
  xác suất ensemble mười mô hình.
- **Chi phí trên miền nguồn:** trên Sleep-EDF, quy tắc được áp dụng lên dự đoán ngoài fold; ngưỡng và hồi
  quy của mỗi fold chỉ ước lượng trên các bản ghi nằm ngoài fold kiểm tra đó. Kết quả này được báo cáo dù
  có lợi hay có hại, vì một quy tắc giúp miền đích có thể làm giảm hiệu năng trên miền nguồn.
- **Arm:** mười arm lưu trữ (E0, E2, E3, E4, E6 ở seed 42 và 123); trên holdout thêm E4 seed 2025, 2027,
  2029 và E6R seed 42, 123 của A2. Arm chính: E4 seed 42.

## 4. Phân tích G: đoàn hệ đích thứ hai (khóa)

**Sleep-EDF Telemetry** (PhysioNet, `sleep-edfx/1.0.0/sleep-telemetry`): toàn bộ 44 cặp PSG/hypnogram của 22
đối tượng, cả hai đêm (giả dược và temazepam); kiểm tra SHA-256 theo `SHA256SUMS.txt` của PhysioNet. Kênh
`EEG Fpz-Cz`. Tiền xử lý bằng đúng hàm của Sleep-EDF SC (`sleeptcn.preprocessing.process_record`, cấu hình
mặc định): cửa sổ đánh giá, ánh xạ nhãn và các biến thể `paper_raw_v1`, `bandpass_v2`, `filtered_v2`,
`filtered_zscore_v2` không đổi. Đối tượng là năm ký tự đầu của khóa bản ghi. Telemetry cùng chuyển đạo và
cùng nhóm nghiên cứu với dữ liệu huấn luyện nhưng khác đối tượng, khác thiết bị ghi và khác quần thể (người
khó vào giấc nhẹ); nó được báo cáo như một phép thử trung gian, không phải một đoàn hệ độc lập hoàn toàn.

**ISRUC nhóm I** (100 bệnh nhân): máy chủ dữ liệu không truy cập được tại thời điểm khóa (dữ liệu hiện
phát hành qua MEGA, 14,12 GB). Nếu lấy được bản ghi thô: kênh C4-A1 từ tệp `.rec` (không dùng tệp
ExtractedChannels đã được nhà cung cấp lọc sẵn), lấy mẫu lại 200→100 Hz đa pha (up 1, down 2, Kaiser
β = 5,0) trên tín hiệu liên tục, nhãn của người chấm 1 (0 W, 1 N1, 2 N2, 3 N3, 5 REM; giá trị khác bị
loại), cùng cửa sổ đánh giá và cùng các biến thể. Chi tiết đọc tệp và kiểm tra đơn vị được chốt trước khi
so bất kỳ nhãn nào với dự đoán; mọi thay đổi so với mô tả này được báo cáo là sai lệch.

**Suy luận:** như giao thức gốc: mười mô hình fold của mỗi arm, xác suất softmax lấy trung bình ở float64
rồi argmax, không cập nhật trọng số, checkpoint đã khóa (`runs/holdout_ckpt_restore_20261010`), CPU.

**Đối chiếu trên đoàn hệ mới** (macro-F1 trung bình theo đối tượng, từng seed): E3−E0, E3−E6, E3−E2, E4−E0,
E4−E2, E4−E3, E4−E6; cùng tỷ lệ N3→N2 và độ thu hồi N3 của mọi arm.

## 5. Chỉ số và khoảng tin cậy

Macro-F1 trung bình theo đối tượng (chỉ số chính), macro-F1 gộp, accuracy, κ, precision/recall/F1 từng lớp,
tỷ lệ N3→N2. Khoảng tin cậy: bootstrap cụm theo đối tượng có ghép cặp, phân vị, 10.000 lần lấy mẫu lại,
seed 2044. Trên Telemetry và Sleep-EDF, cụm là đối tượng (cả hai đêm).

## 6. Quy tắc diễn giải (khóa)

- Một đối chiếu **lặp lại** trên đoàn hệ mới nếu khoảng tin cậy ở seed 42 nằm hoàn toàn trên không; seed
  123 được báo cáo bên cạnh.
- Quy tắc hiệu chỉnh **cải thiện** nếu khoảng của (có quy tắc − không quy tắc) về macro-F1 trung bình theo
  đối tượng nằm trên không; là **đánh đổi** nếu khoảng của độ thu hồi N3 nằm trên không nhưng khoảng của
  macro-F1 chứa không; **gây hại** nếu khoảng của macro-F1 nằm dưới không.
- Cả hai quy tắc, mọi arm, mọi đoàn hệ và chi phí trên miền nguồn đều được báo cáo bất kể kết quả.

## 7. Chương trình

| Bước | Chương trình |
|---|---|
| Tải và tiền xử lý Telemetry | `scripts/a3_prepare_telemetry.py` |
| Suy luận (Telemetry; dự đoán ngoài fold của Sleep-EDF) | `scripts/a3_infer.py` |
| Phân tích F và G | `scripts/a3_analyze.py` |

Kết quả tóm tắt: `runs/a3/analysis/a3_summary.json` (ghi một lần). Lỗi lập trình phát hiện sau khi khóa
được sửa và ghi vào mục sai lệch của báo cáo; không lựa chọn phân tích nào được đổi theo kết quả.
