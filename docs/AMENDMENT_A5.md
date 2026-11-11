# Phụ lục A5: phát hiện bỏ sót N3 không cần nhãn và xác nhận trên ISRUC

Tệp cấu hình khóa: `configs/amendment_a5.json`. A5 không thay đổi, kiểm định lại hay diễn giải lại H1–H3 và
các báo cáo của phân tích gốc, A1–A4. A5 là phân tích mô tả, không có kiểm định giả thuyết.

## 1. Câu hỏi

Có thể phát hiện, không cần nhãn, khi một mô hình đang bỏ sót N3 trên dữ liệu mới, và chỉ sửa ở những bản
ghi đó, mà không làm hỏng những nơi không có lỗi không?

## 2. Dữ liệu phát triển và dữ liệu đánh giá

Thống kê và cổng được phát triển ngày 11/11/2026, sau khi đã xem mẫu SHHS1 giữ lại, Telemetry và dự đoán
ngoài fold trên Sleep-EDF. Các đoàn hệ này là **dữ liệu phát triển**, không phải bằng chứng. Kết quả thăm dò
trên đó (chỉ để giải thích vì sao A5 tồn tại):

- Bên trong SHHS1, thống kê biên độ tương quan với độ thu hồi N3 của từng bản ghi ở mức Spearman −0,92, còn
  entropy trung bình chỉ −0,12; mô hình sai một cách tự tin.
- Quy tắc có cổng: E4 seed 42 cho +0,038 trên SHHS1, −0,001 trên Sleep-EDF và −0,010 trên Telemetry, trong
  khi quy tắc không cổng cho −0,077 trên Telemetry.

**ISRUC nhóm I là đoàn hệ đánh giá duy nhất**; nó chưa từng được tải hay xem trước khi khóa.

## 3. ISRUC nhóm I (khóa)

100 người. Dùng tệp ghi thô `N/N.rec` (EDF), không dùng tệp ExtractedChannels đã được nhà cung cấp lọc sẵn.
Hai kênh trên **cùng người và cùng người chấm**: C4-A1 (cùng chuyển đạo với SHHS1) và F4-A1 (vùng trán, gần
Fpz-Cz của dữ liệu huấn luyện hơn). Lấy mẫu lại 200→100 Hz đa pha (up 1, down 2, Kaiser β = 5,0) trên tín hiệu
liên tục; bỏ epoch cuối chưa đủ 30 giây; cửa sổ đánh giá và các biến thể giống hệt Sleep-EDF và SHHS1. Nhãn
của người chấm 1 là chính, người chấm 2 là phụ; 0 W, 1 N1, 2 N2, 3 và 4 N3, 5 REM, giá trị khác bị loại. Nhãn
dài hơn tín hiệu thì cắt phần đuôi; ngắn hơn thì phần tín hiệu còn lại đánh dấu không xác định. Một người chỉ bị
loại khi thiếu kênh, đơn vị không phải µV hoặc không căn được nhãn; danh sách loại được chốt và báo cáo trước
khi so bất kỳ nhãn nào với dự đoán. Chi tiết đọc tệp được kiểm thử trên một người trước khi chạy toàn bộ; mọi
thay đổi so với mô tả này được ghi là sai lệch.

## 4. Suy luận

Tám arm lưu trữ: E0, E2, E3, E4 ở seed 42 và 123; arm chính E4 seed 42. Ensemble mười mô hình fold, không cập
nhật trọng số. Thiết bị nào cũng được, nhưng trước khi chạy ISRUC, thiết bị đó phải tái tạo được xác suất
ensemble đã lưu trên Telemetry của E0 và E4 seed 42 cho hai bản ghi (sai số dưới 1e-4, argmax giống hệt).

## 5. Quy tắc (khóa)

- **Thống kê bất đồng biên độ** của một bản ghi: trong các epoch có độ lệch chuẩn đầu vào thông dải từ ngưỡng
  Sleep-EDF trở lên (16,513 µV) và được ensemble đoán là N2 hoặc N3, tỷ lệ bị đoán là N2. Không xác định (không
  áp quy tắc) khi có dưới 20 epoch như vậy.
- **Cổng:** với mỗi arm, phân vị 95 của thống kê trên 153 bản ghi Sleep-EDF chấm bằng dự đoán ngoài fold của
  arm đó. Chỉ dùng dữ liệu nguồn.
- **Sửa:** ở bản ghi vượt cổng, mọi epoch được đoán N2 có độ lệch chuẩn từ ngưỡng trở lên được đổi thành N3;
  các bản ghi khác giữ nguyên.
- **Đối chứng:** không sửa; quy tắc A3 không cổng; cùng cách sửa nhưng cổng bằng entropy trung bình (phân vị 95
  trên Sleep-EDF ngoài fold).
- Không có tham số nào được chọn trên ISRUC.

## 6. Phân tích và quy tắc diễn giải (khóa)

Bootstrap cụm theo người có ghép cặp, 10.000 lần, seed 2047. Chỉ số chính: macro-F1 trung bình theo người.

- **Quy tắc được ủng hộ** nếu trên ISRUC C4 với arm chính, (có cổng − không sửa) có khoảng tin cậy trên không,
  và cận dưới của cùng hiệu số trên ISRUC F4 và trên Sleep-EDF ngoài fold đều lớn hơn −0,01.
- **Không được ủng hộ** nếu trên ISRUC C4 khoảng đó chứa không hoặc dưới không trong khi độ thu hồi N3 khi không
  sửa dưới 0,5.
- **Chưa được kiểm tra** nếu độ thu hồi N3 khi không sửa trên ISRUC C4 từ 0,7 trở lên (ISRUC không có lỗi bỏ sót).
- **So với cổng entropy:** báo cáo hiệu số và khoảng tin cậy; chỉ nói "tốt hơn" khi khoảng nằm trên không.
- **Giải thích theo chuyển đạo:** lỗi N3 được mô tả là liên quan đến chuyển đạo nếu (F4 − C4) về độ thu hồi N3
  có khoảng trên không cho E3 và E4 ở cả hai seed.
- **Chuyển quần thể:** E3−E0, E4−E0, E4−E2, E3−E2 trên từng kênh; một đối chiếu lặp lại nếu khoảng ở seed 42
  nằm trên không.
- Mọi arm, kênh, đối chứng và cả hai người chấm đều được báo cáo bất kể kết quả.

## 7. Sai lệch D1 (12/11/2026): ISRUC nhóm I không dùng được cho A5

Trước khi chạy toàn bộ, bộ đọc được thử trên hai người thật (số 1 và 26). Bộ đọc hoạt động đúng (kênh C4-A1 và
F4-A1, 200 Hz, số nhãn khớp số epoch), nhưng dữ liệu thô có ba vấn đề làm A5 không thực hiện được như đã khóa.
Chưa có dự đoán nào được tính trên ISRUC.

| | Người 1 | Người 26 | Tham chiếu Sleep-EDF / SHHS1 |
|---|---|---|---|
| Header EEG | ±25, ghi là µV | ±25, ghi là µV | |
| Độ lệch chuẩn trung vị epoch N3 sau lọc 0,5–30 Hz, C4-A1 | 2,1 | 1,2 | 20,8 / 18,8 µV |
| Biên độ đỉnh–đỉnh trung vị epoch N3, C4-A1 | 15,4 | 8,5 | quy tắc chấm N3 yêu cầu ≥ 75 µV |
| Bộ lọc phần cứng ghi trong header | 0,3–35 Hz, notch 50 Hz | 1–35 Hz, notch 50 Hz | |

1. **Thang biên độ sai so với header:** epoch được chấm N3 có biên độ nhỏ hơn mức tối thiểu của quy tắc chấm
   điểm khoảng 5–9 lần, nên giá trị trong tệp không phải µV.
2. **Thang khác nhau giữa các người:** biên độ N3 của người 26 bằng khoảng nửa người 1, nên không có hệ số chung
   nào khôi phục được µV (và chọn hệ số sau khi xem dữ liệu là điều chỉnh hậu nghiệm).
3. **Bộ lọc phần cứng khác nhau và đã áp dụng khi ghi:** người 26 bị lọc thông cao ở 1 Hz, loại phần lớn sóng
   chậm 0,5–1 Hz; đầu vào "không lọc" của ISRUC vì vậy không phải tín hiệu chưa lọc.

Quy tắc của A5 dựa trên biên độ tuyệt đối, và các arm E0–E4 nhận tín hiệu µV thô; cả hai đều không có nghĩa trên
dữ liệu này. Vì vậy phần đánh giá trên ISRUC của A5 **không được thực hiện**. Quy tắc có cổng chỉ được báo cáo như
một phân tích chẩn đoán trên dữ liệu phát triển (mục 2), không có xác nhận độc lập. Nhận định này áp dụng cho câu
hỏi của A5 (biên độ tuyệt đối, so sánh lọc); các mô hình chuẩn hóa theo từng bản ghi vẫn có thể dùng ISRUC.
