# Kết quả zero-shot SHHS1 đã khóa

Ngày hoàn tất: **2026-08-15**
Trạng thái: **SUY LUẬN, KIỂM ĐỊNH VÀ PHÂN TÍCH BẮT CẶP ĐÃ ĐẠT**

## Câu hỏi và phạm vi

Chiến dịch đánh giá trực tiếp các checkpoint đã huấn luyện trên Sleep-EDF Expanded trên mẫu SHHS
Visit 1 đã khóa, không cập nhật trọng số bằng SHHS. Đây là đánh giá zero-shot đồng thời chịu dịch
chuyển quần thể, thiết bị và montage EEG; nó không phải xác nhận lâm sàng và không đại diện cho toàn bộ
SHHS.

Ba cấu hình được đánh giá:

- E0: `paper_raw_v1`, 15 CNN và BiLSTM;
- E3: `filtered_v2`, ResNet-1D và TCN;
- E6: `filtered_zscore_v2`, ResNet-1D và TCN.

Mỗi cấu hình dùng đúng `best.pt` của từng outer fold Sleep-EDF. Không chọn một fold tốt nhất và không
trộn thành phần giữa fold. Xác suất softmax của 10 fold được lấy trung bình số học bằng bộ tích lũy
float64 trước `argmax`.

## Toàn vẹn thực nghiệm

- Validation: 15 đối tượng, 450 dự đoán theo fold và 45 tổ hợp; cổng độc lập đạt, 0 lỗi.
- Test: 180 đối tượng, 169.012 epoch hợp lệ, 5.400 dự đoán theo fold và 540 tổ hợp; cổng độc lập đạt,
  0 lỗi.
- Thiết bị: CPU, PyTorch 2.5.1, 12 luồng, batch 256.
- Thời gian test: 6.692,6 giây, khoảng 111,5 phút.
- Test chỉ được mở sau khi validation gate đạt và chỉ chạy một chiến dịch xác nhận.

## Chỉ số mô tả

| Cấu hình | Macro-F1 trung bình theo đối tượng (CI 95%) | Macro-F1 gộp | Accuracy | Kappa | F1 N1 |
|---|---:|---:|---:|---:|---:|
| E0 | 0,5268 [0,5101; 0,5431] | 0,5761 | 0,6648 | 0,5339 | 0,2998 |
| E3 | **0,5680 [0,5503; 0,5850]** | **0,6099** | **0,7016** | **0,5801** | **0,3451** |
| E6 | 0,5407 [0,5227; 0,5584] | 0,5732 | 0,6742 | 0,5408 | 0,3102 |

Chỉ số chính đã định trước là trung bình Macro-F1 theo đối tượng. Chỉ số gộp theo epoch và từng lớp là
thứ cấp vì đối tượng có số epoch khác nhau.

## Phân bố nhãn của mẫu test

Phân bố dưới đây được tính trên 169.012 epoch hợp lệ của mẫu SHHS1 đã khóa. N2 là lớp chiếm tỷ trọng
lớn nhất, trong khi N1 là lớp ít xuất hiện nhất.

| Nhãn | Số epoch | Tỷ lệ |
|---|---:|---:|
| W | 36.983 | 21,88% |
| N1 | 7.002 | 4,14% |
| N2 | 76.287 | 45,14% |
| N3 | 22.806 | 13,49% |
| REM | 25.934 | 15,34% |
| **Tổng hợp lệ** | **169.012** | **100,00%** |

## So sánh xác nhận bắt cặp

Bootstrap lấy lại mẫu toàn bộ 180 đối tượng theo cặp 10.000 lần, seed 2030. Wilcoxon hạng có dấu hai
phía dùng Macro-F1 từng đối tượng; Holm hiệu chỉnh đúng hai so sánh đã khóa E3-E0 và E3-E6.

| So sánh | Chênh lệch chính | CI 95% | Trung vị chênh lệch | Thắng/Hòa/Thua | p Holm |
|---|---:|---:|---:|---:|---:|
| E3-E0 | **0,0412** | **[0,0314; 0,0512]** | 0,0372 | 138/0/42 | 2,65e-13 |
| E3-E6 | **0,0274** | **[0,0182; 0,0370]** | 0,0292 | 125/0/55 | 1,87e-08 |

Hai khoảng tin cậy hoàn toàn dương và hai kiểm định vẫn có ý nghĩa sau Holm. Vì vậy, trong mẫu SHHS1
đã khóa và giao thức zero-shot này, E3 có Macro-F1 trung bình theo đối tượng cao hơn E0 và E6.

Phân tích thứ cấp nhất quán với kết luận chính:

- E3-E0: Macro-F1 gộp +0,0338, CI [0,0230; 0,0445]; accuracy +0,0368; kappa +0,0461.
- E3-E6: Macro-F1 gộp +0,0367, CI [0,0266; 0,0469]; accuracy +0,0274; kappa +0,0392.
- Tại vùng ±1 epoch quanh chuyển pha thật liên tiếp, E3-E0 tăng Macro-F1 0,0366 và E3-E6 tăng
  0,0183; cả hai CI bootstrap hỗ trợ đều dương.

## Kết luận được phép

- E3 tổng quát sang mẫu SHHS1 đã khóa tốt hơn E0 và E6 theo tiêu chí chính đã định trước.
- E6 không cải thiện E3; dữ liệu hiện tại không ủng hộ z-score từng bản ghi như một cách tăng hiệu năng
  zero-shot so với cách xử lý E3.
- E3 cải thiện F1 N1 so với E0 và E6, nhưng N1 vẫn là lớp yếu nhất; E0 có recall N1 cao hơn E3 nên
  đánh đổi precision-recall phải được nêu rõ.

## Chẩn đoán lỗi theo lớp của E3 và E2

Phân tích hậu nghiệm E3-E2 sử dụng cùng 169.012 epoch hợp lệ. Các số liệu sau đây là mô tả gộp,
không phải một họ kiểm định mới.

| Lớp | E3 precision | E3 recall | E3 F1 | E2 precision | E2 recall | E2 F1 | ΔF1 (E3-E2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| W | 0,882267 | 0,801395 | 0,839889 | 0,878039 | 0,727469 | 0,795694 | +0,044195 |
| N1 | 0,252822 | 0,543702 | 0,345150 | 0,236726 | 0,352756 | 0,283322 | +0,061828 |
| N2 | 0,739422 | 0,734896 | 0,737152 | 0,736197 | 0,693735 | 0,714336 | +0,022816 |
| N3 | 0,944044 | 0,258178 | 0,405468 | 0,965898 | 0,249627 | 0,396725 | +0,008743 |
| REM | 0,605237 | 0,893923 | 0,721785 | 0,488616 | 0,944976 | 0,644158 | +0,077626 |

E3 tăng F1 ở cả năm lớp; mức tăng lớn nhất nằm ở REM và N1. Mẫu hình này đi kèm đánh đổi giữa
precision và recall: recall N3 vẫn thấp, còn precision REM còn hạn chế. Các cặp nhầm lẫn lớn nhất của
E3 là N3→N2 (16.674), N2→REM (11.605), N2→N1 (5.947), W→N1 (4.481) và N1→REM (1.717) epoch.
Đối chiếu E0 cho thấy N3→N2 không phải lỗi riêng của E3: E0 có 16.480/22.806 epoch N3 bị gán N2
(72,3%) và recall N3 0,2610, gần như E3 với 73,1% và recall 0,2582. Vì vậy, E3 cải thiện hiệu năng
tổng thể nhưng không cải thiện nhận diện N3 so với 15-CNN--BiLSTM. Ma trận E3--E2 đầy đủ được giữ
trong `Reports/SHHS_E3_E2_PAIRED_AUDIT.json`; đối chiếu E0/E3/E6 được giữ trong
`Reports/SHHS_E0_E3_E6_N3_AUDIT.json`.

## Điều không được suy ra

- Không quy chênh lệch E3-E0 riêng cho ResNet, TCN hoặc tiền xử lý vì ba thành phần thay đổi đồng thời.
- Không kết luận chuẩn hóa gây ra khác biệt sinh lý hoặc luôn có hại trên mọi miền.
- Không tuyên bố tương đương/không thua kém, xác nhận lâm sàng hay tổng quát cho toàn bộ hơn 6.000 bản
  ghi SHHS.
- Không dùng 10 fold như 10 mẫu độc lập; đơn vị suy luận là 180 đối tượng SHHS.

## Nguồn chân lý

- Test gate SHA-256: `51828329b2ebb2d99e5d71d6b9c78fd5a3fad037162fa50855af52066e4d2646`.
- Test manifest SHA-256: `f9cd5ebbd20f26b188b5dc13ac6e417ff8ef0fa8dcae78760cfcb27940bf58cf`.
- Phân tích bắt cặp SHA-256: `83aa53fed3dc7be9b6f14cb63ddbd7417a7af256b9f308383500ee6e068943df`.
- Artifact chứa ID và dự đoán nằm ngoài Git tại `E:\research\Dataset\SHHS_v1\zero_shot_v1`.

## Phân tích thành phần bổ sung E1/E2

Sau khi chiến dịch v1 đã đóng, một giao thức riêng được khóa trước lần suy luận E1/E2 trên cùng 180
đối tượng test. E1--E0 đạt chênh lệch Macro-F1 theo đối tượng `+0,00651`, CI 95%
`[+0,00192; +0,01087]`, p Holm `0,00325`; E2--E1 có hướng ngược lại `-0,01280`, CI
`[-0,02209; -0,00350]`, p Holm `0,01005`. Vì cohort đã được mở cho E0/E3/E6, đây là bằng chứng
ngoài miền thứ cấp chứ không phải cohort xác nhận độc lập. Xem `SHHS_COMPONENT_EXTENSION_RESULTS.md`.
