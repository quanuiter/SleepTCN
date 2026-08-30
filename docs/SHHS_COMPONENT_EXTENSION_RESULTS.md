# Phân tích bổ sung thành phần E1/E2 trên SHHS1

Ngày hoàn tất: **2026-08-15**
Trạng thái: **SUY LUẬN, KIỂM ĐỊNH ARTIFACT VÀ PHÂN TÍCH BẮT CẶP ĐÃ ĐẠT**

## Mục tiêu và địa vị bằng chứng

Phân tích này được khóa trước lần suy luận SHHS đầu tiên của E1/E2 để trả lời hai câu hỏi:

1. E1--E0: thay BiLSTM bằng TCN trong quy trình 15-CNN có cải thiện Macro-F1 không?
2. E2--E1: thay bộ trích đặc trưng 15-CNN bằng ResNet-1D dưới cùng họ TCN có cải thiện Macro-F1 không?

E0 được tái sử dụng nguyên trạng từ chiến dịch zero-shot v1. E1 và E2 dùng `paper_raw_v1`, đủ 10
checkpoint theo fold và trung bình xác suất; không cập nhật trọng số bằng SHHS.

Đây là **phân tích ngoại miền thứ cấp được khóa trước suy luận E1/E2**, không phải tái lập độc lập trên một
cohort SHHS chưa từng mở: 180 đối tượng test này đã được dùng trước đó cho E0/E3/E6. Giới hạn này phải
được giữ trong mọi bản thảo.

## Toàn vẹn thực nghiệm

- Inventory: 20 bộ thí nghiệm--fold, 180 tham chiếu checkpoint, 180 hash duy nhất; trạng thái `PASSED`.
- Validation kỹ thuật: 15 đối tượng, 300 dự đoán fold, 30 ensemble; 0 lỗi.
- Test: 180 đối tượng, 169.012 epoch hợp lệ, 3.600 dự đoán fold, 360 ensemble; 0 lỗi.
- CPU: PyTorch 2.5.1, 12 luồng, batch 256; thời gian test 5.681,1 giây, khoảng 94,7 phút.
- Phân tích: bootstrap cụm bắt cặp theo 180 đối tượng, 10.000 lần, seed 2031; Wilcoxon hai phía và Holm
  trên đúng hai so sánh chính.
- Phân tích được chạy lại byte-giống-hệt.

## Kết quả

| E | Macro-F1 trung bình theo đối tượng | Macro-F1 gộp | Accuracy | Kappa | F1 N1 | Macro-F1 chuyển pha |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 0,5268 | 0,5761 | 0,6648 | 0,5339 | 0,2998 | 0,4323 |
| E1 | **0,5333** | **0,5822** | **0,6709** | **0,5409** | **0,3197** | **0,4352** |
| E2 | 0,5205 | 0,5668 | 0,6656 | 0,5327 | 0,2833 | 0,4219 |

| So sánh | Chênh lệch chính | CI 95% | Trung vị | Thắng/Hòa/Thua | p Holm | Quy tắc khóa trước |
|---|---:|---:|---:|---:|---:|---|
| E1--E0 | **+0,00651** | **[+0,00192; +0,01087]** | +0,00648 | 108/0/72 | 0,00325 | Đạt |
| E2--E1 | **-0,01280** | **[-0,02209; -0,00350]** | -0,00429 | 80/0/100 | 0,01005 | Không đạt giả thuyết E2 tốt hơn |

Phân tích thứ cấp phù hợp với kết luận chính:

- E1--E0 có chênh lệch Macro-F1 gộp +0,00607, CI `[+0,00115; +0,01086]`. Ở vùng chuyển pha,
  chênh lệch +0,00288 có CI `[-0,00297; +0,00850]`, nên chưa chứng minh lợi ích riêng tại chuyển pha.
- E2--E1 có chênh lệch Macro-F1 gộp -0,01533, CI `[-0,02527; -0,00518]`. Ở vùng chuyển pha,
  chênh lệch -0,01330, CI `[-0,02186; -0,00453]`.

## Kết luận được phép

- Trên mẫu 180 đối tượng SHHS1 cố định này, E1 có Macro-F1 trung bình theo đối tượng cao hơn E0 theo
  quy tắc đã khóa. Đây là bằng chứng ngoại miền thứ cấp ủng hộ TCN thay BiLSTM trong quy trình 15-CNN;
  hiệu ứng nhỏ, không phải ưu thế phổ quát.
- Giả thuyết E2 tốt hơn E1 không được ủng hộ. Trong mẫu này, chênh lệch có hướng ngược lại và nhất quán
  ở tiêu chí chính, Macro-F1 gộp và vùng chuyển pha.
- Vì E3 vẫn cao hơn E0 trong chiến dịch zero-shot v1, kết quả mới cho thấy không thể giải thích lợi ích
  của toàn bộ quy trình E3 bằng việc thay 15-CNN bằng ResNet-1D đơn thuần.

## Điều không được suy ra

- Không gọi đây là tái lập độc lập trên cohort chưa mở hoặc xác nhận lâm sàng.
- Không nói TCN luôn tốt hơn BiLSTM hoặc ResNet luôn kém hơn 15-CNN trên mọi dữ liệu.
- Không quy E2--E1 cho riêng một khái niệm trừu tượng về ``chất lượng đặc trưng'': hai quy trình khác kích
  thước biểu diễn, tham số giao diện TCN và quá trình huấn luyện extractor.
- Không dùng kết quả này để tuyên bố tương đương/không thua kém hay kết luận P/N vô dụng.
- Không nhập hai p-value mới vào họ Holm lịch sử của Sleep-EDF hoặc zero-shot v1.

## Nguồn chân lý

- Giao thức: `configs/shhs_component_extension_v1.json`.
- Inventory SHA-256: `0f33782e6ff2ed0455695c97bca3cc1f8a9e881ba97e38d413a7deaf204d1e3a`.
- Test manifest SHA-256: `064a7c1ca586e0f19bfa753017edd5ba2a5be1b8476b9767e69ce46d8e12f174`.
- Test gate SHA-256: `fbc4080f4e25625382c1658e7ee25bc25ec23588b09e88e33e2ac3ab1596228c`.
- Phân tích SHA-256: `39ad18082eadc263b479e6badfcf87149cae16d0267cad050a026ab8d949a74c`.
- Artifact chứa dữ liệu chịu điều kiện NSRR nằm ngoài Git tại
  `E:\research\Dataset\SHHS_v1\zero_shot_components_v1`.
