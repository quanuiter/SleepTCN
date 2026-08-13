# Kết quả Gate 5 — phân tích thống kê bắt cặp trên test

Ngày thực hiện: **2026-08-14**.

## Trạng thái kiểm định

- Gate 4 đã đạt: 60/60 run có prediction và metrics test đã kiểm định.
- Sáu thí nghiệm E0, E1, E2, E3, E4 và E6 đều phủ đúng 78 đối tượng, 153 bản ghi và 195.469 epoch hợp lệ.
- Khóa ghép cặp `(subject_id, record_key, original_epoch_index)` và nhãn thật khớp tuyệt đối giữa các mô hình.
- Bootstrap bắt cặp theo cụm đối tượng, 10.000 lần, seed 2026.
- Wilcoxon signed-rank hai phía trên Macro-F1 từng đối tượng.
- Holm chỉ áp dụng cho đúng bốn so sánh chính đã đăng ký trước. E4−E2 được tách riêng là phân tích cơ chế thứ cấp.
- Báo cáo chạy lại có SHA-256 giống hệt lần đầu:
  `40eb9fa3f6ec3c775b8838304647a27dd1743d1e22f059ce07f8397bf301fc3c`.

Báo cáo máy đọc: `runs/v2/analysis/gate5_paired_results_seed42.json`.

## Hiệu năng gộp out-of-fold

| Thí nghiệm | Macro-F1 | Accuracy | Cohen's kappa | F1 N1 |
|---|---:|---:|---:|---:|
| E0 | 0,775419 | 0,826233 | 0,761431 | 0,500915 |
| E1 | 0,780230 | 0,831482 | 0,768301 | 0,511469 |
| E2 | 0,783481 | 0,834061 | 0,771642 | 0,518029 |
| E3 | 0,790443 | 0,836266 | 0,775381 | 0,527597 |
| E4 | 0,789067 | 0,835954 | 0,774558 | 0,527007 |
| E6 | 0,769124 | 0,822969 | 0,757320 | 0,512397 |

Đây là chỉ số mô tả trên toàn bộ dự đoán test out-of-fold. Không dùng riêng thứ hạng bảng này để kết luận
có ý nghĩa thống kê.

## Bốn so sánh chính

| So sánh | Δ Macro-F1 | CI 95% bootstrap | p Wilcoxon | p Holm | Thắng/Hòa/Thua theo đối tượng | Kết luận |
|---|---:|---:|---:|---:|---:|---|
| E1−E0 | +0,004811 | [−0,000915; +0,010193] | 0,034064 | 0,102193 | 51/0/27 | Chưa đủ bằng chứng sau hiệu chỉnh Holm |
| E2−E1 | +0,003251 | [−0,002370; +0,008808] | 0,051777 | 0,103554 | 44/0/34 | Chưa đủ bằng chứng |
| E3−E2 | +0,006962 | [+0,000305; +0,014520] | 0,898933 | 0,898933 | 37/0/41 | Hiệu ứng gộp dương nhưng không đồng đều theo đối tượng |
| E3−E6 | +0,021319 | [+0,012179; +0,030698] | 0,000296 | 0,001185 | 49/0/29 | Có bằng chứng nhất quán theo cả hai phân tích |

Δ được tính theo chiều mô hình đứng trước trừ mô hình đứng sau.

### Diễn giải chính xác

1. **E1−E0 — thay BiLSTM bằng TCN khi giữ 15CNN:** Macro-F1 tăng khoảng 0,0048, nhưng CI bootstrap
   chứa 0 và p sau Holm là 0,1022. Không được tuyên bố TCN tốt hơn có ý nghĩa thống kê trong thí nghiệm
   seed 42 này.
2. **E2−E1 — thay 15CNN bằng ResNet-1D khi giữ TCN:** Macro-F1 tăng khoảng 0,0033; CI chứa 0 và
   p Holm là 0,1036. Kết quả mô tả tốt hơn nhẹ nhưng chưa đủ bằng chứng thống kê.
3. **E3−E2 — gói tiền xử lý hoàn chỉnh so với raw:** Macro-F1 gộp tăng khoảng 0,0070 và CI bootstrap
   vừa vượt 0. Tuy nhiên median chênh lệch Macro-F1 theo người là −0,00118, chỉ 37/78 người tăng và
   Wilcoxon p=0,8989. Điều này cho thấy lợi ích gộp có thể tập trung ở một số đối tượng/lớp hoặc đối tượng
   có nhiều epoch, không phải cải thiện đồng đều. Phải báo cáo sự bất đồng này, không chỉ chọn CI có lợi.
4. **E3−E6 — chia hằng số bảo toàn quan hệ biên độ so với z-score theo bản ghi:** Macro-F1 tăng khoảng
   0,0213; CI hoàn toàn dương và p Holm=0,00119. Đây là kết quả chính mạnh nhất. Chênh lệch F1 lớn nhất
   nằm ở N3 (+0,06278), sau đó N1 (+0,01520), REM (+0,01311), N2 (+0,01295) và W (+0,00255).

## Phân tích cơ chế thứ cấp

E4−E2 (riêng lọc dải so với raw) cho Δ Macro-F1 **+0,005586**, CI 95%
**[−0,001699; +0,013618]**, Wilcoxon p=**0,902877**, thắng/hòa/thua **38/0/40**. P-value này không
tham gia nhóm Holm chính. Chưa có bằng chứng rằng riêng lọc dải tạo ra cải thiện đồng đều theo đối tượng.

Không chạy E5−E4 vì dữ liệu E5 và E4 trùng bitwise (`clip_fraction=0`); tạo p-value cho hai điều kiện
trùng lặp không có ý nghĩa khoa học.

## Điều được phép và không được phép kết luận

Được phép kết luận:

- E3 có hiệu năng mô tả cao nhất trong sáu cấu hình trên Sleep-EDF Expanded.
- Chia hằng số bảo toàn quan hệ biên độ (E3) tốt hơn z-score theo bản ghi (E6) trong giao thức hiện tại.
- Thay BiLSTM bằng TCN và thay 15CNN bằng ResNet-1D cho mức tăng mô tả nhỏ, nhưng chưa có bằng chứng
  thống kê sau hiệu chỉnh đa kiểm định.
- Tác động của gói tiền xử lý E3 không đồng đều theo đối tượng và cần được trình bày thận trọng.

Không được kết luận:

- Không khẳng định quan hệ nhân quả từ các ablation này.
- Không tuyên bố đã giải quyết domain shift, zero-shot SHHS hoặc giá trị lâm sàng.
- Không coi 10 fold là 10 mẫu độc lập.
- Không khẳng định độ ổn định theo random seed vì huấn luyện chính thức mới dùng seed 42.
- Không gọi ResNet-1D + TCN là tiết kiệm tham số trước khi benchmark độ phức tạp được hoàn tất.

## Bước tiếp theo

Gate 5 đã hoàn tất. Công việc tiếp theo là Gate 6:

1. benchmark có kiểm soát trên cùng một GPU: tham số, độ trễ trung vị/p95, throughput và peak VRAM;
2. phân tích không gian đặc trưng với quy tắc lấy mẫu được định trước, t-SNE/UMAP để mô tả và
   Silhouette Score để hỗ trợ định lượng;
3. tạo bảng và hình cho khóa luận từ báo cáo JSON đã khóa;
4. nếu ngân sách cho phép, đăng ký trước và chạy thêm seed 123/2025 như thí nghiệm độ ổn định riêng,
   không trộn với kết quả seed 42 hiện tại.
