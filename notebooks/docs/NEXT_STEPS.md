# Việc cần làm tiếp theo

Nguồn bằng chứng: `VALIDATION_AUDIT_10FOLD.md`.

Giao thức thống kê: `STATISTICAL_ANALYSIS.md`.

## Cổng 1 — dữ liệu và split — ĐẠT

- 765/765 NPZ đã được kiểm định.
- Split 10-fold seed 42 theo đối tượng đã khóa.
- E4/E5 giống bitwise; E5 bị loại khỏi fold 01--09.

Không tiền xử lý lại và không sửa manifest/split.

## Cổng 2 — smoke CPU/GPU — ĐẠT

- Smoke CPU/GPU E0--E6 fold 00 đã đạt.
- Trình chạy, checkpoint, resume và kiểm định artifact đã qua cổng kỹ thuật.

## Cổng 3 — full validation-only 10-fold — ĐẠT

- Đủ 60/60 run của E0, E1, E2, E3, E4 và E6 trên fold 00--09, seed 42.
- Tất cả validation report đạt; test vẫn khóa.
- Đủ và kiểm tra SHA-256 của checkpoint.
- Không chạy thêm fold, không thay siêu tham số và không chọn lại mô hình.

## Cổng 4 — khóa checkpoint và mở test một lần — BƯỚC KẾ TIẾP

Trước khi thuê GPU hoặc chạy test:

1. Ghi nhận commit khóa chứa báo cáo audit và tài liệu trạng thái cập nhật.
2. Tạo runbook test-unlock riêng; không tái sử dụng nguyên vòng lặp validation.
3. Kiểm tra cách trình chạy cập nhật manifest để worktree không bị bẩn và chặn run kế tiếp.
4. Chạy thử logic kiểm tra trên bản sao/fixture không chứa test prediction thật nếu cần.
5. Chốt danh sách 60 run và bốn so sánh chính trước khi thấy test.
6. Xác nhận không còn quyết định mô hình, preprocessing, seed hoặc thống kê chưa được khóa.

Khi các điều kiện trên đạt, chạy từng run từ checkpoint đã chọn bằng:

```text
--resume --allow-test-evaluation
```

Mục tiêu của lần chạy này chỉ là sinh prediction/metrics test; không huấn luyện hoặc chọn
checkpoint lại. Sau test đầu tiên, không sửa code, config, split, dữ liệu, seed, checkpoint hoặc
giả thuyết dựa trên kết quả test.

## Cổng 5 — phân tích thống kê

Sau khi đủ test prediction của 60 run:

1. Ghép prediction out-of-fold; mỗi đối tượng chỉ lấy từ fold mà họ là test.
2. Bắt buộc khớp `(subject_id, record_key, original_epoch_index)` và nhãn thật giữa hai mô hình.
3. Chạy bốn so sánh chính đã định trước:
   - E1 − E0: TCN so với BiLSTM.
   - E2 − E1: ResNet-1D so với 15CNN khi dùng chung TCN.
   - E3 − E2: gói tiền xử lý hoàn chỉnh so với raw.
   - E3 − E6: chia hằng số bảo toàn biên độ so với z-score.
4. Bootstrap ghép cặp theo cụm đối tượng 10.000 lần để lấy hiệu ứng và CI 95%.
5. Wilcoxon signed-rank hai phía trên Macro-F1 từng đối tượng.
6. Hiệu chỉnh Holm cho bốn p-value chính.
7. Báo cáo thắng/hòa/thua theo đối tượng và chỉ số từng lớp, đặc biệt N1.

E4 − E2 là phân tích cơ chế thứ cấp. Không tạo p-value E5 − E4 vì hai điều kiện dữ liệu trùng
bitwise.

## Cổng 6 — benchmark và bằng chứng bổ sung

- Chạy `scripts/benchmark_model_complexity.py` trên cùng một GPU cho E0, E1 và E2--E6.
- Báo cáo số tham số, latency trung vị/p95, throughput và peak VRAM.
- Không dùng monitoring thiếu của chiến dịch huấn luyện để tuyên bố tốc độ chính thức.
- Chạy phân tích không gian đặc trưng bằng `scripts/analyze_feature_space.py` sau khi chọn rõ
  checkpoint/fold/role; t-SNE chỉ mô tả, Silhouette là số định lượng hỗ trợ.

## Cổng 7 — viết khóa luận hoặc bài báo

- Dùng Macro-F1 làm chỉ số chính; Accuracy và kappa là phụ.
- Mô tả “đơn giản hóa” theo số mô hình/giai đoạn vận hành, không gọi E2--E6 tiết kiệm tham số.
- Nêu rõ chỉ dùng một training seed 42.
- Không tuyên bố domain shift, zero-shot SHHS hoặc giá trị lâm sàng từ kết quả Sleep-EDF này.
- Nếu mở rộng SHHS hoặc multi-modal, xây dựng giao thức riêng sau khi hoàn tất báo cáo in-domain.
