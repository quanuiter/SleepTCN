# Việc cần làm tiếp theo

Nguồn trạng thái: `STATUS_V2.md`. Runbook lệnh đầy đủ: `DOCKER_GPU_RUNBOOK.md`.

## Cổng 1 — dữ liệu và split — ĐẠT

- 765/765 NPZ hợp lệ, năm biến thể có cùng nhãn/chỉ số epoch.
- Split 10-fold seed 42 theo đối tượng đã khóa và SHA-256 khớp sidecar.
- E4/E5 giống bitwise; E5 bị loại khỏi fold 01--09.

Không tiền xử lý lại và không sửa manifest/split trong chiến dịch đang chạy.

## Cổng 2 — smoke CPU/GPU và fold dự toán — ĐẠT

- Smoke CPU/GPU E0--E6 fold 00 đã pass.
- Full validation-only fold 00 E0--E6 đã pass.
- Full validation-only fold 01 E0/E1/E2/E3/E4/E6 đã pass.
- Test vẫn khóa.

## Cổng 3 — hoàn tất validation-only 10-fold — ĐANG THỰC HIỆN

Mỗi phiên thuê chạy một fold, seed huấn luyện 42, thứ tự:

```text
E0 -> E1 -> E2 -> E3 -> E4 -> E6
```

Fold tiếp theo chưa có đủ artifact local là fold 02. Nếu fold 02 đang chạy trên Docker, chỉ đánh
dấu hoàn tất sau khi pull về và xác nhận sáu manifest `complete`, sáu validation report
`passed=true`, có checkpoint và `allow_test_evaluation=false`.

Không chạy E5. Không thêm `--allow-test-evaluation`. Luôn export trước khi chạy CUDA:

```bash
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Sau mỗi fold, commit/push có chọn lọc sáu thư mục fold và monitoring; không commit
`data/cache/`. Lặp lại tới fold 09.

Song song, xử lý nợ lưu trữ fold 00: checkpoint local hiện là 54 Git LFS pointer chứ chưa phải
blob PyTorch. Không cần dừng các fold validation mới vì metrics/report fold 00 vẫn nguyên vẹn,
nhưng phải `git lfs pull` và `git lfs fsck` hoặc phục hồi backup trước khi mở test.

## Cổng 4 — khóa checkpoint và mở test một lần — CHƯA ĐƯỢC PHÉP

Điều kiện vào cổng:

- Fold 00--09 đầy đủ cho E0/E1/E2/E3/E4/E6, seed 42.
- Mọi validation report pass và test chưa từng được mở.
- Code/config/split được khóa; không còn quyết định mô hình dựa trên validation.
- Tất cả checkpoint, đặc biệt fold 00, đã được hydrate và kiểm tra SHA-256; không còn LFS pointer
  chưa có blob.

Khi đạt, chạy lại từng run với `--resume --allow-test-evaluation` để tạo dự đoán test từ
checkpoint đã chọn. Không sửa gì sau khi xem test.

## Cổng 5 — phân tích và báo cáo — CHƯA THỰC HIỆN

- Ghép test prediction theo subject/record/original epoch index.
- So sánh chính: E1−E0, E2−E1, E3−E2, E3−E6.
- Bootstrap ghép cặp theo cụm đối tượng 10.000 lần, Wilcoxon hai phía và Holm.
- Báo cáo Macro-F1 chính; accuracy, kappa, precision/recall/F1 từng lớp là phụ.
- Đo tham số, thời gian, throughput và VRAM trên cùng môi trường GPU.
- Nêu rõ giới hạn chỉ dùng một training seed 42 và chỉ đánh giá in-domain Sleep-EDF.
