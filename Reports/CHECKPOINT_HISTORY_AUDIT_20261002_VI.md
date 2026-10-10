# Đã tìm và xác minh checkpoint trong lịch sử Git

Ngày 02-10-2026. Báo cáo này **đính chính nhận định thiếu checkpoint E4 và chín fold E3 còn lại** trong các báo cáo tiến độ trước. Lần rà trước mới kiểm tra hiện vật ở working tree và demo assets, chưa kiểm tra đủ cây Git của nhánh huấn luyện. Người dùng chỉ đúng nơi cần kiểm tra: `run-in-docker` và các commit cũ.

## Kết quả xác minh

Commit được kiểm tra trực tiếp: `a005df314980c23e0cc9ef7467c07482243e5898` — `Restore seed-42 checkpoints and analysis artifacts` (25-08-2026), hiện được local remote-tracking ref `origin/run-in-docker` trỏ tới. Không fetch hoặc khẳng định đây là trạng thái mới nhất trên máy chủ.

| Cấu hình | Seed | Fold đầy đủ | Checkpoint best đã xác minh |
|---|---:|---:|---:|
| E3 | 42 | 00–09 | 10 encoder + 10 TCN |
| E3 | 123 | 00–09 | 10 encoder + 10 TCN |
| E4 | 42 | 00–09 | 10 encoder + 10 TCN |
| E4 | 123 | 00–09 | 10 encoder + 10 TCN |

- Đọc được binary checkpoint thật bằng `git cat-file`, không chỉ thấy tên file hay LFS pointer.
- Tổng cộng **490 đường dẫn checkpoint riêng biệt**, 1.197.533.803 byte, khớp SHA-256 tham chiếu. Các bản tham chiếu trùng giữa inventory đã được gộp theo đường dẫn, không cộng lặp.
- Trong đó 470 đường dẫn được đối chiếu với ba inventory SHHS đã có: inventory chính seed 42, inventory component và inventory mở rộng E4 seed 123. Hai mươi checkpoint E4 seed 42 được đối chiếu với `complete.json` trong lịch sử; encoder còn được đối chiếu với `run_manifest.json`.
- Cả **80 checkpoint E3/E4** được mở bằng restricted `torch.load(weights_only=True)` với allowlist các kiểu NumPy cần cho RNG state; kiểm tra experiment, stage, fold, seed, variant, config/split hash, progress và completion marker (`smoke=false`). Không dùng pickle không hạn chế để kiểm tra.
- Cũng xác nhận các bộ E0/E2/E6 cho cả hai seed theo inventory; E1 seed 42 có mười checkpoint sequence được inventory component tham chiếu. Không suy từ số lượng này rằng mọi campaign khác đều đã được kiểm tra.

Nhánh local `run-in-docker` tại `3ef0c7b` đã có bộ seed 123, gồm E3/E4 đủ mười folds. Commit `8af1d121fc86dde30d1c393dbc9fc4c542c37244` — `Archive all seed 123 folds and artifacts` — cũng chứa chúng. Commit `a005df3` bổ sung các bộ seed 42, nên có thể dùng một revision đó để đọc các bộ vừa xác minh.

Ví dụ đường dẫn trong cây Git:

```text
runs/v2/full/E4/fold_00/seed_42/checkpoints/resnet1d/best.pt
runs/v2/full/E4/fold_00/seed_42/checkpoints/sequence/tcn/best.pt
runs/v2/full/E4/fold_00/seed_123/checkpoints/resnet1d/best.pt
runs/v2/full/E4/fold_00/seed_123/checkpoints/sequence/tcn/best.pt
```

## Kết quả tìm trên ổ E:

Tìm tên file checkpoint/archive trong các vùng ổ E: đọc được chỉ tìm thấy checkpoint smoke ở `E:/SleepTCN/runs/smoke`; không tìm thấy bộ E4 đầy đủ dưới dạng file rời trong lượt tìm này. `System Volume Information` không cho đọc; không cần truy cập vùng hệ thống đó vì đã xác minh được bản đầy đủ trong Git. Inventory SHHS ở `E:/research/Dataset/SHHS_v1/manifests` vẫn là đầu mối để đối chiếu hash.

## Hệ quả cho công việc tiếp theo

1. **Không còn cần yêu cầu người dùng tìm lại hoặc huấn luyện lại các checkpoint E3/E4 nói trên chỉ vì thiếu file ở working tree.** Có thể khôi phục có chọn lọc từ Git vào một thư mục chạy riêng.
2. E4 seed 42 có checkpoint nguồn hợp lệ; điều đó **không có nghĩa đã có kết quả SHHS seed 42 mới**. Vẫn cần lập manifest/giao thức mở rộng riêng và thực hiện inference/đánh giá nếu muốn thêm đối chiếu này.
3. Có đủ encoder E3 cho các fold còn lại để chuẩn bị mở rộng weighted/unweighted và calibration theo fold. Audit hiện vật không thay thế việc chạy các thực nghiệm đó.
4. Không checkout, reset, sửa nhánh, sửa manifest lịch sử hoặc ghi đè working tree trong lượt audit. Chưa khôi phục hàng loạt checkpoint, chưa chạy train/inference mới.

## Tái kiểm tra

Script: `scripts/audit_checkpoint_git_history.py`.

Output máy đọc: `runs/teacher_revision_cpu_20261002/checkpoint_history_audit.json` — status `passed`, chứa revision, Git blob OID, SHA-256, kích thước, metadata/progress E3/E4 và nguồn tham chiếu cho từng checkpoint. Không chứa dữ liệu EEG hoặc danh sách người SHHS.

```powershell
.venv/Scripts/python.exe scripts/audit_checkpoint_git_history.py --revision a005df3 --shhs-root E:/research/Dataset/SHHS_v1 --output runs/teacher_revision_cpu_20261002/checkpoint_history_audit_repeat.json
```

Script từ chối ghi đè output có sẵn.
