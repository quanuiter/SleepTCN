# Runbook Gate 4 — mở test đúng một lần

> **LEGACY / HISTORY:** Đây là hồ sơ quy trình mở test một lần. Không dùng file
> này làm backlog hoặc mở lại locked-test; xem `STATUS_V2.md` và `docs/README.md`.

## Mục tiêu

Đánh giá đúng 60 checkpoint đã khóa: `E0, E1, E2, E3, E4, E6` × fold `00–09`, seed `42`.
Gate này chỉ suy luận từ `best.pt`; không huấn luyện lại, không chọn lại checkpoint và không ghi đè
prediction/metrics validation.

Không dùng `scripts/run_experiment.py --resume --allow-test-evaluation` cho chiến dịch 60 run. Lệnh đó
được thiết kế cho một run riêng lẻ và sẽ cập nhật artifact khiến worktree bẩn giữa chiến dịch. Dùng duy
nhất `scripts/evaluate_locked_test.py`.

## Các chốt an toàn

- Danh sách cố định 60 mục tiêu; E5 không tham gia.
- `--dry-run` kiểm định checkpoint, manifest và validation artifact nhưng không phân giải hay đọc tập test.
- `--execute` chỉ bắt đầu khi Git sạch và chưa tồn tại bất kỳ test artifact nào.
- Phải nhập chính xác câu xác nhận `OPEN-LOCKED-TEST-ONCE`.
- Mỗi test prediction được kiểm định độc lập ngay sau khi sinh.
- Validation prediction, validation metrics và checkpoint phải giữ nguyên SHA-256.
- Nhật ký `runs/v2/test_campaign_seed42.json` cho phép tiếp tục sau khi phiên Docker bị ngắt.
- Khi tiếp tục, chương trình chỉ chấp nhận các thay đổi manifest/report do chính chiến dịch test tạo ra;
  thay đổi mã nguồn, cấu hình hoặc split sẽ bị từ chối.

## Bước 1 — kiểm tra trên máy cục bộ, chưa cần GPU

Từ PowerShell trong `D:\SleepTCN`:

```powershell
git status --short
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -p "test_*.py"
python scripts/evaluate_locked_test.py `
  --workspace "$PWD" `
  --seed 42 `
  --dry-run
```

Điều kiện đạt:

- Git không có thay đổi.
- Toàn bộ kiểm thử đạt.
- Dry-run kết thúc với `status: prepared` và `target_count: 60`.
- Không xuất hiện `predictions/test.npz` hoặc `metrics/test.json`.

## Bước 2 — chuẩn bị máy Docker GPU

Máy Docker phải có đúng commit đã vượt qua Bước 1, toàn bộ checkpoint đã tải từ Git, cùng năm thư mục
dữ liệu đã tiền xử lý dưới `data/processed/`:

```text
paper_raw_v1
bandpass_v2
bandpass_clip_v2
filtered_v2
filtered_zscore_v2
```

`bandpass_clip_v2` vẫn cần cho kiểm toán dữ liệu nhưng chiến dịch test không chạy E5. Không tải
`data/cache/`; đặc trưng test sẽ được sinh lại từ checkpoint đã khóa.

Trên Docker:

```bash
cd /workspace/SleepTCN
git status --short
git rev-parse HEAD
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m unittest discover -s tests -p 'test_*.py'
python scripts/evaluate_locked_test.py \
  --workspace /workspace/SleepTCN \
  --seed 42 \
  --dry-run
```

Không chạy tiếp nếu Git bẩn, checkpoint thiếu, SHA lệch hoặc dry-run không đủ 60 mục tiêu.

## Bước 3 — chạy test trong tmux

```bash
tmux new -s test-gate
cd /workspace/SleepTCN
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/evaluate_locked_test.py \
  --workspace /workspace/SleepTCN \
  --seed 42 \
  --device cuda \
  --num-workers 4 \
  --execute \
  --confirm OPEN-LOCKED-TEST-ONCE \
  2>&1 | tee runs/v2/test_campaign_seed42.log
```

Tách khỏi tmux bằng `Ctrl+B`, sau đó nhấn `D`. Đóng trình duyệt không làm dừng tmux hoặc tiến trình
trong container, miễn máy thuê vẫn đang hoạt động.

Theo dõi từ terminal thứ hai:

```bash
tmux attach -t test-gate
```

hoặc:

```bash
tail -f /workspace/SleepTCN/runs/v2/test_campaign_seed42.log
```

## Bước 4 — tiếp tục sau khi tiến trình bị ngắt

Không xóa artifact và không chạy lại bằng `run_experiment.py`. Trong đúng repository và đúng commit:

```bash
cd /workspace/SleepTCN
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/evaluate_locked_test.py \
  --workspace /workspace/SleepTCN \
  --seed 42 \
  --device cuda \
  --num-workers 4 \
  --resume \
  --confirm OPEN-LOCKED-TEST-ONCE \
  2>&1 | tee -a runs/v2/test_campaign_seed42.log
```

Không commit giữa chừng. Nhật ký chiến dịch và HEAD phải còn nguyên để `--resume` nhận diện đúng trạng
thái.

## Bước 5 — xác nhận hoàn tất trước khi lưu Git

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('/workspace/SleepTCN')
campaign = json.loads((root / 'runs/v2/test_campaign_seed42.json').read_text())
complete = [item for item in campaign['targets'].values() if item['state'] == 'complete']
test_predictions = list((root / 'runs/v2/full').glob('E*/fold_*/seed_42/predictions/test.npz'))
test_metrics = list((root / 'runs/v2/full').glob('E*/fold_*/seed_42/metrics/test.json'))
print({
    'campaign_status': campaign['status'],
    'completed_targets': len(complete),
    'test_predictions': len(test_predictions),
    'test_metrics': len(test_metrics),
})
assert campaign['status'] == 'complete'
assert len(complete) == len(test_predictions) == len(test_metrics) == 60
PY
```

Sau đó mới force-add các artifact nằm dưới `runs/` vì thư mục này bị `.gitignore`:

```bash
git add -f runs/v2/test_campaign_seed42.json \
  runs/v2/test_campaign_seed42.log \
  runs/v2/full/E*/fold_*/seed_42/run_manifest.json \
  runs/v2/full/E*/fold_*/seed_42/validation_report.json \
  runs/v2/full/E*/fold_*/seed_42/predictions/test.npz \
  runs/v2/full/E*/fold_*/seed_42/metrics/test.json
git status --short
git commit -m "Add locked test evaluation for all 10 folds"
git push origin run-in-docker
```

Không thêm `data/processed/` hoặc `data/cache/` vào Git.

## Bước 6 — sau Gate 4

Chỉ khi đủ 60/60 test artifact đã kiểm định mới chạy `scripts/analyze_paired_results.py` theo bốn so sánh
đã đăng ký trước: E1−E0, E2−E1, E3−E2 và E3−E6. E4−E2 là phân tích cơ chế thứ cấp. Không sửa mô hình,
tiền xử lý, split, seed hoặc giả thuyết sau khi đã xem kết quả test.
