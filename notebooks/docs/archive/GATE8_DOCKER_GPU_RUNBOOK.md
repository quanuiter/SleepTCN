# Runbook Docker GPU — Gate 8

> **LEGACY / HISTORY:** Runbook Gate 8 riêng đã được đóng. Dùng
> `../DOCKER_GPU_RUNBOOK.md` cho quy trình vận hành GPU hiện hành.

Runbook này chỉ áp dụng sau khi mã Gate 8 đã được commit và push lên nhánh `run-in-docker`.
Không chạy các lệnh test trước khi đủ 30/30 validation run.

## A. Đồng bộ và kiểm tra trước GPU

```bash
cd /workspace/SleepTCN
git switch run-in-docker
git pull --ff-only origin run-in-docker
git status --short
git rev-parse HEAD

source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m unittest discover -s tests -p 'test_*.py'
python scripts/prepare_gate8_context_ablation.py \
  --workspace /workspace/SleepTCN \
  --seed 42 \
  --output /workspace/SleepTCN/runs/v2/gate8/preflight_seed42.json
```

Điều kiện bắt buộc:

- `git status --short` không in gì;
- toàn bộ kiểm thử đạt;
- preflight có `status: prepared`, `target_count: 30`, `processed_records: 153`,
  `validated_source_runs: 20`;
- chưa tồn tại prediction test trong `runs/v2/gate8/full/`.

## B. Chạy 30 mô hình validation trong tmux

```bash
tmux new -s gate8-train
cd /workspace/SleepTCN
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python scripts/run_gate8_campaign.py \
  --workspace /workspace/SleepTCN \
  --seed 42 \
  --device cuda \
  --num-workers 4 \
  2>&1 | tee runs/v2/gate8/validation_campaign_seed42.log
```

Tách khỏi tmux: `Ctrl+B`, sau đó `D`. Có thể đóng trình duyệt nếu máy thuê/container vẫn hoạt động.

Theo dõi ở terminal thứ hai:

```bash
cd /workspace/SleepTCN
source .venv/bin/activate
watch -n 20 python scripts/show_gate8_progress.py \
  --workspace /workspace/SleepTCN \
  --seed 42
```

Không chạy song song hai campaign trên cùng GPU. Ba điều kiện chạy tuần tự để dùng chung feature cache và
tránh tranh VRAM/CPU/RAM.

Nếu tiến trình bị ngắt:

```bash
cd /workspace/SleepTCN
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

tmux new -s gate8-resume
python scripts/run_gate8_campaign.py \
  --workspace /workspace/SleepTCN \
  --seed 42 \
  --device cuda \
  --num-workers 4 \
  --resume \
  2>&1 | tee -a runs/v2/gate8/validation_campaign_seed42.log
```

Không xóa `latest.pt`, feature cache, vector trung bình hay journal trước khi resume.

## C. Kiểm tra 30/30 trước khi mở test

Khi campaign báo `validation_complete`, chạy dry-run:

```bash
cd /workspace/SleepTCN
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

git status --short
python scripts/evaluate_gate8_locked_test.py \
  --workspace /workspace/SleepTCN \
  --seed 42 \
  --dry-run
```

Chỉ đi tiếp nếu có `status: prepared`, `target_count: 30` và không có lỗi SHA-256. Dừng và lưu lại
toàn bộ thông báo nếu có bất kỳ sai lệch nào.

## D. Mở test đúng một lần

```bash
tmux new -s gate8-test
cd /workspace/SleepTCN
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python scripts/evaluate_gate8_locked_test.py \
  --workspace /workspace/SleepTCN \
  --seed 42 \
  --device cuda \
  --num-workers 4 \
  --execute \
  --confirm OPEN-GATE8-LOCKED-TEST-ONCE \
  2>&1 | tee runs/v2/gate8/test_campaign_seed42.log
```

Nếu bị ngắt:

```bash
python scripts/evaluate_gate8_locked_test.py \
  --workspace /workspace/SleepTCN \
  --seed 42 \
  --device cuda \
  --num-workers 4 \
  --resume \
  --confirm OPEN-GATE8-LOCKED-TEST-ONCE \
  2>&1 | tee -a runs/v2/gate8/test_campaign_seed42.log
```

## E. Phân tích sau test

Chỉ chạy khi journal test có `status: complete` và đủ 30 mục:

```bash
cd /workspace/SleepTCN
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

python scripts/analyze_gate8_results.py \
  --workspace /workspace/SleepTCN \
  --seed 42 \
  --output /workspace/SleepTCN/runs/v2/gate8/analysis_seed42.json
```

Không sửa protocol, split, seed, mô hình, checkpoint hay quy tắc chuyển pha sau khi xem test.

## F. Artifact phải lưu trước khi trả Docker

- `runs/v2/gate8/validation_campaign_seed42.json` và `.log`;
- `runs/v2/gate8/test_campaign_seed42.json` và `.log`;
- 30 `best.pt` và 30 `latest.pt` nếu muốn resume đầy đủ;
- 30 vector `feature_mask/train_replacement_mean.npz`;
- 30 prediction/metrics validation và test;
- 30 `run_manifest.json`, `validation_report.json`;
- `runs/v2/gate8/analysis_seed42.json`.

`data/cache/features/` có thể tạo lại từ E0 checkpoint và dữ liệu, không bắt buộc đưa lên Git. Không đưa
`data/processed/` hoặc dataset gốc lên Git.
