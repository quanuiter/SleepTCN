# Runbook chạy GPU Docker — SleepTCN v2

Tài liệu này là quy trình chuẩn để chạy một outer fold trên Docker GPU rồi lưu artifact lên
branch `run-in-docker`. Đợt hiện tại chỉ dùng training seed `42`, chạy một fold cho mỗi phiên
thuê, với thứ tự `E0 -> E1 -> E2 -> E3 -> E4 -> E6`. Không chạy E5 và không dùng
`--allow-test-evaluation`.

Ví dụ bên dưới chạy fold 02. Với fold khác, thay đồng nhất `2` thành chỉ số fold cần chạy và
`fold_02` thành thư mục tương ứng.

## 1. Tạo workspace và lấy source đúng branch

Mở Jupyter Terminal/terminal trong Docker. Đường dẫn dùng trong tài liệu là
`/workspace/SleepTCN`.

```bash
cd /workspace
git clone --branch run-in-docker https://github.com/quanuiter/SleepTCN.git SleepTCN
cd /workspace/SleepTCN
git status --short
git log -1 --oneline
```

Nếu đã có workspace từ phiên trước, không clone lại:

```bash
cd /workspace/SleepTCN
git fetch origin
git switch run-in-docker
git pull --ff-only
git status --short
```

`git status --short` phải không in gì trước khi chạy full experiment. Nếu source được upload thủ
công chứ không clone, tạo Git repository/remote và chắc chắn checkout đúng commit từ branch
`run-in-docker` trước khi tiếp tục.

## 2. Upload dữ liệu đã xử lý

Không upload EDF gốc, `data/cache/`, `runs/`, hoặc virtual environment cũ. Cần upload các thư
mục dưới `data/processed/` (mỗi thư mục đủ 153 tệp `.npz`):

```text
paper_raw_v1/         # E0, E1, E2
filtered_v2/          # E3
bandpass_v2/          # E4
filtered_zscore_v2/   # E6
```

`bandpass_clip_v2/` chỉ phục vụ E5; E5 đã bị loại vì dữ liệu bitwise-identical với E4, nên không
cần upload cho lịch chạy hiện tại.

Các manifest và split sau phải có trong source:

```text
data/manifests/raw_inventory.json
data/manifests/processed_validation_v2.json
data/manifests/bandpass_clip_identity_v2.json
data/manifests/processed_artifact_manifest_v2.json
data/manifests/processed_artifact_manifest_v2.json.sha256
data/manifests/reproducibility_audit_v2.json
data/manifests/reproducibility_audit_v2.json.sha256
requirements/lock-cu121.txt
requirements/lock-cu121.txt.sha256
environment/pip-freeze.txt
environment/pip-freeze.txt.sha256
data/splits/sleepedf_sc_10fold_seed42_v2.json
```

Kiểm tra nhanh trước khi trả tiền GPU:

```bash
cd /workspace/SleepTCN
df -h .
for variant in paper_raw_v1 filtered_v2 bandpass_v2 filtered_zscore_v2; do
  printf '%s: ' "$variant"
  find "data/processed/$variant" -maxdepth 1 -name '*.npz' | wc -l
done
```

Mỗi dòng phải là `153`. Nên có ít nhất 50 GB trống; 80 GB an toàn hơn.

## 3. Tạo venv và cài PyTorch CUDA

Kiểm tra GPU và Python trước:

```bash
nvidia-smi
python3 --version
```

Tạo environment trong `/workspace` để không lẫn với source. Lock file GPU đã được chốt cho
Python 3.11:

```bash
cd /workspace
python3 -m venv .venv
source /workspace/.venv/bin/activate

cd /workspace/SleepTCN
python -m pip install --upgrade pip
python -m pip install -r requirements/lock-cu121.txt
```

`cu121` là bộ wheel CUDA 12.1 đã dùng cho các fold trước. Driver mới hơn vẫn tương thích ngược
với wheel này. Không dùng wheel CPU-only. Xác minh ngay:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('torch CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
PY
```

Kết quả phải có `cuda available: True` và tên GPU đã thuê. Nếu `False`, dừng lại; không chạy
pipeline bằng CPU trên Docker GPU.

## 4. Kiểm tra môi trường và đặt biến môi trường chuẩn

Mỗi terminal mới cần kích hoạt venv và các biến này:

```bash
cd /workspace/SleepTCN
source /workspace/.venv/bin/activate

export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Từ fold 02 trở đi, `CUBLAS_WORKSPACE_CONFIG` là điều kiện vận hành đã khóa và phải được export
**trước mọi lệnh Python dùng CUDA**. Nó loại cảnh báo cuBLAS và tăng khả năng tái lập bitwise.
Thiếu biến này không làm một run đã hoàn tất trở thành sai khoa học; vì vậy fold 00/01 vẫn hợp lệ.

Sau đó chạy kiểm tra hợp đồng:

```bash
python scripts/check_environment.py \
  --workspace /workspace/SleepTCN \
  --require-gpu \
  --output runs/v2/environment_check_gpu.json

python scripts/audit_reproducibility.py \
  --workspace /workspace/SleepTCN \
  --manifest data/manifests/processed_artifact_manifest_v2.json \
  --variants paper_raw_v1 filtered_v2 bandpass_v2 filtered_zscore_v2 \
  --output runs/v2/reproducibility_audit_gpu.json
```

Hai lệnh trên phải cùng trả `PASS`. Lệnh audit xác nhận hash toàn tệp, ZIP metadata và khả năng đọc
NPZ trong workspace Docker; nó không cần truy cập EDF gốc. Mặc định lệnh kiểm tra bốn biến thể của sáu experiment đang hoạt động. Chỉ dùng thêm
`--include-retired-e5` khi muốn audit lại E5 và đã upload `bandpass_clip_v2`. Chỉ tiếp tục khi
báo `PASS`. Nếu báo `BadZipFile`, upload lại tệp NPZ bị báo lỗi trước khi chạy.

## 5. Chạy một fold đầy đủ, validation-only

Ví dụ này chạy fold 02. Nó tạo một log GPU nhỏ, chạy đúng dependency E0 trước E1, kiểm định
artifact sau từng experiment, và dừng tại lỗi đầu tiên. Không thêm `--resume` cho run mới.

```bash
cd /workspace/SleepTCN
source /workspace/.venv/bin/activate
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

git status --short
mkdir -p runs/v2/monitoring

nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
  --format=csv -l 10 \
  > runs/v2/monitoring/fold_02_seed42_gpu.csv &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT

set -euo pipefail

for experiment in E0 E1 E2 E3 E4 E6; do
  run_root="runs/v2/full/${experiment}/fold_02/seed_42"

  printf 'START %s %s\n' "$experiment" "$(date -Is)" \
    >> runs/v2/monitoring/fold_02_seed42_time.txt

  python scripts/run_experiment.py \
    --workspace /workspace/SleepTCN \
    --experiment "$experiment" \
    --fold 2 \
    --seed 42 \
    --device cuda \
    --num-workers 2

  python scripts/validate_run_artifacts.py \
    --workspace /workspace/SleepTCN \
    --run-root "$run_root" \
    --output "$run_root/validation_report.json"

  printf 'DONE %s %s\n' "$experiment" "$(date -Is)" \
    >> runs/v2/monitoring/fold_02_seed42_time.txt
done
```

Trước khi chạy block, `git status --short` phải trống. Dòng này có thể in ra 0 dòng; đó là kết
quả đúng. E5 không nằm trong vòng lặp. Tập test vẫn khóa vì không có
`--allow-test-evaluation`.

## 6. Theo dõi, dừng và tiếp tục an toàn

Mở terminal thứ hai, kích hoạt venv, rồi theo dõi checkpoint của fold 02:

```bash
cd /workspace/SleepTCN
source /workspace/.venv/bin/activate
watch -n 20 'python scripts/show_training_progress.py --fold 2 --seed 42'
```

Kiểm tra tiến trình/GPU:

```bash
pgrep -af 'run_experiment.py'
nvidia-smi
```

Nếu Docker/session ngắt trong một experiment, không xóa checkpoint. Chạy lại **chỉ experiment
đang dở** và thêm `--resume`; ví dụ E4:

```bash
cd /workspace/SleepTCN
source /workspace/.venv/bin/activate
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python scripts/run_experiment.py \
  --workspace /workspace/SleepTCN \
  --experiment E4 \
  --fold 2 \
  --seed 42 \
  --device cuda \
  --num-workers 2 \
  --resume
```

Sau đó chạy lại `validate_run_artifacts.py` cho E4. E1 luôn cần E0 cùng fold/seed hoàn thành
trước. E2, E3, E4, E6 không phụ thuộc checkpoint lẫn nhau, nhưng vẫn chạy tuần tự để dễ theo dõi
và chỉ dùng một GPU.

Nếu cần dừng một run treo, mở terminal thứ hai:

```bash
pkill -TERM -f 'python scripts/run_experiment.py'
```

Đợi vài giây rồi kiểm tra `pgrep`. Chỉ khi vẫn còn tiến trình mới dùng:

```bash
pkill -KILL -f 'python scripts/run_experiment.py'
```

Không dùng `pkill python`, vì sẽ có thể dừng nhầm Jupyter.

## 7. Xác nhận fold hoàn tất

Sau khi cả fold hoàn tất, xác nhận sáu run đủ manifest/report/checkpoint và test chưa mở:

```bash
python - <<'PY'
import json
from pathlib import Path

experiments = ('E0', 'E1', 'E2', 'E3', 'E4', 'E6')
failed = []
for exp in experiments:
    root = Path('runs/v2/full') / exp / 'fold_02' / 'seed_42'
    manifest = json.loads((root / 'run_manifest.json').read_text())
    report = json.loads((root / 'validation_report.json').read_text())
    checkpoints = list((root / 'checkpoints').rglob('best.pt'))
    ok = (
        manifest.get('status') == 'complete'
        and manifest.get('allow_test_evaluation') is False
        and report.get('passed') is True
        and bool(checkpoints)
    )
    print(exp, 'PASS' if ok else 'FAIL')
    if not ok:
        failed.append(exp)
if failed:
    raise SystemExit(f'Failed: {failed}')
print('ALL FOLD-02 RUNS PASS')
PY
```

## 8. Lưu lên GitHub, không lưu cache

Không commit `data/cache/` và không dùng `git add -A`. Checkpoint/result đang bị `.gitignore`
chặn chủ động, nên phải `git add -f` đúng sáu thư mục đã kiểm định. Kiểm tra trước rằng không có
file nào vượt giới hạn GitHub 100 MB:

```bash
find runs/v2/full/{E0,E1,E2,E3,E4,E6}/fold_02 -type f -size +90M -print
```

Nếu không có output:

```bash
git add -f \
  runs/v2/full/E0/fold_02 \
  runs/v2/full/E1/fold_02 \
  runs/v2/full/E2/fold_02 \
  runs/v2/full/E3/fold_02 \
  runs/v2/full/E4/fold_02 \
  runs/v2/full/E6/fold_02

git add -f \
  runs/v2/monitoring/fold_02_seed42_gpu.csv \
  runs/v2/monitoring/fold_02_seed42_time.txt

git diff --cached --name-only | grep 'data/cache' \
  && { echo 'ERROR: cache was staged'; exit 1; } \
  || echo 'OK: cache is not staged'

git diff --cached --stat
git commit -m 'Add fold 02 validation runs and checkpoints'
git push origin run-in-docker
```

Sau push, ở máy cá nhân chạy `git pull` trên branch `run-in-docker`, rồi kiểm tra đủ sáu
`run_manifest.json` và `validation_report.json`. Có thể tắt Docker khi commit đã push thành công.

Repository hiện có `.gitattributes` gán `*.pt` cho Git LFS. Nếu máy Docker đã cài Git LFS, phải
chạy thêm `git lfs push --all origin run-in-docker` và `git lfs fsck`; nếu không dùng Git LFS,
kiểm tra checkpoint được lưu trực tiếp là file nhị phân vài MB chứ không phải pointer ASCII
130--132 byte. Tuyệt đối không coi commit pointer là backup hoàn chỉnh nếu blob LFS chưa được đẩy.

Kiểm tra nhanh sau khi pull về máy cá nhân:

```bash
find runs/v2/full -path '*/fold_02/seed_42/checkpoints/*.pt' \
  -o -path '*/fold_02/seed_42/checkpoints/*/*.pt' | xargs -r file
```

Checkpoint thật được nhận diện là Zip archive/PyTorch data và có kích thước hàng chục KB đến vài
MB. Nếu là `ASCII text` chứa `git-lfs.github.com/spec`, cần hydrate Git LFS trước khi xóa Docker.

## 9. Nguyên tắc không được thay đổi

- Không chạy E5 ở fold 01--09: nó trùng dữ liệu E4 bitwise, đã có chứng cứ trong
  `data/manifests/bandpass_clip_identity_v2.json`.
- Không thêm `--allow-test-evaluation` cho tới khi tất cả fold validation-only đã hoàn tất và
  protocol được khóa.
- Không đổi code, config, biến thể dữ liệu, seed hoặc split giữa các fold.
- Từ fold 02 trở đi phải export `CUBLAS_WORKSPACE_CONFIG` trước khi bắt đầu hoặc resume. Nếu phát
  hiện thiếu biến khi run đang dở, ưu tiên dừng và chạy lại run đó từ đầu để log điều kiện sạch;
  đây là quy tắc nhất quán vận hành, không phải tuyên bố rằng kết quả thiếu biến luôn vô hiệu.
- Chạy xong phải push artifact và pull/audit trên máy cá nhân trước khi tắt Docker.
