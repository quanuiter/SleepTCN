# Hợp đồng tái chạy package

Package hiện có hai lớp kiểm chứng tách biệt:

1. `processed_artifact_manifest_v2.json` khóa snapshot NPZ đang dùng bằng path tương đối, kích thước
   và SHA-256 toàn tệp.
2. `preprocess_sleepedf.py` có thể tái sinh từ EDF gốc; writer `sleeptcn_deterministic_npz_v1` cố
   định metadata ZIP để Windows/Linux không tạo hash khác nhau cho cùng mảng.

## Kiểm tra trong Docker sạch

Dùng Python 3.11 và lock đầy đủ:

```bash
cd /workspace
python3.11 -m venv .venv
source /workspace/.venv/bin/activate
cd /workspace/SleepTCN
python -m pip install --upgrade pip
python -m pip install -r requirements/lock-cu121.txt
export PYTHONPATH="$PWD/src"

python scripts/audit_reproducibility.py \
  --workspace "$PWD" \
  --manifest data/manifests/processed_artifact_manifest_v2.json \
  --variants paper_raw_v1 filtered_v2 bandpass_v2 filtered_zscore_v2 \
  --output runs/v2/reproducibility_audit_gpu.json
```

Lệnh phải in `PASS`. Khi upload cả E5 để kiểm toán đầy đủ, bỏ `--variants` hoặc thêm
`bandpass_clip_v2`. Không đổi tên thư mục thành `bandpass_clip_v2_NoNeed`.

## Tái sinh từ raw EDF

Không ghi đè snapshot đã dùng cho kết quả công bố. Tạo output staging riêng, dùng raw manifest có
SHA-256 và chạy:

```bash
export PYTHONPATH="$PWD/src"
python scripts/preprocess_sleepedf.py \
  --data-dir /path/to/sleep-cassette \
  --raw-manifest data/manifests/raw_inventory.json \
  --output-root /tmp/sleeptcn-processed \
  --manifest-output /tmp/sleeptcn-preprocess-manifest.json \
  --variants paper_raw_v1 filtered_v2 bandpass_v2 bandpass_clip_v2 filtered_zscore_v2

python scripts/canonicalize_npz.py \
  --processed-root /tmp/sleeptcn-processed --check
```

Manifest tái sinh phải được so sánh theo `output_sha256`, số record, nhãn và chỉ số epoch. Nếu
hash khác, giữ lại cả hai manifest để điều tra; không sửa checksum kỳ vọng cho đến khi xác định rõ
khác biệt ở dữ liệu nguồn, thư viện hoặc quy trình.
