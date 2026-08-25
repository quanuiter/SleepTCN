# Gói tái lập và kiểm toán dữ liệu

Gói tái lập gồm hai lớp kiểm chứng bổ sung:

1. `processed_artifact_manifest_v2.json` khóa snapshot NPZ đang dùng bằng path tương đối, kích thước
   và SHA-256 toàn tệp.
2. `preprocess_sleepedf.py` có thể tái sinh từ EDF gốc; writer `sleeptcn_deterministic_npz_v1` cố
   định metadata ZIP để Windows/Linux không tạo hash khác nhau cho cùng mảng.

## Kiểm tra trong môi trường sạch

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

Kết quả hợp lệ phải có trạng thái `PASS`. Khi kiểm toán toàn bộ năm biến thể, có thể bỏ tùy chọn
`--variants` hoặc bổ sung `bandpass_clip_v2`.

## Tái sinh từ EDF gốc

Không ghi đè snapshot dùng cho kết quả công bố. Hãy tạo một thư mục trung gian riêng, sử dụng manifest có
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

Manifest tái sinh cần được đối chiếu theo `output_sha256`, số bản ghi, nhãn và chỉ số epoch. Khi phát hiện
khác biệt, cần lưu cả hai manifest và xác định nguyên nhân ở dữ liệu nguồn, thư viện hoặc quy trình trước
khi cập nhật bất kỳ checksum kỳ vọng nào.
