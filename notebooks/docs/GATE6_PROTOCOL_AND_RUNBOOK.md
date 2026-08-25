# Gate 6 — giao thức benchmark và không gian đặc trưng

> **LEGACY / HISTORY:** Nội dung này được giữ để truy nguyên Gate 6. Dùng
> `docs/README.md` để tìm protocol và runbook hiện hành; các bước lịch sử ở đây
> không phải công việc đang chờ.

## Phạm vi đã khóa

Gate 6 là bằng chứng bổ sung sau Gate 5, không được dùng để thay đổi E0–E6 hoặc lựa chọn lại checkpoint.

### Benchmark

- Dùng checkpoint thật fold 00, seed 42 của E0, E1, E2, E3, E4 và E6.
- Cùng một GPU, cùng tensor đầu vào `(1, 100, 1, 3000)` float32.
- Đo suy luận end-to-end từ 100 epoch EEG đến logits chuỗi; không gồm đọc đĩa, preprocessing, cache và training.
- Ba vòng; mỗi vòng xáo thứ tự sáu mô hình bằng seed 42.
- Mỗi mô hình/vòng: 20 lượt làm nóng, 100 lượt đo; tổng 300 lượt đo/mô hình.
- Đồng bộ CUDA trước và sau từng lượt. Báo cáo latency trung vị, p95, throughput và peak VRAM.
- Số tham số được tính từ đúng mô hình đã nạp checkpoint.
- E2/E3/E4/E6 cùng kiến trúc; chênh lệch tốc độ giữa chúng chỉ là nhiễu phép đo.

### Không gian đặc trưng

- Câu hỏi hỗ trợ: đặc trưng 128D của ResNet-1D E2 có tách năm lớp tốt hơn 75 xác suất softmax của 15CNN E1 không?
- Dùng E1 và E2 vì cùng dữ liệu `paper_raw_v1` và cùng TCN; khác bộ trích đặc trưng.
- Dùng test của cả 10 fold, checkpoint tương ứng từng fold; không trộn vector của checkpoint khác nhau vào cùng không gian.
- Trong mỗi fold lấy 200 epoch mỗi lớp, tổng 1.000 epoch/fold và 10.000 epoch toàn bộ.
- E1/E2 dùng chính xác cùng khóa epoch. Mẫu được phân tán vòng tròn qua các đối tượng để tránh một người chi phối.
- StandardScaler và PCA 20 chiều được fit riêng cho từng biểu diễn/từng fold; Silhouette tính trên 1.000 mẫu fold đó.
- Báo cáo chênh lệch Silhouette E2−E1 trên 10 fold. Đây là phân tích hỗ trợ sau Gate 5, không phải kiểm định xác nhận.
- t-SNE chỉ vẽ fold 00 để minh họa, cùng 1.000 epoch; không dùng t-SNE để suy luận nhân quả.

## Phần chạy trên máy cục bộ, không cần GPU

Sau khi pull commit Gate 6 và bảo đảm Git sạch:

```powershell
cd D:\SleepTCN
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -p "test_*.py"

python scripts/benchmark_model_complexity.py `
  --workspace "$PWD" `
  --mode parameters `
  --fold 0 `
  --seed 42 `
  --output runs/v2/analysis/gate6_parameters_fold00_seed42.json

python scripts/analyze_feature_space.py `
  --workspace "$PWD" `
  --mode prepare `
  --seed 42 `
  --sample-per-class-per-fold 200 `
  --sample-manifest runs/v2/analysis/gate6_feature_samples_seed42.json
```

Hai lệnh trên chỉ đọc checkpoint/dữ liệu và tạo báo cáo nhỏ; không sửa test prediction.

## Phần chạy trên Docker GPU

Máy Docker cần toàn bộ checkpoint E0–E6 và `data/processed/paper_raw_v1` của 153 bản ghi. Không cần các
biến thể preprocessing khác cho phân tích đặc trưng.

```bash
cd /workspace/SleepTCN
git switch run-in-docker
git pull origin run-in-docker
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m unittest discover -s tests -p 'test_*.py'
git status --short
```

Chạy benchmark trong tmux:

```bash
tmux new -s gate6
cd /workspace/SleepTCN
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
set -o pipefail

python scripts/benchmark_model_complexity.py \
  --workspace /workspace/SleepTCN \
  --mode latency \
  --device cuda \
  --fold 0 \
  --seed 42 \
  --batch-records 1 \
  --sequence-length 100 \
  --warmup 20 \
  --repeats 100 \
  --rounds 3 \
  --output runs/v2/analysis/gate6_latency_fold00_seed42.json \
  2>&1 | tee runs/v2/analysis/gate6_latency.log
```

Sau đó phân tích không gian đặc trưng:

```bash
python scripts/analyze_feature_space.py \
  --workspace /workspace/SleepTCN \
  --mode prepare \
  --seed 42 \
  --sample-per-class-per-fold 200 \
  --sample-manifest runs/v2/analysis/gate6_feature_samples_seed42.json

python scripts/analyze_feature_space.py \
  --workspace /workspace/SleepTCN \
  --mode analyze \
  --sample-manifest runs/v2/analysis/gate6_feature_samples_seed42.json \
  --output-dir runs/v2/analysis/gate6_feature_space \
  --device cuda \
  --batch-size 256 \
  --pca-dimensions 20 \
  --silhouette-sample 1000 \
  2>&1 | tee runs/v2/analysis/gate6_feature_space.log
```

## Kiểm định Gate 6

```bash
python scripts/validate_gate6_artifacts.py \
  --parameters runs/v2/analysis/gate6_parameters_fold00_seed42.json \
  --latency runs/v2/analysis/gate6_latency_fold00_seed42.json \
  --feature-samples runs/v2/analysis/gate6_feature_samples_seed42.json \
  --feature-output-dir runs/v2/analysis/gate6_feature_space \
  --output runs/v2/analysis/gate6_validation_report.json
```

Chỉ lưu Git khi báo cáo trả `status: passed`. Không commit dữ liệu processed, cache hoặc checkpoint mới.
Không commit `gate6_feature_samples_seed42.json` hoặc `tsne_points.csv` vì hai tệp chứa khóa
đối tượng/bản ghi/epoch; chúng phải được giữ cục bộ và có thể tái tạo đúng từ seed đã khóa. Chỉ commit
báo cáo tổng hợp, hình không định danh, log và báo cáo kiểm định.
