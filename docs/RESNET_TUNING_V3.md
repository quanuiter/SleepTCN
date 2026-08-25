# ResNet-1D tuning v3

Chiến dịch này phát triển ResNet-1D trên train/validation của Sleep-EDF. Runner không có tùy
chọn mở khóa test và không ghi checkpoint vào Git.

## Chuẩn bị

Tạo branch từ `main`, dùng source của branch đó trong Docker, rồi đặt artifact trên volume riêng:

```text
/workspace/SleepTCN                 source checkout
/datasets/sleeptcn                  dữ liệu processed, chỉ đọc
/artifacts/sleeptcn/v3              checkpoint và log tuning
```

Thiết lập môi trường:

```bash
cd /workspace/SleepTCN
source /workspace/.venv/bin/activate
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Smoke test

Smoke test chỉ chạy một epoch, một fold và không mở test:

```bash
python scripts/run_resnet_tuning.py \
  --workspace /workspace/SleepTCN \
  --search-config configs/tuning/resnet_v3_search.json \
  --candidate budget_120 \
  --fold 0 \
  --seed 42 \
  --device cpu \
  --output-root /artifacts/sleeptcn/v3/resnet_tuning \
  --smoke
```

Run đạt yêu cầu khi tạo được `best.pt`, `latest.pt`, `training_history.json`,
`validation_metrics.json`, `resolved_config.json` và `run_manifest.json`. Manifest phải ghi
`test_records_loaded: false`.

## Sàng lọc ứng viên

Các ứng viên hiện có trong `configs/tuning/resnet_v3_search.json`:

- `baseline_v2`: cấu hình ResNet v2 hiện tại.
- `budget_120`: giữ kiến trúc, tăng ngân sách lên 120 epoch và patience 20.
- `lr_3e4`, `lr_3e3`: thay đổi learning rate.
- `weight_decay_0`: bỏ weight decay.
- `dropout_02`: giảm classifier dropout.

Mỗi lệnh dưới đây chỉ huấn luyện ResNet trên train/validation:

```bash
python scripts/run_resnet_tuning.py \
  --workspace /workspace/SleepTCN \
  --search-config configs/tuning/resnet_v3_search.json \
  --candidate budget_120 \
  --fold 0 \
  --seed 42 \
  --device cuda \
  --num-workers 2 \
  --output-root /artifacts/sleeptcn/v3/resnet_tuning
```

Thay `--candidate`, `--fold` và `--seed` theo chiến dịch đã định trước. Không dùng kết quả test
để chọn ứng viên. Sau khi chọn được hai hoặc ba ứng viên tốt nhất trên validation, mới huấn luyện
TCN trên embedding đã cache để quyết định cấu hình tốt nhất của toàn bộ ResNet--TCN.

Sau khi các run validation hoàn tất, xếp hạng bằng Macro-F1 trung bình ở cấp đối tượng:

```bash
python scripts/summarize_resnet_tuning.py \
  --output-root /artifacts/sleeptcn/v3/resnet_tuning \
  --output /artifacts/sleeptcn/v3/resnet_tuning/validation_ranking.json
```

Runner vẫn dùng Macro-F1 gộp để chọn checkpoint trong từng run, còn việc chọn giữa các candidate
dùng Macro-F1 trung bình theo đối tượng để phù hợp với đơn vị thống kê của paper.

## Khóa ứng viên và chạy E2--E6

Sau khi ứng viên được chọn bằng validation, tạo một config đầy đủ từ cấu hình v2:

```bash
python scripts/create_resnet_locked_config.py \
  --base-config configs/experiments_v2.json \
  --search-config configs/tuning/resnet_v3_search.json \
  --candidate budget_120 \
  --output configs/tuning/experiments_resnet_v3_locked.json
```

Tệp này phải được kiểm tra, commit và giữ nguyên trước khi mở khóa test. Khi chạy E2--E6, dùng
artifact root bên ngoài Git:

```bash
python scripts/run_experiment.py \
  --workspace /workspace/SleepTCN \
  --config configs/tuning/experiments_resnet_v3_locked.json \
  --artifact-root /artifacts/sleeptcn/v3 \
  --experiment E2 \
  --fold 0 \
  --seed 42 \
  --device cuda \
  --num-workers 2
```

Chỉ thêm `--allow-test-evaluation` sau khi toàn bộ cấu hình, seed và quy tắc phân tích đã được
khóa. E2, E3, E4 và E6 phải được chạy lại từ đầu; TCN cũ không tương thích với embedding ResNet
mới.

## Cấu trúc artifact

```text
resnet_tuning/
  <candidate>/
    fold_00/
      seed_42/
        checkpoints/resnet1d/best.pt
        checkpoints/resnet1d/latest.pt
        resolved_config.json
        training_history.json
        validation_metrics.json
        run_manifest.json
```

Chỉ các manifest, hash và bảng tổng hợp nhỏ được đưa vào Git. Checkpoint, NPZ embedding và dữ liệu
processed phải nằm ở volume hoặc kho artifact bên ngoài repository.

## Áp dụng vào E2--E6

Sau khi khóa một cấu hình ResNet, dùng cùng kiến trúc và hyperparameter cho `E2`, `E3`, `E4` và
`E6`. Mỗi E vẫn phải huấn luyện ResNet và TCN riêng cho từng fold/seed; không gắn ResNet mới vào
TCN checkpoint cũ vì embedding đã thay đổi.
