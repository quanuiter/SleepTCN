# Runbook A4: huấn luyện hai biên bộ lọc trên GPU thuê

Giao thức: `docs/AMENDMENT_A4.md`, config khóa `configs/amendment_a4.json`. Runbook này theo đúng cách đã chạy A2
(`docs/AMENDMENT_A2_RUNBOOK_VI.md`); chỗ nào giống, xem chi tiết ở đó.

Khối lượng: 1 lần kiểm tra tái lập + 40 lần huấn luyện (E4HP, E4LP × seed 42, 123 × 10 fold). A2 chạy 50 lần trong
khoảng 7–8 giờ trên một Tesla V100 với 3 tiến trình, nên A4 dự kiến khoảng 6–7 giờ. GPU cần khoảng 16 GB; máy nên
có ít nhất 6 nhân CPU và 30 GB đĩa trống.

## Phần I. Máy local (đã làm hoặc làm trước khi thuê máy)

1. Khóa A4: commit `configs/amendment_a4.json` với `status = locked_before_a4_training`, tiêu đề commit đúng
   `Lock Amendment A4`, rồi push.
2. Dựng đầu vào (CPU, khoảng 20 phút; ghi một lần):

   ```powershell
   python scripts/run_amendment_a4.py prepare
   ```

   Lệnh ghi `highpass_v4` và `lowpass_v4` cho Sleep-EDF (`data/processed/`), mẫu SHHS1 giữ lại
   (`data/processed/shhs_holdout_a4/`) và Telemetry (`data/processed/telemetry_a4/`), kèm manifest
   `runs/a4/manifests/a4_preprocess_manifest.json`. Nếu con đường lọc trên cửa sổ không tái tạo được
   `bandpass_v2` đã lưu ở bên trong cửa sổ, lệnh dừng và báo lỗi.
3. Đóng gói **chỉ phần Sleep-EDF** (khoảng 6,5 GB):

   ```powershell
   python scripts/run_amendment_a4.py bundle
   Get-FileHash ..\a4_inputs_sleepedf.tar -Algorithm SHA256
   ```

   Ghi lại mã SHA-256. Tải `D:\a4_inputs_sleepedf.tar` lên Google Drive, chia sẻ "Bất kỳ ai có đường liên kết", lấy
   ID tệp (đoạn giữa `/d/` và `/view`).

   **Không đưa dữ liệu SHHS lên Drive hay máy thuê.** Thỏa thuận sử dụng dữ liệu của NSRR không cho phép chia sẻ lại.
   Suy luận trên SHHS1 và Telemetry chạy ở máy local bằng CPU sau khi chép checkpoint về (Phần III).

## Phần II. Máy GPU thuê

### II.1. Mã nguồn (cần toàn bộ lịch sử Git)

```bash
nvidia-smi
cd /workspace
git clone https://github.com/quanuiter/SleepTCN.git SleepTCN
cd /workspace/SleepTCN
git status --short          # phải trống
git cat-file -t 8af1d12     # phải in "commit"
```

Không sửa tệp nào trong repo trên máy GPU: runner và bước mở test từ chối chạy khi worktree không sạch.

### II.2. Môi trường

Giống A2 mục II.2: Python 3.11 (dùng `uv` nếu máy chỉ có 3.10), rồi

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/lock-cu121.txt
python -m pip install -e . --no-deps
python -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Dòng cuối phải in `3.11.x 2.5.1+cu121 12.1 True`.

### II.3. Mỗi terminal mới

```bash
cd /workspace/SleepTCN
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### II.4. Dữ liệu

```bash
cd /workspace
python -m pip install --quiet gdown
gdown <ID của a4_inputs_sleepedf.tar> -O /workspace/a4_inputs_sleepedf.tar
sha256sum /workspace/a4_inputs_sleepedf.tar
# phải bằng 4271ea88b4aeb90affe0162a835e3dc00eb53e6835994af0a759bc3c857842ab (gói 7,06 GB, 460 tệp, tạo ngày 11/11/2026)
cd /workspace/SleepTCN
tar -xf /workspace/a4_inputs_sleepedf.tar -C /workspace/SleepTCN
python scripts/run_amendment_a4.py verify-bundle   # BUNDLE VERIFIED
python scripts/run_amendment_a4.py env-check --device cuda   # ENVIRONMENT OK
nvidia-smi > runs/a4/monitoring/gpu_info.txt
python -m pip freeze > runs/a4/monitoring/pip_freeze.txt
python scripts/run_amendment_a4.py plan            # locked_before_a4_training, 40 runs missing
git status --short                                 # vẫn phải trống
```

### II.5. Kiểm tra tái lập (khoảng 20 phút)

```bash
python scripts/run_amendment_a4.py repro --device cuda
```

In `IDENTICAL` hoặc `DIFFERS`; cả hai đều tiếp tục và được báo cáo nguyên trạng. Chỉ dừng nếu lệnh lỗi.

### II.6. Huấn luyện (khoảng 6–7 giờ)

```bash
nohup python scripts/run_amendment_a4.py train --device cuda --workers 3 --num-workers 1 > runs/a4/monitoring/train.log 2>&1 &
python scripts/run_amendment_a4.py status          # xem tiến độ bất cứ lúc nào
```

Nếu mất kết nối hay một tiến trình lỗi, chạy lại đúng lệnh `train`: các lần chạy đã xong được bỏ qua, lần chạy dở
được tiếp tục từ checkpoint.

### II.7. Mở tập test Sleep-EDF một lần

```bash
python scripts/run_amendment_a4.py test                       # preflight, không đọc dữ liệu test
python scripts/run_amendment_a4.py test --device cuda --confirm OPEN-A4-LOCKED-TEST-ONCE
```

### II.8. Xuất kết quả và tải về

```bash
python scripts/run_amendment_a4.py export     # ghi /workspace/a4_results.tar.gz (checkpoint, dự đoán, log)
sha256sum /workspace/a4_results.tar.gz
```

Tải `a4_results.tar.gz` về máy local (Jupyter/SFTP của nhà cung cấp, hoặc tải lên Drive bằng `rclone`). Kiểm tra
mã SHA-256 trước khi tắt máy thuê.

## Phần III. Máy local sau khi có kết quả

```powershell
tar -xzf ..\a4_results.tar.gz -C .
python scripts/run_amendment_a4.py status      # 40 runs tested
python scripts/a4_infer.py --cohort shhs1_holdout
python scripts/a4_infer.py --cohort telemetry
python scripts/a4_analyze.py
```

## Lỗi thường gặp

| Lỗi | Nguyên nhân |
|---|---|
| `git_worktree_not_clean` | Có tệp lạ trong repo (kể cả tệp tar); đặt tệp tải về ngoài `/workspace/SleepTCN` |
| `a4_config_not_locked` | Clone trước khi commit khóa A4; `git pull` |
| `git_history_without_8af1d12...` | Clone nông; clone lại không có `--depth` |
| `export CUBLAS_WORKSPACE_CONFIG...` | Chưa làm II.3 trong terminal hiện tại |
