# Runbook chạy Amendment A2 trên GPU thuê

Protocol: `docs/SHHS_HOLDOUT_AMENDMENT_A2.md`. Cấu hình khóa: `configs/shhs_holdout_amendment_a2.json`.
Script điều phối: `scripts/run_amendment_a2.py`.

> Các mục đánh dấu **[bước 3]** dùng script sẽ được viết ở bước 3 của kế hoạch A2. Chưa chạy các mục đó
> cho tới khi script có trong repo và runbook được cập nhật.

## 0. Trước khi thuê GPU (ở máy local)

1. Khóa config: chỉ đổi `"status": "draft"` thành `"status": "locked_before_a2_training"` trong
   `configs/shhs_holdout_amendment_a2.json`, commit, push. Ghi lại mã commit khóa.
2. **[bước 3]** Tiền xử lý Sleep-EDF biến thể `filtered_zscore_rolling90_v3` từ
   `E:/research/Dataset/physionet.org` vào `data/processed/filtered_zscore_rolling90_v3` và chạy bước kiểm
   tra tiền xử lý A2 (so nhãn và chỉ số epoch với `filtered_v2`).
3. **[bước 3]** Tiền xử lý SHHS1 holdout biến thể `filtered_zscore_rolling90_v3` vào
   `data/processed/shhs_holdout_a2` kèm manifest A2.
4. Chuẩn bị gói dữ liệu để tải lên (tên thư mục giữ nguyên tương đối với gốc repo):
   - `data/processed/bandpass_v2/` (153 tệp, khoảng 2,2 GB): cho E4 và kiểm tra tái lập;
   - `data/processed/filtered_zscore_rolling90_v3/` (153 tệp, khoảng 2,2 GB): cho E6R;
   - **[bước 3]** dữ liệu holdout cho suy luận trên GPU (biến thể `bandpass_v2` của manifest gốc và
     manifest A2), cùng xác suất fold CPU đã lưu của E4_s42 cho hai bản ghi ở fold 0 và 9 để kiểm tra
     CUDA so với CPU.

## 1. Máy GPU

- Linux, Python 3.11, GPU NVIDIA. Ưu tiên Tesla V100 để so được với chiến dịch gốc; GPU khác vẫn dùng
  được (phân tích E so bằng tỷ số).
- Ổ đĩa trống tối thiểu 40 GB: repo đầy đủ lịch sử khoảng 3,5 GB, dữ liệu Sleep-EDF khoảng 4,4 GB, dữ liệu
  holdout, cache đặc trưng và 51 thư mục lần chạy.
- Ít nhất 8 lõi CPU nếu chạy 3 tiến trình train, mỗi tiến trình 2 DataLoader worker.

## 2. Lấy mã nguồn

Cần **toàn bộ lịch sử Git** (không dùng `--depth`), vì kiểm tra tái lập đọc lần chạy lưu trữ ở commit
`8af1d12`.

```bash
cd /workspace
git clone https://github.com/quanuiter/SleepTCN.git SleepTCN
cd /workspace/SleepTCN
git checkout <mã commit khóa A2>
git status --short          # phải trống
git cat-file -t 8af1d12     # phải in "commit"
```

Không sửa tệp nào trong repo trên máy GPU. Runner và bước mở test đều từ chối chạy khi worktree không
sạch. Mọi đầu ra của A2 (`runs/a2`, `data/cache`, `data/processed`) đã nằm trong `.gitignore`.

## 3. Môi trường

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/lock-cu121.txt
python -m pip install -e . --no-deps
mkdir -p runs/a2/monitoring
python scripts/check_environment.py --workspace . --require-gpu --output runs/a2/monitoring/environment_check.json
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Phải thấy `2.5.1+cu121` và `True`. Lưu lại thông tin GPU:

```bash
nvidia-smi > runs/a2/monitoring/gpu_info.txt
python -m pip freeze > runs/a2/monitoring/pip_freeze.txt
```

## 4. Biến môi trường bắt buộc

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Phải export trong **mọi** terminal trước khi chạy `repro`, `train` hoặc `test`. Script dừng ngay nếu thiếu.

## 5. Tải dữ liệu lên và kiểm tra

Giải nén gói dữ liệu ở bước 0.4 vào gốc repo, rồi kiểm tra:

```bash
ls data/processed/bandpass_v2 | wc -l                    # 153
ls data/processed/filtered_zscore_rolling90_v3 | wc -l   # 153
python scripts/run_amendment_a2.py plan
```

`plan` phải in `A2 config status: locked_before_a2_training`, 50 lần chạy ở trạng thái `missing` và
kiểm tra tái lập `missing`.

## 6. Kiểm tra tái lập (khoảng 20 phút)

```bash
python scripts/run_amendment_a2.py repro --device cuda
```

Kết quả ghi vào `runs/a2/reproduction_check.json`. Dòng cuối in `IDENTICAL` hoặc `DIFFERS`. Cả hai trường
hợp đều tiếp tục; kết quả được báo cáo nguyên trạng. Chỉ dừng lại nếu lệnh **lỗi**.

## 7. Huấn luyện 50 lần chạy

```bash
nohup python scripts/run_amendment_a2.py train --device cuda --workers 3 --num-workers 2 \
    > runs/a2/monitoring/train_driver.log 2>&1 &
```

Theo dõi:

```bash
tail -f runs/a2/monitoring/train_worker_0.log
nvidia-smi
python scripts/run_amendment_a2.py status
```

- Ước lượng: khoảng 15 GPU-giờ, tức 6–7 giờ thực với 3 tiến trình trên V100.
- Nếu GPU hết bộ nhớ, dừng và chạy lại với `--workers 2`.
- **Bị ngắt giữa chừng** (mất kết nối, máy khởi động lại): export lại biến ở mục 4 rồi chạy lại đúng lệnh
  `train`. Lần chạy đã xong được bỏ qua, lần chạy dở được tiếp tục từ checkpoint. Không xóa thư mục lần
  chạy, không đổi seed.
- Khi xong, driver in `ALL A2 TRAINING RUNS FINISHED` và bảng trạng thái: cả 50 lần chạy phải là `trained`.

## 8. Mở tập test Sleep-EDF (một lần duy nhất)

Kiểm tra trước, không đọc dữ liệu test:

```bash
python scripts/run_amendment_a2.py test
```

Nếu in `Preflight passed for 50 runs`, mở test:

```bash
python scripts/run_amendment_a2.py test --device cuda --confirm OPEN-A2-LOCKED-TEST-ONCE
```

Nhật ký: `runs/a2/test_campaign_a2.json`. Nếu bị ngắt, chạy lại đúng lệnh trên với cùng commit; các lần
chạy đã có kết quả test được bỏ qua. Không bao giờ xóa nhật ký này để chạy lại từ đầu.

## 9. **[bước 3]** Đo thời gian mỗi bước (phân tích E, khoảng 30 phút)

Chạy khi GPU không còn tiến trình nào khác. Lệnh sẽ được bổ sung.

## 10. **[bước 3]** Suy luận holdout SHHS1 cho năm cặp arm–seed mới

Gồm bước kiểm tra CUDA so với CPU trên E4_s42 trước khi chạy. Lệnh sẽ được bổ sung.

## 11. Đóng gói và tải về

```bash
python scripts/run_amendment_a2.py export --output /workspace/a2_results.tar.gz
sha256sum /workspace/a2_results.tar.gz
```

Tải tệp về máy local cùng mã SHA-256, kiểm tra lại mã sau khi tải, rồi mới tắt máy GPU.

## 12. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `A2 config status is 'draft'` | Chưa checkout đúng commit khóa (mục 2) |
| `export CUBLAS_WORKSPACE_CONFIG...` | Chưa làm mục 4 trong terminal hiện tại |
| `full experiment requires a clean Git worktree` | Có tệp bị sửa hoặc tệp lạ trong repo; `git status --short` để xem |
| `Missing split NPZ files` | Thiếu dữ liệu ở mục 5 hoặc giải nén sai thư mục |
| CUDA out of memory | Giảm `--workers` |
| Một worker báo `FAILED` | Đọc `train_worker_<n>.log`, sửa nguyên nhân môi trường, chạy lại `train` (mục 7) |
