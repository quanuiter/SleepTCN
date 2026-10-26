# Runbook Amendment A2

Protocol: `docs/SHHS_HOLDOUT_AMENDMENT_A2.md`. Cấu hình khóa: `configs/shhs_holdout_amendment_a2.json`.
Lệnh điều phối: `scripts/run_amendment_a2.py`. Phân tích: `scripts/analyze_amendment_a2.py`.

Quy trình gồm ba chặng: **(I) máy local trước khi thuê GPU**, **(II) GPU thuê** (chỉ dữ liệu Sleep-EDF),
**(III) máy local sau khi tải kết quả về**. Mọi lệnh chạy từ gốc repo. Ở máy Windows, dùng Python của
`.venv` và đặt `PYTHONIOENCODING=utf-8`.

Mọi lệnh tạo dữ liệu, huấn luyện, suy luận hay phân tích đều **từ chối chạy** khi config chưa khóa. Mọi
manifest và báo cáo phân tích đều chỉ ghi một lần; không xóa chúng để chạy lại.

## I. Máy local (trước khi thuê GPU)

### I.1. Khóa config

Trong `configs/shhs_holdout_amendment_a2.json`, chỉ đổi `"status": "draft"` thành
`"status": "locked_before_a2_training"`. Commit với thông điệp "Lock Amendment A2", rồi push. Ghi lại mã
commit khóa. Từ đây không sửa config nữa.

### I.2. Tiền xử lý Sleep-EDF (khoảng 10 phút)

```bash
python scripts/run_amendment_a2.py prepare-sleepedf \
    --data-dir E:/research/Dataset/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette --workers 4
```

Lệnh ghi 153 tệp vào `data/processed/filtered_zscore_rolling90_v3`, so từng bản ghi với `filtered_v2` (khóa,
nhãn, mặt nạ hợp lệ, chỉ số epoch), tính lại `filtered_zscore_v2` cho ba bản ghi để kiểm tra pipeline lọc
của máy này (sai số tối đa 1e-4), rồi ghi `runs/a2/manifests/preprocess_sleepedf_a2.json`. Dòng cuối phải là
`SLEEP-EDF A2 VARIANT WRITTEN AND CHECKED`.

### I.3. Tiền xử lý SHHS1 holdout (khoảng 10–20 phút)

```bash
python scripts/run_amendment_a2.py prepare-shhs \
    --parent-manifest E:/research/Dataset/SHHS_v1/manifests/preprocess_holdout_v1.json \
    --technical-audit E:/research/Dataset/SHHS_v1/manifests/holdout_audit_v1.json \
    --edf-dir E:/research/Dataset/SHHS_v1/shhs/polysomnography/edfs/shhs1 \
    --xml-dir E:/research/Dataset/SHHS_v1/shhs/polysomnography/annotations-events-profusion/shhs1 \
    --workers 4
```

Lệnh kiểm tra hash của manifest gốc, audit kỹ thuật và config tiền xử lý; ghi 500 tệp vào
`data/processed/shhs_holdout_a2`; so từng bản ghi với `bandpass_v2` gốc (thêm `raw_stage` và hash nguồn);
ghi `runs/a2/manifests/preprocess_holdout_a2.json` và bản sao nguyên byte của manifest gốc
(`runs/a2/manifests/parent_preprocess_holdout_v1.json`). A2 không có quy tắc thay người: một bản ghi lỗi làm
lệnh dừng.

### I.4. Phân tích D (CPU, khoảng 1,5 giờ; chạy được ngay)

```bash
python scripts/analyze_amendment_a2.py d
```

Ghi `runs/a2/analysis/D_em_sample_size.json`.

### I.5. Đóng gói dữ liệu cho GPU

```bash
python scripts/run_amendment_a2.py bundle
```

Tạo `../a2_inputs.tar` (khoảng 4,4 GB): hai biến thể Sleep-EDF và manifest Sleep-EDF A2. Ghi lại mã SHA-256
của tệp (`sha256sum ../a2_inputs.tar`).

## II. GPU thuê

### II.1. Máy

Linux, Python 3.11, GPU NVIDIA; ưu tiên Tesla V100 để so được với chiến dịch gốc (phân tích E vẫn so bằng tỷ
số nếu GPU khác). Ổ trống tối thiểu 40 GB. Ít nhất 8 lõi CPU nếu chạy 3 tiến trình train.

### II.2. Mã nguồn

Cần **toàn bộ lịch sử Git** (không dùng `--depth`): kiểm tra tái lập và phân tích E đọc checkpoint ở commit
`8af1d12`.

```bash
cd /workspace
git clone https://github.com/quanuiter/SleepTCN.git SleepTCN
cd /workspace/SleepTCN
git checkout <mã commit khóa A2>
git status --short          # phải trống
git cat-file -t 8af1d12     # phải in "commit"
```

Không sửa tệp nào trong repo trên máy GPU. Runner và bước mở test đều từ chối chạy khi worktree không sạch.
Mọi đầu ra A2 (`runs/`, `data/processed/`, `data/cache/`) đều nằm trong `.gitignore`.

### II.3. Môi trường

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/lock-cu121.txt
python -m pip install -e . --no-deps
mkdir -p runs/a2/monitoring
python scripts/check_environment.py --workspace . --require-gpu --output runs/a2/monitoring/environment_check.json
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi > runs/a2/monitoring/gpu_info.txt
python -m pip freeze > runs/a2/monitoring/pip_freeze.txt
```

Phải thấy `2.5.1+cu121` và `True`.

### II.4. Biến môi trường bắt buộc

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Phải export trong **mọi** terminal trước `repro`, `train`, `test` và `step-benchmark`.

### II.5. Tải dữ liệu lên và kiểm tra

Tải `a2_inputs.tar` lên, kiểm tra SHA-256 khớp với mã đã ghi ở I.5, rồi giải nén **tại gốc repo**:

```bash
tar -xf /workspace/a2_inputs.tar -C /workspace/SleepTCN
python scripts/run_amendment_a2.py verify-bundle
python scripts/run_amendment_a2.py plan
git status --short          # vẫn phải trống
```

`verify-bundle` phải in `BUNDLE VERIFIED`. `plan` phải in trạng thái `locked_before_a2_training`, 50 lần chạy
`missing` và kiểm tra tái lập `missing`.

### II.6. Kiểm tra tái lập (khoảng 20 phút)

```bash
python scripts/run_amendment_a2.py repro --device cuda
```

Kết quả ghi vào `runs/a2/reproduction_check.json`; dòng cuối in `IDENTICAL` hoặc `DIFFERS`. Cả hai trường hợp đều
tiếp tục và được báo cáo nguyên trạng. Chỉ dừng nếu lệnh **lỗi**.

### II.7. Huấn luyện 50 lần chạy (khoảng 6–7 giờ với 3 tiến trình trên V100)

```bash
nohup python scripts/run_amendment_a2.py train --device cuda --workers 3 --num-workers 2 \
    > runs/a2/monitoring/train_driver.log 2>&1 &
tail -f runs/a2/monitoring/train_worker_0.log
python scripts/run_amendment_a2.py status
```

- GPU hết bộ nhớ: dừng, chạy lại với `--workers 2`.
- Bị ngắt (mất kết nối, khởi động lại): export lại biến ở II.4 và chạy lại đúng lệnh `train`. Lần chạy xong được
  bỏ qua, lần chạy dở được tiếp tục từ checkpoint. Không xóa thư mục lần chạy, không đổi seed.
- Khi xong: `ALL A2 TRAINING RUNS FINISHED`, cả 50 lần chạy ở trạng thái `trained`.

### II.8. Mở tập test Sleep-EDF (một lần duy nhất)

```bash
python scripts/run_amendment_a2.py test
python scripts/run_amendment_a2.py test --device cuda --confirm OPEN-A2-LOCKED-TEST-ONCE
```

Lệnh đầu chỉ kiểm tra, không đọc dữ liệu test; phải in `Preflight passed for 50 runs`. Nhật ký:
`runs/a2/test_campaign_a2.json`. Nếu bị ngắt, chạy lại đúng lệnh thứ hai với cùng commit. Không bao giờ xóa nhật
ký này.

### II.9. Đo thời gian mỗi bước (phân tích E, khoảng 15–30 phút)

Chạy khi GPU không còn tiến trình nào khác:

```bash
nvidia-smi                  # không còn tiến trình python
python scripts/run_amendment_a2.py step-benchmark --device cuda
```

Ghi `runs/a2_step_time_benchmark.json`.

### II.10. Đóng gói kết quả và tải về

```bash
python scripts/run_amendment_a2.py export --output /workspace/a2_results.tar.gz
sha256sum /workspace/a2_results.tar.gz
```

Tải tệp về cùng mã SHA-256, kiểm tra lại mã sau khi tải, rồi mới tắt máy GPU.

## III. Máy local (sau khi tải kết quả về)

### III.1. Giải nén kết quả

```bash
tar -xzf ../a2_results.tar.gz -C .
python scripts/run_amendment_a2.py status
```

Phải thấy 50 lần chạy ở trạng thái `tested`.

### III.2. Suy luận holdout trên CPU (khoảng 1–2 giờ với 6 tiến trình)

```bash
python scripts/run_amendment_a2.py holdout-check --device cpu
python scripts/run_amendment_a2.py holdout --device cpu --workers 6 --threads 2
```

`holdout-check` tái tạo xác suất fold CPU đã lưu của E4_s42 (hai bản ghi, fold 0 và 9) bằng code A2; phải in
`DEVICE CHECK PASSED`. Lệnh `holdout` chỉ chạy sau khi kiểm tra này qua và chiến dịch test Sleep-EDF đã hoàn tất.
Nếu bị ngắt, chạy lại đúng lệnh; tệp fold đã ghi được giữ. Khi xong: `A2 HOLDOUT INFERENCE COMPLETE`.

Tùy chọn chạy trên GPU thuê: `bundle --with-holdout` ở I.5 (thêm khoảng 10 GB dữ liệu holdout và checkpoint
E4_s42 fold 0/9 cho bước kiểm tra), rồi chạy hai lệnh trên với `--device cuda` sau II.8. Bước kiểm tra là bắt
buộc với mọi thiết bị.

### III.3. Phân tích A, B, C, E

```bash
python scripts/analyze_amendment_a2.py c
python scripts/analyze_amendment_a2.py a
python scripts/analyze_amendment_a2.py b
python scripts/analyze_amendment_a2.py e
```

Mỗi lệnh ghi một báo cáo vào `runs/a2/analysis/` và in kết luận của quy tắc diễn giải đã khóa. Bản thảo được cập
nhật đúng theo các kết luận đó.

## Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `A2 config status is 'draft'` | Chưa khóa config (I.1) hoặc chưa checkout commit khóa (II.2) |
| `export CUBLAS_WORKSPACE_CONFIG...` | Chưa làm II.4 trong terminal hiện tại |
| `full experiment requires a clean Git worktree` | Có tệp bị sửa hoặc tệp lạ trong repo; `git status --short` để xem |
| `Missing split NPZ files` | Thiếu dữ liệu ở II.5 hoặc giải nén sai thư mục |
| `... is write-once` | Bước đó đã chạy xong; không chạy lại |
| `differs from filtered_v2` / `bandpass_v2` | Dữ liệu nguồn hoặc pipeline khác bản gốc: dừng và báo lại, không bỏ qua |
| CUDA out of memory | Giảm `--workers` |
| Một worker báo `FAILED` | Đọc log của worker, sửa nguyên nhân môi trường, chạy lại cùng lệnh |
