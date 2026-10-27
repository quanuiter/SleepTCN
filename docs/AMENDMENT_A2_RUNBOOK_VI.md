# Runbook Amendment A2

Protocol: `docs/SHHS_HOLDOUT_AMENDMENT_A2.md`. Cấu hình khóa: `configs/shhs_holdout_amendment_a2.json`.
Lệnh điều phối: `scripts/run_amendment_a2.py`. Phân tích: `scripts/analyze_amendment_a2.py`.

## Tổng quan

Mọi bước tính bằng mạng nơ-ron (tái lập, train, mở test, suy luận holdout, đo thời gian mỗi bước) chạy trên
GPU thuê. Tiền xử lý (lọc tín hiệu, việc của CPU) chạy ở máy local trước khi thuê. Phân tích D chạy ở máy local
**trong lúc** GPU đang train; các phân tích còn lại (bootstrap numpy) chạy ở máy local sau khi tải kết quả về.

| Chặng | Việc | Thời gian ước lượng |
|---|---|---|
| I. Local, trước khi thuê | Khóa config, tiền xử lý Sleep-EDF và SHHS, đóng gói | khoảng 45 phút |
| II. GPU thuê | Tái lập, 50 lần train, mở test, suy luận holdout, đo thời gian mỗi bước | khoảng 9–10 giờ |
| (song song với II) | Phân tích D ở máy local; tải gói holdout lên GPU | không cộng thêm |
| III. Local, sau khi tải về | Phân tích C, A, B, E | khoảng 1 giờ |

Mọi lệnh chạy từ gốc repo. Ở máy Windows, dùng Python của `.venv` và đặt `PYTHONIOENCODING=utf-8`. Mọi lệnh tạo
dữ liệu, huấn luyện, suy luận hay phân tích đều **từ chối chạy** khi config chưa khóa. Mọi manifest và báo cáo
phân tích đều chỉ ghi một lần; không xóa chúng để chạy lại.

## I. Máy local, trước khi thuê GPU

### I.1. Khóa config

Trong `configs/shhs_holdout_amendment_a2.json`, chỉ đổi `"status": "draft"` thành
`"status": "locked_before_a2_training"`. Commit với thông điệp "Lock Amendment A2", rồi push. Ghi lại mã commit
khóa. Từ đây không sửa config nữa.

### I.2. Tiền xử lý (chạy hai lệnh song song trong hai terminal, khoảng 20 phút)

```bash
python scripts/run_amendment_a2.py prepare-sleepedf \
    --data-dir E:/research/Dataset/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette --workers 4
```

```bash
python scripts/run_amendment_a2.py prepare-shhs \
    --parent-manifest E:/research/Dataset/SHHS_v1/manifests/preprocess_holdout_v1.json \
    --technical-audit E:/research/Dataset/SHHS_v1/manifests/holdout_audit_v1.json \
    --edf-dir E:/research/Dataset/SHHS_v1/shhs/polysomnography/edfs/shhs1 \
    --xml-dir E:/research/Dataset/SHHS_v1/shhs/polysomnography/annotations-events-profusion/shhs1 \
    --workers 4
```

- `prepare-sleepedf` ghi 153 tệp vào `data/processed/filtered_zscore_rolling90_v3`, so từng bản ghi với
  `filtered_v2`, tính lại `filtered_zscore_v2` cho ba bản ghi để kiểm tra pipeline lọc (sai số tối đa 1e-4), rồi
  ghi `runs/a2/manifests/preprocess_sleepedf_a2.json`. Dòng cuối phải là `SLEEP-EDF A2 VARIANT WRITTEN AND CHECKED`.
- `prepare-shhs` kiểm tra hash của manifest gốc, audit kỹ thuật và config tiền xử lý; ghi 500 tệp vào
  `data/processed/shhs_holdout_a2`; so từng bản ghi với `bandpass_v2` gốc (thêm `raw_stage` và hash nguồn); ghi
  `runs/a2/manifests/preprocess_holdout_a2.json` cùng bản sao nguyên byte của manifest gốc. A2 không có quy tắc
  thay người: một bản ghi lỗi làm lệnh dừng. Dòng cuối phải là `HOLDOUT A2 VARIANT WRITTEN AND CHECKED`.

### I.3. Đóng gói hai phần

```bash
python scripts/run_amendment_a2.py bundle --part sleepedf
python scripts/run_amendment_a2.py bundle --part holdout
sha256sum ../a2_inputs_sleepedf.tar ../a2_inputs_holdout.tar
```

- `a2_inputs_sleepedf.tar` (khoảng 4,4 GB): đủ cho tái lập, train và mở test. Tải lên trước.
- `a2_inputs_holdout.tar` (khoảng 10,5 GB): dữ liệu holdout và tệp E4_s42 gốc cho bước kiểm tra thiết bị. Tải lên
  **trong lúc GPU đang train**.

Ghi lại hai mã SHA-256.

## II. GPU thuê

Chặng này viết cho máy **Docker Linux, Tesla V100-PCIE-16GB, 6 nhân CPU, 30 GB RAM** (cùng loại GPU với chiến dịch
gốc). Máy thuê chạy **Windows**: dùng `docs/AMENDMENT_A2_RUNBOOK_WINDOWS_VI.md` thay cho chặng II.

Đường dẫn làm việc: `/workspace`. Không dừng hoặc xóa container trước khi đã tải `a2_results.tar.gz` về (II.11).

### II.1. Mã nguồn

Cần **toàn bộ lịch sử Git** (không dùng `--depth`): kiểm tra tái lập đọc checkpoint ở commit `8af1d12`. Tag
`a2-gpu-run` trỏ tới commit công cụ cho máy GPU; `env-check` (II.3) xác nhận config A2 trùng với commit khóa
`38eb6dc` ("Lock Amendment A2").

```bash
nvidia-smi                  # phải thấy Tesla V100-PCIE-16GB
cd /workspace
git clone https://github.com/quanuiter/SleepTCN.git SleepTCN
cd /workspace/SleepTCN
git checkout a2-gpu-run
git status --short          # phải trống
git cat-file -t 8af1d12     # phải in "commit"
```

Không sửa tệp nào trong repo trên máy GPU. Runner và bước mở test đều từ chối chạy khi worktree không sạch. Mọi đầu
ra A2 (`runs/`, `data/processed/`, `data/cache/`) và `.venv/` đều nằm trong `.gitignore`.

### II.2. Môi trường (Python 3.11 như chiến dịch gốc)

File khóa `requirements/lock-cu121.txt` được chụp từ Python 3.11. Nếu `python3 --version` không phải 3.11.x (ví dụ
Docker có sẵn 3.10), tạo venv bằng `uv`: uv tự tải một bản Python 3.11 độc lập (không cần apt, không đụng Python hệ
thống) và cài sẵn `pip` vào venv. uv được cài vào một venv công cụ riêng ngoài repo:

```bash
python3 -m venv /workspace/uvtool
/workspace/uvtool/bin/python -m pip install --quiet "uv==0.5.0"
cd /workspace/SleepTCN
/workspace/uvtool/bin/python -m uv venv --python-preference only-managed --python 3.11 --seed .venv
```

Nếu máy đã có sẵn Python 3.11 thì có thể thay bằng `python3 -m venv .venv`. Sau đó cài thư viện bằng `pip` như bình
thường:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/lock-cu121.txt
python -m pip install -e . --no-deps
python -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Dòng cuối phải in `3.11.x 2.5.1+cu121 12.1 True Tesla V100-PCIE-16GB`.

### II.3. Mỗi terminal mới

Mỗi terminal mới (kể cả sau khi mất kết nối) đều phải chạy lại:

```bash
cd /workspace/SleepTCN
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Lần đầu, kiểm tra môi trường và lưu thông tin máy:

```bash
python scripts/run_amendment_a2.py env-check --device cuda
nvidia-smi > runs/a2/monitoring/gpu_info.txt
python -m pip freeze > runs/a2/monitoring/pip_freeze.txt
```

`env-check` phải in `ENVIRONMENT OK`: Python 3.10/3.11, torch 2.5.1 với CUDA 12.1, hash file split khớp sidecar,
config A2 đã khóa và trùng commit khóa, worktree sạch, có commit `8af1d12`.

### II.4. Đưa dữ liệu lên

Docker không có copy-paste như Remote Desktop. Cách dễ nhất: ở máy local, tải hai tệp `D:\a2_inputs_sleepedf.tar`
và `D:\a2_inputs_holdout.tar` lên Google Drive, đặt chia sẻ "Bất kỳ ai có đường liên kết", rồi trên máy GPU:

```bash
cd /workspace
python -m pip install --quiet gdown
gdown --fuzzy "<link Google Drive của a2_inputs_sleepedf.tar>" -O /workspace/a2_inputs_sleepedf.tar
sha256sum /workspace/a2_inputs_sleepedf.tar
```

Mã phải bằng `2d74180cc118ed3d5bcc631653b042f34dee5ecac2919a8394f3662ee8e42107`. Nếu khác, tệp bị hỏng khi chuyển:
tải lại. Nếu `gdown` báo vượt hạn mức tải của Google Drive, đợi vài phút rồi thử lại, hoặc dùng chức năng tải tệp của
nhà cung cấp (Jupyter, SFTP) nếu có. Tệp holdout (II.6) tải theo cùng cách, trong lúc train chạy.

### II.5. Dữ liệu Sleep-EDF

```bash
cd /workspace/SleepTCN
tar -xf /workspace/a2_inputs_sleepedf.tar -C /workspace/SleepTCN
python scripts/run_amendment_a2.py verify-bundle --part sleepedf
python scripts/run_amendment_a2.py plan
git status --short          # vẫn phải trống
```

`verify-bundle` phải in `BUNDLE VERIFIED (sleepedf)`. `plan` phải in trạng thái `locked_before_a2_training`, 50 lần
chạy `missing` và kiểm tra tái lập `missing`.

### II.6. Kiểm tra tái lập (khoảng 20 phút)

```bash
python scripts/run_amendment_a2.py repro --device cuda
```

Kết quả ghi vào `runs/a2/reproduction_check.json`; dòng cuối in `IDENTICAL` hoặc `DIFFERS`. Cả hai trường hợp đều
tiếp tục và được báo cáo nguyên trạng. Chỉ dừng nếu lệnh **lỗi**.

### II.7. Huấn luyện 50 lần chạy (khoảng 7–8 giờ)

Máy có 6 nhân CPU: chạy 3 tiến trình train, mỗi tiến trình 1 worker nạp dữ liệu (tổng 6 tiến trình). Số worker nạp
dữ liệu không ảnh hưởng kết quả số.

```bash
nohup python scripts/run_amendment_a2.py train --device cuda --workers 3 --num-workers 1 \
    > runs/a2/monitoring/train_driver.log 2>&1 &
tail -f runs/a2/monitoring/train_worker_0.log      # Ctrl+C để thoát xem, không dừng train
python scripts/run_amendment_a2.py status
```

- GPU hết bộ nhớ: dừng, chạy lại với `--workers 2`.
- Bị ngắt (mất kết nối terminal): mở terminal mới, làm II.3, xem `status`; train vẫn chạy nền nhờ `nohup`. Nếu
  container khởi động lại: làm II.3 rồi chạy lại đúng lệnh `train`. Lần chạy xong được bỏ qua, lần chạy dở được tiếp
  tục từ checkpoint. Không xóa thư mục lần chạy, không đổi seed. Không chạy hai lệnh `train` cùng lúc.
- Khi xong: `train_driver.log` có `ALL A2 TRAINING RUNS FINISHED`, `status` cho 50 lần chạy `trained`.

**Trong lúc train chạy, làm song song:**

1. Ở máy local: chạy phân tích D (khoảng 1,5 giờ CPU): `python scripts/analyze_amendment_a2.py d`.
2. Trên máy GPU, ở terminal khác (sau II.3): tải tệp holdout như II.4, kiểm tra, giải nén:

   ```bash
   gdown --fuzzy "<link Google Drive của a2_inputs_holdout.tar>" -O /workspace/a2_inputs_holdout.tar
   sha256sum /workspace/a2_inputs_holdout.tar
   tar -xf /workspace/a2_inputs_holdout.tar -C /workspace/SleepTCN
   python scripts/run_amendment_a2.py verify-bundle --part holdout
   git status --short          # vẫn phải trống
   ```

   Mã phải bằng `5314831d653f960f2ee9d97b91ce350f2e7f54cb2886772edaac31c7c02487f4`.

### II.8. Mở tập test Sleep-EDF (một lần duy nhất)

```bash
python scripts/run_amendment_a2.py test
python scripts/run_amendment_a2.py test --device cuda --confirm OPEN-A2-LOCKED-TEST-ONCE
```

Lệnh đầu chỉ kiểm tra, không đọc dữ liệu test; phải in `Preflight passed for 50 runs`. Nhật ký:
`runs/a2/test_campaign_a2.json`. Nếu bị ngắt, chạy lại đúng lệnh thứ hai với cùng commit. Không bao giờ xóa nhật ký
này.

### II.9. Suy luận holdout trên GPU (khoảng 30–45 phút)

```bash
python scripts/run_amendment_a2.py holdout-check --device cuda
python scripts/run_amendment_a2.py holdout --device cuda --workers 2 --threads 2
```

`holdout-check` tái tạo xác suất fold CPU đã lưu của E4_s42 (hai bản ghi, fold 0 và 9) bằng code A2 trên GPU;
phải in `DEVICE CHECK PASSED`. Nếu **không** qua, không chạy `holdout` trên GPU: chạy hai lệnh trên ở máy local
với `--device cpu --workers 6` sau khi tải kết quả về (III.1). Lệnh `holdout` chỉ chạy sau khi kiểm tra thiết bị
qua và chiến dịch test Sleep-EDF đã hoàn tất. Nếu bị ngắt, chạy lại đúng lệnh; tệp fold đã ghi được giữ. Khi xong:
`A2 HOLDOUT INFERENCE COMPLETE`.

### II.10. Đo thời gian mỗi bước (phân tích E, khoảng 15–30 phút)

Chạy khi GPU không còn tiến trình nào khác:

```bash
nvidia-smi                  # không còn tiến trình python
python scripts/run_amendment_a2.py step-benchmark --device cuda
```

Ghi `runs/a2_step_time_benchmark.json`. Vì GPU là V100 như chiến dịch gốc, phân tích E tính được cả phần thời gian
còn lại theo giây.

### II.11. Đóng gói kết quả và tải về

```bash
python scripts/run_amendment_a2.py export --output /workspace/a2_results.tar.gz
sha256sum /workspace/a2_results.tar.gz
```

Tải tệp về (qua chức năng tải của nhà cung cấp, hoặc tải lên Google Drive từ máy GPU), kiểm tra lại mã SHA-256 sau
khi tải, rồi mới dừng container.

## III. Máy local, sau khi tải kết quả về

### III.1. Giải nén kết quả

```bash
tar -xzf ../a2_results.tar.gz -C .
python scripts/run_amendment_a2.py status
```

Phải thấy 50 lần chạy ở trạng thái `tested`, và `runs/a2/holdout/run_manifest.json` tồn tại.

### III.2. Phân tích C, A, B, E

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
| `A2 config status is 'draft'` | Chưa checkout tag `a2-gpu-run` (II.1) |
| `export CUBLAS_WORKSPACE_CONFIG...` | Chưa làm II.3 trong terminal hiện tại |
| `ModuleNotFoundError` / `No module named torch` | Chưa kích hoạt venv: làm II.3 |
| `ENVIRONMENT NOT READY` | Đọc danh sách `errors`; thường do sai commit (II.1), sai Python/torch (II.2) hoặc giải nén sai chỗ (II.5) |
| `full experiment requires a clean Git worktree` | Có tệp bị sửa hoặc tệp lạ trong repo; `git status --short` để xem |
| `Missing split NPZ files` | Thiếu dữ liệu ở II.5 hoặc giải nén sai thư mục |
| `... is write-once` | Bước đó đã chạy xong; không chạy lại |
| `differs from filtered_v2` / `bandpass_v2` | Dữ liệu nguồn hoặc pipeline khác bản gốc: dừng và báo lại, không bỏ qua |
| `DEVICE CHECK FAILED` | Không suy luận holdout trên thiết bị đó; dùng CPU ở máy local (II.9) |
| CUDA out of memory | Giảm `--workers` |
| Một worker báo `FAILED` | Đọc log của worker, sửa nguyên nhân môi trường, chạy lại cùng lệnh |
