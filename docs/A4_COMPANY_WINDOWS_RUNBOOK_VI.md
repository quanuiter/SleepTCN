# A4 trên máy Windows có GPU (máy công ty), phần SHHS chạy ở máy cá nhân

Máy công ty **chỉ dùng dữ liệu Sleep-EDF** (công khai): huấn luyện 31 lần chạy A4 còn lại và mở tập test Sleep-EDF.
Dữ liệu SHHS (NSRR) **không** được đưa lên máy công ty; suy luận trên SHHS1 và Telemetry chạy ở máy cá nhân bằng
CPU sau khi chép kết quả về. Tiến độ 9/40 lần chạy và kiểm tra tái lập đã làm trên Colab được dùng tiếp.

Hỏi IT/quản lý trước khi chạy GPU nhiều giờ trên máy công ty.

## Phần A. Máy công ty (Windows, GPU NVIDIA, ví dụ RTX A3000 6 GB)

### A1. Kiểm tra máy (PowerShell)

```powershell
py -3.11 --version        # cần Python 3.11 (cài từ python.org nếu chưa có, tick "py launcher")
git --version
nvidia-smi                # phải thấy GPU NVIDIA
```

Đặt Windows không ngủ khi cắm điện (Settings → System → Power → Sleep: Never) và cắm sạc nếu là laptop.

### A2. Mã nguồn và môi trường (khoảng 15 phút)

```powershell
mkdir C:\SleepTCN_work; cd C:\SleepTCN_work
git clone https://github.com/quanuiter/SleepTCN.git
cd SleepTCN
git status --short ; git cat-file -t 8af1d12        # trống; "commit"
py -3.11 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements\lock-cu121-windows.txt
& $py -m pip install -e . --no-deps
& $py -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Dòng cuối phải in `2.5.1+cu121 12.1 True <tên GPU>`.

### A3. Dữ liệu Sleep-EDF và tiến độ cũ (từ Google Drive)

1. Tải `SleepTCN_upload/a4_inputs_sleepedf.tar` (7,06 GB) từ Drive về `C:\SleepTCN_work\` (ngoài thư mục repo).
2. Trên Drive, mở `SleepTCN_final_backup/runs`, tải thư mục `a4` về (Drive nén thành tệp zip), giải nén sao cho
   có đúng đường dẫn `C:\SleepTCN_work\SleepTCN\runs\a4\full\...` và `...\runs\a4\reproduction_check.json`.
   Không chép thư mục `runs/final`.

```powershell
cd C:\SleepTCN_work\SleepTCN
tar -xf C:\SleepTCN_work\a4_inputs_sleepedf.tar -C .
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
& $py scripts\run_amendment_a4.py verify-bundle             # BUNDLE VERIFIED: 460 files
& $py scripts\run_amendment_a4.py env-check --device cuda   # ENVIRONMENT OK
& $py scripts\run_amendment_a4.py plan                      # trạng thái hiện tại; kiểm tra tái lập: trained
git status --short                                          # vẫn trống
```

`plan` phải cho thấy khoảng 9 lần chạy `trained`, phần còn lại `missing` hoặc `partial`.

### A4. Huấn luyện (khoảng 10–13 giờ, ước lượng)

Mỗi cửa sổ PowerShell mới: `cd C:\SleepTCN_work\SleepTCN; $py = ".\.venv\Scripts\python.exe"; $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"`.

```powershell
& $py scripts\run_amendment_a4.py train --device cuda --workers 2 --num-workers 2
```

Với 6 GB VRAM chỉ dùng 2 tiến trình. Xem tiến độ ở cửa sổ khác: `& $py scripts\run_amendment_a4.py status`.

**Đo tốc độ trước:** khi lần chạy mới đầu tiên xong (khoảng 30–60 phút), mở
`runs\a4\monitoring\training_times.jsonl`, xem `wall_seconds` của dòng cuối. Với 2 tiến trình song song, nếu mỗi lần
khoảng 3.000 giây trở xuống thì máy xong trong khoảng 13 giờ; chậm hơn nhiều thì cân nhắc thuê GPU. Dừng bằng
Ctrl+C bất cứ lúc nào: lần chạy đã xong được giữ, chạy lại đúng lệnh `train` để tiếp tục.

### A5. Mở tập test Sleep-EDF và xuất kết quả

```powershell
& $py scripts\run_amendment_a4.py status                    # 40 runs trained
& $py scripts\run_amendment_a4.py test                      # preflight
& $py scripts\run_amendment_a4.py test --device cuda --confirm OPEN-A4-LOCKED-TEST-ONCE
& $py scripts\run_amendment_a4.py export --output C:\SleepTCN_work\a4_results.tar.gz
Get-FileHash C:\SleepTCN_work\a4_results.tar.gz -Algorithm SHA256
```

Chép `a4_results.tar.gz` (vài trăm MB) sang máy cá nhân (Drive hoặc USB).

## Phần B. Máy cá nhân (CPU)

```powershell
cd D:\SleepTCN
tar -xzf <đường dẫn>\a4_results.tar.gz -C .
python scripts\run_amendment_a4.py status                   # 40 runs tested
python scripts\run_amendment_a4.py prepare --cohorts shhs1_holdout telemetry   # dựng lại đầu vào A4, ~15 phút, ~12 GB
python scripts\final_infer.py a4 --device cpu --threads 8   # Telemetry rồi SHHS1, ~3-4 giờ
python scripts\a4_analyze.py
```

Sau đó gửi `runs\a4\analysis\a4_summary.json` để điền mục kết quả A4 vào bản thảo.

## Lỗi thường gặp

| Lỗi | Xử lý |
|---|---|
| `git_worktree_not_clean` | Có tệp lạ trong repo; tệp tải về đặt ở `C:\SleepTCN_work\`, ngoài `SleepTCN\` |
| `CUDA out of memory` | Chạy lại `train` với `--workers 1` |
| `torch.cuda.is_available()` là `False` | Driver NVIDIA quá cũ hoặc cài nhầm bản torch CPU; cài lại theo A2 |
| `export CUBLAS_WORKSPACE_CONFIG...` | Chưa đặt biến môi trường trong cửa sổ hiện tại |
