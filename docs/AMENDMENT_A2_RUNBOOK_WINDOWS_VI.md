# Runbook Amendment A2 trên GPU thuê chạy Windows (RTX 3090)

Tài liệu này thay chặng II của `docs/AMENDMENT_A2_RUNBOOK_VI.md` khi máy thuê chạy Windows 10/11. Chặng I
(máy local, trước khi thuê) và chặng III (máy local, sau khi tải về) giữ nguyên.

Mọi lệnh dưới đây chạy trong **PowerShell**. Không cần kích hoạt venv: các lệnh gọi thẳng Python của venv qua
biến `$py`.

## Khác biệt so với bản Linux

- Môi trường cài từ `requirements/lock-cu121-windows.txt`: cùng `torch==2.5.1+cu121` và cùng phiên bản mọi thư
  viện chạy được như `lock-cu121.txt`, bỏ các gói chỉ có cho Linux (`nvidia-*`, `triton`; bản torch cho Windows đã
  kèm CUDA 12.1) và nhóm Jupyter không dùng. Hệ điều hành khác chiến dịch gốc và được ghi vào phần sai lệch khi báo
  cáo A2.
- Máy GPU checkout **commit công cụ mới nhất** (có runbook này và file khóa Windows), không phải commit khóa. Lệnh
  `env-check` xác nhận config A2 trên máy đúng là bản đã khóa ở commit "Lock Amendment A2" và chưa bị sửa.
- Đã kiểm tra trên Windows: tiền xử lý, toàn bộ test, runner với DataLoader worker (kết quả giống hệt khi không
  dùng worker), và việc cài đặt thử file khóa (35 gói giải quyết đúng phiên bản, có bản
  `torch-2.5.1+cu121-cp311-win_amd64`).

## W.1. Chuẩn bị máy

Kiểm tra Git và Python 3.11:

```powershell
git --version
py -3.11 --version
nvidia-smi
```

Nếu thiếu, cài rồi **mở lại PowerShell**:

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.11 -e
```

Giữ máy không ngủ trong lúc chạy (Settings > System > Power & sleep > Never). Khi rời máy, chỉ **đóng cửa sổ Remote
Desktop** (ngắt kết nối); **không Sign out**, vì đăng xuất sẽ dừng mọi tiến trình.

## W.2. Lấy mã nguồn

Cần **toàn bộ lịch sử Git** (không dùng `--depth`). Tag `a2-gpu-run` trỏ tới commit công cụ dùng cho máy GPU.

```powershell
New-Item -ItemType Directory -Force C:\work | Out-Null
Set-Location C:\work
git clone https://github.com/quanuiter/SleepTCN.git
Set-Location C:\work\SleepTCN
git checkout a2-gpu-run
git status --short          # phải trống
git cat-file -t 8af1d12     # phải in "commit"
```

Không sửa tệp nào trong repo trên máy GPU. Mọi đầu ra A2 (`runs\`, `data\processed\`, `data\cache\`) đều nằm trong
`.gitignore`.

## W.3. Môi trường

```powershell
py -3.11 -m venv .venv
$py = "C:\work\SleepTCN\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements\lock-cu121-windows.txt
& $py -m pip install -e . --no-deps
& $py -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Dòng cuối phải in `2.5.1+cu121 12.1 True NVIDIA GeForce RTX 3090`.

## W.4. Biến môi trường (mỗi cửa sổ PowerShell mới đều phải làm lại)

```powershell
Set-Location C:\work\SleepTCN
$py = "C:\work\SleepTCN\.venv\Scripts\python.exe"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:PYTHONIOENCODING = "utf-8"
```

Kiểm tra môi trường và lưu thông tin máy:

```powershell
& $py scripts\run_amendment_a2.py env-check --device cuda
nvidia-smi | Out-File -Encoding utf8 runs\a2\monitoring\gpu_info.txt
& $py -m pip freeze | Out-File -Encoding utf8 runs\a2\monitoring\pip_freeze.txt
```

`env-check` phải in `ENVIRONMENT OK`. Nếu không, đọc danh sách `errors` và sửa trước khi đi tiếp.

## W.5. Dữ liệu Sleep-EDF

Đặt `a2_inputs_sleepedf.tar` vào `C:\work`, kiểm tra mã SHA-256:

```powershell
(Get-FileHash C:\work\a2_inputs_sleepedf.tar -Algorithm SHA256).Hash
```

Phải bằng `2D74180CC118ED3D5BCC631653B042F34DEE5ECAC2919A8394F3662EE8E42107`. Giải nén **tại gốc repo** và kiểm tra:

```powershell
tar -xf C:\work\a2_inputs_sleepedf.tar -C C:\work\SleepTCN
& $py scripts\run_amendment_a2.py verify-bundle --part sleepedf
& $py scripts\run_amendment_a2.py plan
git status --short          # vẫn phải trống
```

`verify-bundle` phải in `BUNDLE VERIFIED (sleepedf)`. `plan` phải in trạng thái `locked_before_a2_training`, 50 lần
chạy `missing` và kiểm tra tái lập `missing`.

## W.6. Kiểm tra tái lập (khoảng 20 phút)

```powershell
& $py scripts\run_amendment_a2.py repro --device cuda
```

Kết quả ghi vào `runs\a2\reproduction_check.json`; dòng cuối in `IDENTICAL` hoặc `DIFFERS`. Cả hai trường hợp đều
tiếp tục và được báo cáo nguyên trạng. Chỉ dừng nếu lệnh **lỗi**.

## W.7. Huấn luyện 50 lần chạy (chạy nền)

RTX 3090 có 24 GB VRAM và máy có 56 GB RAM, nên chạy 4 tiến trình:

```powershell
Start-Process -FilePath $py -WorkingDirectory C:\work\SleepTCN -WindowStyle Hidden `
    -ArgumentList "scripts\run_amendment_a2.py","train","--device","cuda","--workers","4","--num-workers","2" `
    -RedirectStandardOutput runs\a2\monitoring\train_driver.log `
    -RedirectStandardError runs\a2\monitoring\train_driver.err
```

Tiến trình nền nhận biến môi trường của cửa sổ hiện tại, nên phải làm W.4 trước. Theo dõi:

```powershell
Get-Content runs\a2\monitoring\train_worker_0.log -Wait -Tail 20     # Ctrl+C để thoát xem, không dừng train
& $py scripts\run_amendment_a2.py status
nvidia-smi
```

- Nếu một worker báo hết bộ nhớ GPU: chạy lại lệnh `train` ở trên với `"--workers","3"`.
- Bị ngắt (máy khởi động lại, lỡ đăng xuất): làm lại W.4 rồi chạy lại đúng lệnh `train`. Lần chạy xong được bỏ qua,
  lần chạy dở được tiếp tục từ checkpoint. Không xóa thư mục lần chạy, không đổi seed. Không chạy hai lệnh `train`
  cùng lúc.
- Khi xong, `train_driver.log` có dòng `ALL A2 TRAINING RUNS FINISHED` và `status` cho 50 lần chạy `trained`.

**Trong lúc train chạy:**

1. Ở máy local: chạy phân tích D (`python scripts/analyze_amendment_a2.py d`, khoảng 1,5 giờ).
2. Đặt `a2_inputs_holdout.tar` vào `C:\work`, kiểm tra và giải nén (trong một cửa sổ PowerShell khác, sau W.4):

   ```powershell
   (Get-FileHash C:\work\a2_inputs_holdout.tar -Algorithm SHA256).Hash
   tar -xf C:\work\a2_inputs_holdout.tar -C C:\work\SleepTCN
   & $py scripts\run_amendment_a2.py verify-bundle --part holdout
   git status --short          # vẫn phải trống
   ```

   Mã phải bằng `5314831D653F960F2EE9D97B91CE350F2E7F54CB2886772EDAAC31C7C02487F4`.

## W.8. Mở tập test Sleep-EDF (một lần duy nhất)

```powershell
& $py scripts\run_amendment_a2.py test
& $py scripts\run_amendment_a2.py test --device cuda --confirm OPEN-A2-LOCKED-TEST-ONCE
```

Lệnh đầu chỉ kiểm tra, không đọc dữ liệu test; phải in `Preflight passed for 50 runs`. Nếu bị ngắt, chạy lại đúng lệnh
thứ hai. Không bao giờ xóa `runs\a2\test_campaign_a2.json`.

## W.9. Suy luận holdout trên GPU

```powershell
& $py scripts\run_amendment_a2.py holdout-check --device cuda
& $py scripts\run_amendment_a2.py holdout --device cuda --workers 3 --threads 2
```

`holdout-check` phải in `DEVICE CHECK PASSED`. Nếu **không** qua, không chạy `holdout` trên GPU; sau khi tải kết quả
về, chạy hai lệnh trên ở máy local với `--device cpu --workers 6`. Khi xong: `A2 HOLDOUT INFERENCE COMPLETE`.

## W.10. Đo thời gian mỗi bước

Chạy khi GPU không còn tiến trình Python nào khác (`nvidia-smi` không còn `python.exe`):

```powershell
& $py scripts\run_amendment_a2.py step-benchmark --device cuda
```

## W.11. Đóng gói kết quả và tải về

```powershell
& $py scripts\run_amendment_a2.py export --output C:\work\a2_results.tar.gz
(Get-FileHash C:\work\a2_results.tar.gz -Algorithm SHA256).Hash
```

Tải `a2_results.tar.gz` về máy local, kiểm tra lại mã SHA-256 sau khi tải, rồi mới trả máy. Ở máy local tiếp tục từ
chặng III của `docs/AMENDMENT_A2_RUNBOOK_VI.md`.

## Xử lý sự cố riêng của Windows

| Hiện tượng | Cách xử lý |
|---|---|
| `py` không nhận `-3.11` | Cài Python 3.11 (W.1), mở lại PowerShell |
| `running scripts is disabled on this system` | Không kích hoạt venv; dùng `& $py ...` như trong runbook |
| `ENVIRONMENT NOT READY` với `a2_config_...` | Checkout sai commit, hoặc config bị sửa; checkout lại đúng commit ở W.2 |
| `ENVIRONMENT NOT READY` với `git_worktree_not_clean` | Có tệp lạ trong repo (thường do giải nén sai chỗ); `git status --short` để xem |
| `UnicodeEncodeError ... charmap` | Chưa đặt `$env:PYTHONIOENCODING = "utf-8"` (W.4) |
| `PermissionError` khi ghi tệp | Có hai lệnh cùng chạy trên cùng thư mục; dừng bớt, chạy lại một lệnh |

## Commit cho máy GPU

- Công cụ: tag `a2-gpu-run`.
- Commit khóa config A2: `38eb6dca6803c5ba8096ee23c97ff005361b9571` ("Lock Amendment A2"); `env-check` xác nhận
  config trên máy trùng với commit này.
