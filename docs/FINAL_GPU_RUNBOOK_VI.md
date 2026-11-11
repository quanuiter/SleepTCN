# Runbook phiên GPU cuối cùng (A4)

Một phiên máy thuê làm toàn bộ phần tính toán còn lại của bài: A4, huấn luyện hai biên bộ lọc (E4HP chỉ lọc
thông cao 0,5 Hz, E4LP chỉ lọc thông thấp 30 Hz; seed 42 và 123; 10 fold), mở tập test Sleep-EDF một lần, rồi
suy luận zero-shot trên Telemetry và, nếu bạn đưa dữ liệu lên, trên SHHS1 giữ lại. ISRUC không còn trong phiên này
(sai lệch D1 của A5, `docs/AMENDMENT_A5.md` mục 7).

Giao thức: `docs/AMENDMENT_A4.md`. Chi tiết cài môi trường: `docs/AMENDMENT_A4_RUNBOOK_VI.md` mục II.1–II.3.

**Máy cần:** GPU ít nhất 16 GB (V100 như A2), ít nhất 6 nhân CPU, ít nhất 60 GB đĩa. Không cần quyền root.
**Thời gian:** khoảng 8–9 giờ (A4 huấn luyện 6–7 giờ, phần còn lại khoảng 1–2 giờ), phần lớn máy tự chạy.

## Phần 0. Ở máy local, trước khi thuê

| Tệp tải lên Google Drive | Dung lượng | SHA-256 |
|---|---|---|
| `C:\Users\ADMIN\SleepTCN_upload\a4_inputs_sleepedf.tar` | 7,06 GB | `4271ea88b4aeb90affe0162a835e3dc00eb53e6835994af0a759bc3c857842ab` |
| `C:\Users\ADMIN\SleepTCN_upload\final_core.tar` | 2,24 GB | `32d735b393934fd94489d6358232ce02369e49979802bae0bb461b9ea6644887` |
| `C:\Users\ADMIN\SleepTCN_upload\final_holdout.tar` (đầu vào A4 của SHHS1; dữ liệu NSRR, theo DUA) | 11,03 GB | `af618a968f3bf4ec7f7f176af561810c86f607462bfc2128a9c7b66ae2e907e9` |

Tạo `final_holdout.tar` nếu dùng:

```powershell
python scripts/final_bundle.py pack --part holdout --output C:/Users/ADMIN/SleepTCN_upload/final_holdout.tar
Get-FileHash C:\Users\ADMIN\SleepTCN_upload\final_holdout.tar -Algorithm SHA256
```

Mỗi tệp đặt chia sẻ "Bất kỳ ai có đường liên kết", ghi lại ID (đoạn giữa `/d/` và `/view`).

## Phần 1. Mã nguồn và môi trường

```bash
nvidia-smi
cd /workspace && git clone https://github.com/quanuiter/SleepTCN.git SleepTCN && cd SleepTCN
git status --short && git cat-file -t 8af1d12      # trống; "commit"
# môi trường Python 3.11 + requirements/lock-cu121.txt như AMENDMENT_A4_RUNBOOK_VI.md mục II.2
source .venv/bin/activate && export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Mỗi terminal mới: `cd /workspace/SleepTCN && source .venv/bin/activate && export CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## Phần 2. Dữ liệu từ Drive

```bash
cd /workspace && python -m pip install --quiet gdown
gdown <ID a4_inputs_sleepedf.tar> -O /workspace/a4_inputs_sleepedf.tar
gdown <ID final_core.tar>        -O /workspace/final_core.tar
gdown <ID final_holdout.tar>     -O /workspace/final_holdout.tar      # nếu dùng
sha256sum /workspace/*.tar                                             # so với bảng Phần 0
cd /workspace/SleepTCN
for f in /workspace/*.tar; do tar -xf "$f" -C /workspace/SleepTCN; done
python scripts/run_amendment_a4.py verify-bundle
python scripts/final_bundle.py verify --part core
python scripts/final_bundle.py verify --part holdout                   # nếu dùng
python scripts/run_amendment_a4.py env-check --device cuda             # ENVIRONMENT OK
python scripts/final_infer.py device-check --device cuda               # DEVICE CHECK PASSED
git status --short                                                     # vẫn trống
```

## Phần 3. A4: kiểm tra tái lập và huấn luyện (khoảng 7 giờ)

```bash
python scripts/run_amendment_a4.py repro --device cuda                 # ~20 phút; IDENTICAL hoặc DIFFERS, đều tiếp tục
nohup python scripts/run_amendment_a4.py train --device cuda --workers 3 --num-workers 1 > /workspace/a4_train.log 2>&1 &
python scripts/run_amendment_a4.py status                              # xem tiến độ bất cứ lúc nào
```

Nếu mất kết nối hay một tiến trình lỗi, chạy lại đúng lệnh `train`: lần chạy đã xong được bỏ qua, lần chạy dở được
tiếp tục từ checkpoint.

## Phần 4. Sau khi huấn luyện xong: mở test và suy luận

```bash
python scripts/run_amendment_a4.py status                              # 40 runs trained
python scripts/run_amendment_a4.py test                                # preflight
python scripts/run_amendment_a4.py test --device cuda --confirm OPEN-A4-LOCKED-TEST-ONCE
python scripts/final_infer.py a4 --device cuda                         # Telemetry, rồi SHHS1 nếu có đầu vào
```

Nếu không đưa SHHS1 lên máy thuê, lệnh `a4` in "skipped" cho SHHS1; phần đó chạy ở máy local bằng CPU (khoảng
3 giờ) sau khi tải kết quả về: `python scripts/final_infer.py a4 --device cpu`.

## Phần 5. Xuất kết quả và tải về

```bash
cd /workspace/SleepTCN
tar -czf /workspace/final_results.tar.gz runs/a4/full runs/a4/monitoring runs/a4/test_campaign_a4.json \
    runs/a4/reproduction_check.json runs/final /workspace/a4_train.log
sha256sum /workspace/final_results.tar.gz
```

Tải `final_results.tar.gz` về và kiểm tra SHA-256 trước khi tắt máy thuê. Ở máy local:

```powershell
tar -xzf final_results.tar.gz -C D:\SleepTCN
python scripts/a4_analyze.py
```

## Lỗi thường gặp

| Lỗi | Xử lý |
|---|---|
| `git_worktree_not_clean` | Có tệp lạ trong repo; mọi tệp tải về đặt ở `/workspace`, ngoài `SleepTCN/` |
| `a4_config_not_locked` | Clone trước khi có commit khóa A4; `git pull` |
| `git_history_without_8af1d12...` | Clone nông; clone lại không có `--depth` |
| `export CUBLAS_WORKSPACE_CONFIG...` | Chưa đặt biến môi trường trong terminal hiện tại |
| `DEVICE CHECK FAILED` | Không chạy suy luận; gửi `runs/final/device_check.json` về |

## Phương án Google Colab

Notebook `notebooks/final_colab_a4.ipynb` làm toàn bộ các phần trên trên Colab: đặt ba tệp tar vào
`MyDrive/SleepTCN_upload/`, mở notebook trên Colab với runtime GPU, chạy ô 1 → ô 5. Tiến độ được đồng bộ lên
`MyDrive/SleepTCN_final_backup/` mỗi 5 phút; khi bị ngắt kết nối, chạy lại ô 1, 2, 3 rồi ô 4 để tiếp tục. Kết quả
`final_results.tar.gz` được ghi vào thư mục backup đó.
