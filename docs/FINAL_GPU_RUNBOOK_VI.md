# Runbook phiên GPU cuối cùng (A4 + A5)

Một phiên máy thuê làm toàn bộ phần tính toán còn lại:
- A4: huấn luyện hai biên bộ lọc, mở tập test Sleep-EDF, suy luận trên SHHS1 giữ lại và Telemetry.
- A5: tải và tiền xử lý ISRUC nhóm I, suy luận ở kênh C4 và F4.

Giao thức: `docs/AMENDMENT_A4.md`, `docs/AMENDMENT_A5.md`. Chi tiết cài môi trường giống
`docs/AMENDMENT_A4_RUNBOOK_VI.md` mục II.1–II.3.

**Máy cần:** GPU ít nhất 16 GB (V100 như A2), ít nhất 6 nhân CPU, ít nhất 100 GB đĩa, quyền root (để cài công cụ
tải MEGA và giải nén RAR). **Thời gian:** khoảng 10–12 giờ, phần lớn máy tự chạy.

## Phần 0. Ở máy local, trước khi thuê

| Tệp tải lên Google Drive | Dung lượng | SHA-256 |
|---|---|---|
| `C:\Users\ADMIN\SleepTCN_upload\a4_inputs_sleepedf.tar` | 7,06 GB | `4271ea88b4aeb90affe0162a835e3dc00eb53e6835994af0a759bc3c857842ab` |
| `C:\Users\ADMIN\SleepTCN_upload\final_core.tar` | 2,24 GB | `32d735b393934fd94489d6358232ce02369e49979802bae0bb461b9ea6644887` |
| `final_holdout.tar` (chỉ khi bạn quyết định đưa đầu vào SHHS1 lên máy thuê; xem DUA của NSRR) | khoảng 11 GB | ghi khi tạo |

Tạo `final_holdout.tar` (nếu dùng):

```powershell
python scripts/final_bundle.py pack --part holdout --output C:/Users/ADMIN/SleepTCN_upload/final_holdout.tar
Get-FileHash C:\Users\ADMIN\SleepTCN_upload\final_holdout.tar -Algorithm SHA256
```

Mỗi tệp đặt chia sẻ "Bất kỳ ai có đường liên kết", ghi lại ID (đoạn giữa `/d/` và `/view`). Bộ đọc ISRUC phải đã
được thử trên một người thật ở máy local trước phiên này.

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
git status --short                                                     # vẫn trống
```

## Phần 3. Bắt đầu A4 (chạy nền khoảng 7 giờ)

```bash
python scripts/run_amendment_a4.py repro --device cuda                 # ~20 phút; IDENTICAL hoặc DIFFERS, đều tiếp tục
nohup python scripts/run_amendment_a4.py train --device cuda --workers 3 --num-workers 1 > /workspace/a4_train.log 2>&1 &
```

## Phần 4. Trong lúc A4 huấn luyện: ISRUC và kiểm tra thiết bị

Kiểm tra thiết bị (bắt buộc trước mọi suy luận; phải in `DEVICE CHECK PASSED`):

```bash
python scripts/final_infer.py device-check --device cuda
```

Tải ISRUC nhóm I bằng MEGAcmd. Bản cài cho Ubuntu lấy từ trang tải chính thức của MEGA
(https://mega.io/cmd); chọn đúng phiên bản Ubuntu của máy (`lsb_release -a`):

```bash
apt-get update && apt-get install -y unrar || apt-get install -y p7zip-full
# cài gói megacmd .deb của MEGA cho đúng phiên bản Ubuntu, ví dụ:
#   wget https://mega.nz/linux/repo/xUbuntu_22.04/amd64/megacmd-xUbuntu_22.04_amd64.deb && apt-get install -y ./megacmd-*.deb
mkdir -p /workspace/isruc && cd /workspace/isruc
mega-get "https://mega.nz/folder/QJgDQDDZ#ZMDj3w82msavACurqP48IA" /workspace/isruc/
```

Nếu MEGA báo vượt hạn mức tải (quota), đợi rồi chạy lại đúng lệnh `mega-get` (tệp đã tải được giữ). Sau khi tải
xong, đưa các tệp `N.rar` (hoặc thư mục `N/`) về thẳng `/workspace/isruc/`, rồi:

```bash
cd /workspace/SleepTCN
python scripts/a5_prepare_isruc.py inspect --raw-root /workspace/isruc   # chỉ đọc định dạng; xem danh sách loại
python scripts/a5_prepare_isruc.py run --raw-root /workspace/isruc       # ghi data/processed/isruc_a5
nohup python scripts/final_infer.py a5 --device cuda > /workspace/a5_infer.log 2>&1 &
```

`inspect` in kênh, đơn vị, tần số, mã nhãn và số nhãn của từng người. Nếu dừng vì lỗi, hoặc loại quá 10 người,
**dừng phần ISRUC** và gửi log về; không sửa mã trên máy thuê. A4 vẫn chạy bình thường.

## Phần 5. Sau khi A4 huấn luyện xong

```bash
python scripts/run_amendment_a4.py status                             # 40 runs trained
python scripts/run_amendment_a4.py test                               # preflight
python scripts/run_amendment_a4.py test --device cuda --confirm OPEN-A4-LOCKED-TEST-ONCE
python scripts/final_infer.py a4 --device cuda                        # cần final_holdout.tar; nếu không, xem dưới
```

Nếu không đưa SHHS1 lên máy thuê, lệnh `a4` sẽ dừng ở phần SHHS1: chạy phần đó ở máy local bằng CPU (khoảng
3 giờ) sau khi tải kết quả về, cùng lệnh với `--device cpu`.

## Phần 6. Xuất kết quả và tải về

```bash
cd /workspace/SleepTCN
tar -czf /workspace/final_results.tar.gz runs/a4/full runs/a4/monitoring runs/a4/test_campaign_a4.json \
    runs/a4/reproduction_check.json runs/final data/processed/isruc_a5/isruc_preprocess_manifest.json \
    data/processed/isruc_a5/C4/bandpass_v2 data/processed/isruc_a5/F4/bandpass_v2 /workspace/*.log
sha256sum /workspace/final_results.tar.gz
```

Tải `final_results.tar.gz` về máy local và kiểm tra SHA-256 trước khi tắt máy thuê. Ở máy local:

```powershell
tar -xzf final_results.tar.gz -C D:\SleepTCN
python scripts/a4_analyze.py
python scripts/a5_analyze.py
```

## Lỗi thường gặp

| Lỗi | Xử lý |
|---|---|
| `git_worktree_not_clean` | Có tệp lạ trong repo; mọi tệp tải về đặt ở `/workspace`, ngoài `SleepTCN/` |
| `DEVICE CHECK FAILED` | Không chạy suy luận; gửi `runs/final/device_check.json` về |
| MEGA vượt hạn mức | Đợi rồi chạy lại `mega-get`; A4 không bị ảnh hưởng |
| `a5_prepare_isruc.py` dừng | Dừng phần ISRUC, gửi log; A4 vẫn hoàn tất được |
