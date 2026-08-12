# Trạng thái giao thức v2

Ngày audit gần nhất: **2026-08-12**. Trạng thái dưới đây chỉ ghi nhận những gì có bằng chứng trong
bản repository local hiện tại; tiến trình đang chạy trên Docker nhưng chưa pull về không được tính
là hoàn tất.

## Đã hoàn thành và kiểm chứng

### Dữ liệu và split

- Kiểm kê 153 cặp PSG/Hypnogram của 78 đối tượng, kèm SHA-256 nguồn và audit metadata EDF.
- Sinh đủ năm biến thể, tổng 765 NPZ: `paper_raw_v1`, `bandpass_v2`, `bandpass_clip_v2`,
  `filtered_v2`, `filtered_zscore_v2`.
- `processed_validation_v2.json` xác nhận 765/765 file hợp lệ, 0 lỗi file, 0 lỗi toàn cục.
- Mỗi biến thể có 195.767 epoch, gồm 195.469 epoch hợp lệ và 298 Movement/Unknown.
- Split v2 có 10 outer run theo đối tượng; test là fold `i`, validation là fold `(i+1) mod 10`,
  train là tám fold còn lại. SHA-256 manifest khớp sidecar:
  `6bc7ad74c07ff05f1d880cb5e720eea12386824ef465b966507906fa248925de`.
- Kiểm tra bitwise 153/153 cặp xác nhận E4 và E5 có `x`, `y`, `valid_mask`,
  `original_epoch_index` giống hệt nhau; `clip_fraction=0` cho mọi bản ghi.

### Mã và cổng kỹ thuật

- 51 test case tồn tại trong bộ kiểm thử v2; lần xác nhận CPU trước đây đã đạt 51/51.
- Smoke CPU và GPU v2 E0--E6 fold 00, seed 42 đã hoàn tất và artifact pass.
- Runner khóa test khỏi quá trình fit, hỗ trợ checkpoint `latest.pt`/`best.pt`, resume tại cuối
  epoch và từ chối full run khi worktree bẩn.
- Môi trường GPU đã dùng: Python 3.10.13, PyTorch 2.5.1+cu121, CUDA wheel 12.1, Tesla V100
  PCIe 16 GB. Kiểm tra môi trường GPU đã PASS.

### Full validation-only

| Fold | Experiment có artifact | Trạng thái | Test |
|---|---|---|---|
| 00 | E0, E1, E2, E3, E4, E5, E6 | tất cả `complete`, report `passed=true` | khóa |
| 01 | E0, E1, E2, E3, E4, E6 | tất cả `complete`, report `passed=true` | khóa |
| 02--09 | chưa có đủ artifact trong bản local | chưa xác nhận | khóa |

Fold 00 giữ E5 làm bằng chứng kiểm toán. Theo quyết định protocol ngày 2026-08-11, không chạy E5
ở fold 01--09 và không đưa E5 vào kiểm định hiệu năng cuối cùng.

Các run fold 00/01 dùng cùng config SHA-256
`1d812bbfb45e9ca90e2654b41311954fd6e66a56e1bbcdbfba48df8147d0ae1b`, cùng split SHA-256
và cùng runner code SHA-256
`12245ec0d2fe51a0843ed873d3080f752db23621c6e2b583bba4922be9be9a39`.
Commit Git giữa một số run khác nhau do cập nhật tài liệu/dependency/artifact; code runner và config
được ghi trong manifest vẫn giống nhau.

Tình trạng checkpoint local khác nhau giữa hai fold:

- Fold 01: 50 file `.pt` là checkpoint nhị phân thật; SHA-256 của checkpoint sequence, manifest,
  prediction và metrics đều khớp validation report.
- Fold 00: 54 file `.pt` hiện là Git LFS pointer, chưa được hydrate trên máy local này. OID/size
  trong pointer của checkpoint sequence khớp SHA-256/metadata đã ghi, còn manifest, prediction,
  metrics và validation report đều khớp byte hiện tại. Kết quả fold 00 vẫn hợp lệ, nhưng chưa thể
  resume/mở test từ máy local cho tới khi blob LFS thật được pull hoặc phục hồi từ bản backup.

## Lưu ý audit

- Trường `status` trong `configs/experiments_v2.json` vẫn có chuỗi lịch sử
  `gpu_smoke_pending`. Không sửa riêng trường này trong chiến dịch hiện tại vì mọi thay đổi config
  sẽ đổi SHA-256 và làm các fold sau không còn cùng cấu hình với fold 00/01. Trạng thái vận hành
  hiện tại lấy từ tài liệu này và artifact, không lấy từ trường mô tả đó.
- Fold 01 có thể đã phát cảnh báo cuBLAS deterministic. Cảnh báo không làm run sai: dữ liệu,
  split, code, checkpoint và artifact đều hợp lệ. Từ fold 02 trở đi luôn export
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` trước khi khởi động Python CUDA để tăng khả năng tái lập.
- Lần audit này không chạy lại pytest trên máy local vì environment hiện tại chưa cài PyTorch và
  package `sleeptcn`; đây là thiếu dependency của máy audit, không phải test failure của code.
- `data/cache/features/v2/` là cache tái tạo được và không được commit. Resume trong cùng máy cần
  checkpoint; nếu chuyển máy mà không mang cache, feature sẽ được trích xuất lại từ checkpoint.
- Bản local hiện giữ bốn thư mục dữ liệu hoạt động; `bandpass_clip_v2` đã được đổi tên/đưa ra khỏi
  đường dẫn hoạt động sau khi E5 bị loại. Manifest kiểm định 765 file và báo cáo bitwise là bằng
  chứng của lần sinh đầy đủ trước đó. `check_environment.py` mặc định kiểm tra bốn biến thể hoạt
  động; dùng `--include-retired-e5` chỉ khi cần audit lại E5.

## Bước hiện tại

1. Hoàn tất và pull về fold 02 cho E0, E1, E2, E3, E4, E6; chạy validation artifact sau từng E.
2. Audit fold 02 giống bảng trên rồi mới chuyển lần lượt sang fold 03--09, mỗi phiên thuê một fold.
3. Trước khi mở test, cài Git LFS trên một máy, pull/`git lfs fsck` checkpoint fold 00 và xác nhận
   các file `.pt` là blob PyTorch thật, không còn là pointer 130--132 byte. Nếu blob remote thiếu,
   phục hồi checkpoint fold 00 từ backup Docker/USB hoặc chạy lại fold đó.
4. Chỉ sau khi đủ 10 fold validation-only và mọi report pass mới khóa checkpoint cuối cùng và mở
   test đúng một lần bằng `--resume --allow-test-evaluation`.
5. Sau test: ghép dự đoán out-of-fold theo đối tượng, chạy bootstrap/Wilcoxon/Holm, benchmark tài
   nguyên và viết báo cáo. Không thay đổi code/config dựa trên test.
