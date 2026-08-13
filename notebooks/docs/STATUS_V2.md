# Trạng thái giao thức v2

Ngày audit gần nhất: **2026-08-14**.

Nhánh được kiểm toán: `run-in-docker`.

Commit: `b4ce94cb3b4f1d5d03e44b8c5287137cfa771767`.

## Tóm tắt

| Cổng | Trạng thái |
|---|---|
| Dữ liệu và split | ĐẠT |
| Smoke CPU/GPU | ĐẠT |
| Full validation-only 10-fold | ĐẠT — 60/60 run |
| Khóa checkpoint và mở test một lần | SẴN SÀNG CHUẨN BỊ, CHƯA MỞ TEST |
| Phân tích thống kê và báo cáo cuối | CHƯA THỰC HIỆN |

## Dữ liệu và split

- Sleep-EDF Expanded, phân tập Sleep Cassette: 78 đối tượng, 153 PSG và 153 Hypnogram.
- Năm biến thể đã được sinh và kiểm định, tổng cộng 765 NPZ.
- Mỗi biến thể có 195.767 epoch, gồm 195.469 epoch hợp lệ và 298 Movement/Unknown bị mask khỏi
  loss/metrics.
- Split v2 gồm 10 outer run theo đối tượng; hai đêm của cùng người luôn cùng vai trò.
- 78/78 đối tượng xuất hiện đúng một lần ở validation trên 10 fold.
- Split SHA-256:
  `6bc7ad74c07ff05f1d880cb5e720eea12386824ef465b966507906fa248925de`.
- E4 và E5 giống bitwise ở các trường khoa học trên 153/153 bản ghi; E5 bị loại khỏi fold 01--09.

Không tiền xử lý lại, không sửa manifest/split và không phục hồi E5 vào nhóm so sánh hiệu năng.

## Mã và môi trường

- Bộ kiểm thử v2 từng đạt 51/51 trong lần xác nhận CPU trước chiến dịch GPU.
- Smoke CPU/GPU E0--E6 fold 00 đã đạt.
- Môi trường GPU đã dùng: Python 3.10.13, PyTorch 2.5.1+cu121 và Tesla V100 PCIe 16 GB.
- Từ fold 02 trở đi đã khóa `CUBLAS_WORKSPACE_CONFIG=:4096:8` trong runbook.
- Config SHA-256 chung:
  `1d812bbfb45e9ca90e2654b41311954fd6e66a56e1bbcdbfba48df8147d0ae1b`.
- Runner SHA-256 chung:
  `12245ec0d2fe51a0843ed873d3080f752db23621c6e2b583bba4922be9be9a39`.

## Full validation-only 10-fold

Sáu thí nghiệm đang hoạt động là E0, E1, E2, E3, E4 và E6; training seed duy nhất là `42`.

| Fold | E0 | E1 | E2 | E3 | E4 | E6 | Test |
|---:|---|---|---|---|---|---|---|
| 00 | đạt | đạt | đạt | đạt | đạt | đạt | khóa |
| 01 | đạt | đạt | đạt | đạt | đạt | đạt | khóa |
| 02 | đạt | đạt | đạt | đạt | đạt | đạt | khóa |
| 03 | đạt | đạt | đạt | đạt | đạt | đạt | khóa |
| 04 | đạt | đạt | đạt | đạt | đạt | đạt | khóa |
| 05 | đạt | đạt | đạt | đạt | đạt | đạt | khóa |
| 06 | đạt | đạt | đạt | đạt | đạt | đạt | khóa |
| 07 | đạt | đạt | đạt | đạt | đạt | đạt | khóa |
| 08 | đạt | đạt | đạt | đạt | đạt | đạt | khóa |
| 09 | đạt | đạt | đạt | đạt | đạt | đạt | khóa |

Kết quả kiểm toán tổng thể:

- 60/60 manifest `complete` và được tạo từ worktree sạch.
- 60/60 validation report `passed=true`.
- 60/60 run chỉ có validation metrics; không có test prediction/metrics.
- Prediction của sáu E trong từng fold ghép cặp đúng theo bản ghi, epoch gốc và nhãn thật.
- Tính lại metrics từ prediction khớp tuyệt đối tệp metrics đã lưu.
- 250 `best.pt` và 250 `latest.pt` đọc được là checkpoint nhị phân thật.
- SHA-256 của 250/250 checkpoint tốt nhất khớp `complete.json`.

Báo cáo chi tiết: `VALIDATION_AUDIT_10FOLD.md`.

## Lưu ý diễn giải

- Trung bình validation chỉ dùng kiểm tra tính hợp lý, không phải kết quả cuối.
- Không chạy kiểm định thống kê trên 10 giá trị fold như 10 mẫu độc lập.
- Không thay đổi mô hình hoặc siêu tham số dựa trên validation sau khi cổng này được khóa.
- Chỉ dùng một training seed 42; đây là giới hạn phải nêu trong khóa luận/bài báo.
- E0--E6 trên Sleep-EDF chỉ cho phép kết luận in-domain, chưa chứng minh domain shift hoặc zero-shot.
- Monitoring thời gian không đầy đủ ở một số fold; benchmark tốc độ phải được đo lại có kiểm soát.

## Trạng thái hiện tại

Cổng validation-only đã đóng và đạt. Test **chưa mở**. Bước kế tiếp là chuẩn bị runbook mở test
một lần từ checkpoint đã khóa, rà soát cơ chế giữ worktree sạch và kiểm định test artifact. Chỉ
sau khi runbook đó được duyệt mới chạy `--resume --allow-test-evaluation`.
