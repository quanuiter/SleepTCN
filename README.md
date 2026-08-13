# SleepTCN

Kho nghiên cứu tái triển khai ZleepAnlystNet và đánh giá TCN, ResNet-1D cùng các biến thể tiền xử lý
trên Sleep-EDF Expanded — Sleep Cassette.

> **Trạng thái cuối tại Gate 8 (2026-08-14):** toàn bộ chiến dịch chính E0–E6 và ablation nhóm đặc
> trưng C/P/N đã hoàn tất, mở test đúng quy trình, phân tích bắt cặp và kiểm toán artifact. Dự án dừng
> tại đây; không có chiến dịch GPU nào đang chờ chạy. Xem `notebooks/docs/GATE8_FINAL_RESULTS.md`,
> `notebooks/docs/STATUS_V2.md` và `runs/v2/publication/gate8/`.

## Kết quả chính

- Sleep-EDF Expanded: 78 đối tượng, 153 bản ghi và 195.469 epoch hợp lệ.
- Sáu cấu hình E0, E1, E2, E3, E4 và E6 dùng cùng split 10-fold theo đối tượng, seed huấn luyện 42.
- E3 đạt Macro-F1 mô tả cao nhất, `0,790443`; E3−E6 đạt chênh lệch `0,021319`, CI 95%
  `[0,012179; 0,030698]`, p Holm `0,001185`.
- ResNet-1D–TCN nhanh hơn E0 khoảng `3,76×` trong benchmark V100 đã khóa, nhưng có `4,37×` số tham
  số và peak VRAM cao hơn `28,4%`.
- Gate 8 không tìm thấy bằng chứng thống kê rằng P/N mang lại lợi ích tăng thêm cho Macro-F1 vùng
  chuyển pha khi đã có C và TCN cố định. Kết quả này không chứng minh tương đương và không cho phép
  diễn giải thành “phần trăm thông tin”.

## Trạng thái các cổng

| Cổng | Nội dung | Trạng thái |
|---:|---|---|
| 1 | Dữ liệu, tiền xử lý và split theo đối tượng | ĐẠT |
| 2 | Smoke CPU/GPU và cơ chế resume/checkpoint | ĐẠT |
| 3 | 60 run validation-only của E0/E1/E2/E3/E4/E6 | ĐẠT |
| 4 | Mở test một lần cho 60 checkpoint | ĐẠT |
| 5 | Phân tích bắt cặp theo đối tượng | ĐẠT |
| 6 | Benchmark và phân tích không gian đặc trưng | ĐẠT |
| 7 | Gói bảng, hình, bản thảo và ma trận bằng chứng | ĐẠT |
| 8 | Ablation C/P/N: 30 validation + 30 test | ĐẠT |

## Đọc theo thứ tự

1. `notebooks/docs/STATUS_V2.md` — trạng thái cuối và phạm vi đã hoàn thành.
2. `notebooks/docs/GATE8_FINAL_RESULTS.md` — kết quả và ranh giới kết luận Gate 8.
3. `runs/v2/publication/gate8/MANUSCRIPT_DRAFT_VI.md` — bản thảo tích hợp Gate 1–8.
4. `runs/v2/publication/gate8/CLAIM_EVIDENCE_MATRIX.md` — phát biểu được phép và bị cấm.
5. `notebooks/docs/NEXT_STEPS.md` — trạng thái dừng và điều kiện nếu mở nghiên cứu mới.

Các tài liệu không có hậu tố `V2`, cùng các runbook của Gate đã qua, là hồ sơ lịch sử để truy nguyên;
không dùng các câu “bước tiếp theo” trong đó để lập lịch hiện tại.

## Phạm vi và giới hạn

- Sleep-EDF Expanded 1.0.0, Sleep Cassette, EEG Fpz-Cz 100 Hz, epoch 30 giây.
- Năm lớp W/N1/N2/N3/REM; Movement/Unknown giữ vị trí thời gian nhưng bị mask khỏi loss/metrics.
- Train/validation/test tách theo đối tượng; hai đêm của cùng người luôn cùng vai trò.
- Chỉ một seed huấn luyện 42: chưa định lượng biến thiên do khởi tạo.
- Chưa đánh giá SHHS, zero-shot, domain shift, đa kênh hoặc giá trị lâm sàng.
- Không có kiểm định tương đương hoặc không thua kém.

## Cấu trúc thư mục

```text
configs/          Cấu hình và giao thức đã khóa
data/manifests/   Manifest nguồn và báo cáo kiểm định dữ liệu
data/splits/      Split 10-fold theo đối tượng
data/processed/   Dữ liệu NPZ lớn, không lưu Git
notebooks/docs/   Giao thức, runbook, trạng thái và báo cáo
runs/v2/          Checkpoint, prediction, metrics và gói công bố
scripts/          CLI tiền xử lý, huấn luyện, kiểm định và phân tích
src/sleeptcn/     Mã nguồn dùng chung
tests/            Kiểm thử tự động
```

## Quy tắc bất biến sau Gate 8

- Không sửa kết quả E0–E6 hoặc Gate 8 dựa trên việc đã xem test.
- Không gọi `p > 0,05` là bằng chứng tương đương.
- Không gọi ablation C/P/N là phép đo phần trăm thông tin hay quan hệ nhân quả.
- Không đưa dataset hoặc `data/cache/` lên kho Git.
- Mọi mở rộng nhiều seed hoặc SHHS phải là giao thức mới, tách khỏi kết quả Gate 1–8.
