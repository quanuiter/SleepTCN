# SleepTCN

SleepTCN là kho nghiên cứu tái triển khai và đánh giá các quy trình phân giai đoạn giấc ngủ từ EEG một
kênh trên Sleep-EDF Expanded — Sleep Cassette. Nghiên cứu tập trung vào sự khác biệt giữa 15CNN–BiLSTM,
15CNN–TCN và ResNet-1D–TCN, đồng thời đánh giá các phương án tiền xử lý tín hiệu.

> **Trạng thái ngày 2026-08-25:** Gate 1–8 trên Sleep-EDF, chiến dịch zero-shot SHHS1, phân tích thành
> phần E1/E2, đối chiếu E3−E2 và phân tích độ nhạy với seed 42/123 đã hoàn tất. Kết quả hiện hành được
> trình bày tại `docs/STATUS_V2.md` và trong báo cáo `Reports/output/pdf/SleepTCN_Gate1_8_SHHS_Report.pdf`.

## Kết quả chính

- Sleep-EDF Expanded: 78 đối tượng, 153 bản ghi và 195.469 epoch hợp lệ.
- Sáu cấu hình E0, E1, E2, E3, E4 và E6 dùng cùng split 10-fold theo đối tượng. Seed 42 là chiến dịch
  chính Gate 1--8; seed 123 là lần lặp đầy đủ sau giao thức để đánh giá độ nhạy theo khởi tạo.
- E3 đạt Macro-F1 mô tả cao nhất, `0,790443`; E3−E6 đạt chênh lệch `0,021319`, CI 95%
  `[0,012179; 0,030698]`, p Holm `0,001185`.
- Ở seed 123, E3−E6 vẫn dương `+0,010249`, CI `[0,002435; 0,017989]`, nhưng p Holm `0,131289`.
  Hướng hiệu ứng lặp lại, còn ý nghĩa sau hiệu chỉnh Holm thì không.
- ResNet-1D–TCN nhanh hơn E0 khoảng `3,76×` trong benchmark V100 đã khóa, nhưng có `4,37×` số tham
  số và peak VRAM cao hơn `28,4%`.
- Gate 8 không tìm thấy bằng chứng thống kê rằng P/N mang lại lợi ích tăng thêm cho Macro-F1 vùng
  chuyển pha khi đã có C và TCN cố định. Kết quả này không chứng minh tương đương và không cho phép
  diễn giải thành “phần trăm thông tin”.
- Zero-shot SHHS1 dùng 180 đối tượng và 169.012 epoch: E3−E0 đạt `+0,041219`, CI 95%
  `[0,031367; 0,051196]`; E3−E6 đạt `+0,027359`, CI `[0,018172; 0,036979]`.
- Phân tích thành phần SHHS cho E1−E0 `+0,006515`, nhưng E2−E1 `-0,012800`.
- Đối chiếu E3−E2 trên SHHS đạt `+0,047504`, CI `[0,037242; 0,057923]`, E3 thắng `147/180`
  đối tượng. Kết quả cung cấp bằng chứng mạnh cho toàn chế độ tiền xử lý E3 so với raw trên mẫu
  hiện tại; ba thao tác vẫn chưa được tách riêng về nhân quả.

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

## Tài liệu kết quả

1. `docs/README.md` — mục lục tài liệu.
2. `docs/STATUS_V2.md` — nguồn duy nhất cho trạng thái cuối.
3. `docs/GATE8_FINAL_RESULTS.md` — kết quả và ranh giới kết luận Gate 8.
4. `docs/SHHS_ZERO_SHOT_RESULTS.md` — kết quả zero-shot và phân tích thành phần.
5. `docs/MULTISEED_SENSITIVITY_RESULTS.md` — độ nhạy theo hai seed cố định.
6. `Reports/output/pdf/SleepTCN_Gate1_8_SHHS_Report.pdf` — báo cáo LaTeX hoàn chỉnh.
7. `runs/v2/publication/gate8/CLAIM_EVIDENCE_MATRIX.md` — phát biểu được phép và bị cấm.
8. `docs/REPRODUCIBILITY_PACKAGE.md` — lock môi trường, audit hash và quy trình tái sinh NPZ.
9. `docs/SOURCE_ARCHITECTURE.md` — ranh giới module và nguyên tắc tổ chức source.

`STATUS_V2.md` là tài liệu chính thức về trạng thái và phạm vi kết luận. Các protocol và runbook ghi lại
thiết kế thực nghiệm, điều kiện tái lập và nguồn gốc của từng nhóm kết quả.

## Demo bảo vệ khóa luận

Giao diện Streamlit được giữ ở một màn hình trực quan: chọn bản ghi, xem số epoch W/N1/N2/N3/REM, đọc
timeline bậc thang của cả đêm ngủ và lọc epoch theo W/N1/N2/N3/REM để xem 30 giây EEG cùng giai đoạn tương ứng.
Người xem chọn một model E3 hoặc E0; cả bản ghi mẫu lẫn EDF tải lên đều hiển thị rõ đây là dự đoán của model đã
chọn. Toàn bộ so sánh model, kiểm định và số liệu nghiên cứu được giữ trong báo cáo thay vì nhồi vào demo.

```powershell
python -m pip install -e ".[demo]"
python -m streamlit run demo/app.py
```

Chuẩn bị prediction artifact/checkpoint local cho E3/E0 bằng
`python scripts/prepare_demo_assets.py --ref run-in-docker --fold 0 --seed 123`.
Hướng dẫn đầy đủ và kịch bản trình bày: `docs/DEMO_STREAMLIT.md`.

## Phạm vi và giới hạn

- Sleep-EDF Expanded 1.0.0, Sleep Cassette, EEG Fpz-Cz 100 Hz, epoch 30 giây.
- Năm lớp W/N1/N2/N3/REM; Movement/Unknown giữ vị trí thời gian nhưng bị mask khỏi loss/metrics.
- Train/validation/test tách theo đối tượng; hai đêm của cùng người luôn cùng vai trò.
- Có hai seed huấn luyện cố định 42 và 123 trên Sleep-EDF. Seed 123 được chạy sau khi đã xem kết quả
  seed 42; phân tích này đánh giá khả năng lặp lại của hướng hiệu ứng và độ nhạy, chưa đủ để ước lượng
  toàn bộ phân phối biến thiên do khởi tạo.
- Đã đánh giá zero-shot giới hạn trên 180 đối tượng SHHS1; chưa đại diện toàn bộ SHHS, chưa thích nghi
  miền, đa kênh hoặc xác nhận lâm sàng.
- Không có kiểm định tương đương hoặc không thua kém.

## Cấu trúc thư mục

```text
configs/          Cấu hình và giao thức đã khóa
data/manifests/   Manifest nguồn và báo cáo kiểm định dữ liệu
data/splits/      Split 10-fold theo đối tượng
data/processed/   Dữ liệu NPZ lớn, không lưu Git
docs/             Giao thức, runbook, trạng thái và báo cáo
runs/v2/          Phân tích nhẹ và gói công bố cuối; không lưu trọng số
scripts/          CLI tiền xử lý, huấn luyện, kiểm định và phân tích
src/sleeptcn/     Mã nguồn dùng chung
tests/            Kiểm thử tự động
```

## Phạm vi kết luận

- Các kết quả E0–E6 và Gate 8 được xem là các kết quả đã khóa; mọi phân tích mới phải dùng protocol riêng.
- Giá trị p lớn hơn 0,05 không được diễn giải như bằng chứng về tính tương đương hoặc không thua kém.
- Ablation C/P/N chỉ ước lượng hiệu ứng dự báo có điều kiện trong quy trình hiện tại, không đo tỷ lệ thông
  tin và không xác lập quan hệ nhân quả.
- Kết luận SHHS chỉ áp dụng cho cohort và giao thức zero-shot đã khóa; chưa có cơ sở để suy rộng sang toàn
  bộ SHHS hoặc thực hành lâm sàng.
