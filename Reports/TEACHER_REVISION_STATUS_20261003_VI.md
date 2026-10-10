# Trạng thái chỉnh sửa và thực nghiệm — 03-10-2026

> Bản chụp lịch sử ngày 03-10. Trạng thái này không còn là danh sách việc hiện tại:
> ngày 04-10 đã hoàn tất và kiểm chứng weighted/unweighted đủ mười fold trên SHHS.
> Xem `COLAB_SOURCE_TRAINING_20261004_VI.md` và phản hồi thầy đã cập nhật.
> ADAST mười fold cũng đã hoàn tất và kiểm chứng; xem
> `COLAB_ADAST_RESULTS_20261004_VI.md`. N3 cải thiện nhưng có đánh đổi ở N1/REM.

Đã hoàn tất các phần tính toán nhẹ trên CPU và cập nhật bộ bản thảo. Chưa chạy
huấn luyện weighted/unweighted hoặc ADAST đủ mười fold. Báo cáo này thay thế
danh sách việc còn thiếu ở bản trạng thái 02-10; các kết quả lịch sử vẫn được giữ nguyên.

## 1. Đã hoàn tất

| Hạng mục | Phạm vi và kết quả | Kiểm chứng |
|---|---|---|
| Khôi phục checkpoint | 80 checkpoint E3/E4, seed 42/123, mỗi seed đủ 10 fold và encoder/TCN | Hash, metadata, config, split và dấu hoàn tất khớp hiện vật Git; không checkout nhánh hay sửa checkpoint lịch sử. |
| E4 seed 42 trên SHHS | 180 người, 169.012 epoch chấm điểm, trung bình đủ 10 fold | 1.800 ma trận xác suất, metrics/CI và replay 20 trường hợp fold–bản ghi đã kiểm chứng. |
| Calibration/EM seed 123 | 10 fold, 180 bản ghi toàn phần; 183.528 epoch suy luận, 169.012 epoch chấm điểm | 1.800 ma trận xác suất; khớp lại temperature/prior của 10 fold; replay 40 trường hợp, gồm hai phía điểm gián đoạn; metrics và bootstrap khớp. |
| Mã huấn luyện có thể tiếp tục sau gián đoạn | Module cho cặp weighted/unweighted; kiểm thử trên chuỗi giả lập nhỏ | Trạng thái mô hình và lịch sử tối ưu sau resume khớp chạy liên tục; công thức huấn luyện khớp runner một fold cũ. Chưa phải kiểm thử chiến dịch GPU. |
| Kiểm thử toàn dự án | 201 test qua, 29 cảnh báo | Cảnh báo dependency và một cảnh báo thu thập test đã tồn tại; không có test thất bại. |
| Bản thảo và bộ nguồn | Anh 16 trang, Việt 16 trang, phụ lục Anh 9 trang | Đã dựng lại, đối chiếu tham chiếu/citation/label Anh–Việt, giữ các bảng lịch sử, rà 41 trang và các trang sửa ở kích thước đầy đủ. |

Tính toán chạy tại máy người dùng, `D:/SleepTCN/.venv`, PyTorch 2.5.1+cpu.
Dữ liệu SHHS được đọc từ `E:/research/Dataset/SHHS_v1`; kết quả mới nằm trong
`D:/SleepTCN/runs`. Không cài CUDA, không sử dụng cloud, không commit/push/nộp bài.

## 2. Kết quả E3–E4 và cách diễn giải

| Seed | E3: subject-mean macro-F1 | E4 | E4 − E3, CI 95% theo người |
|---|---:|---:|---|
| 42 | 0,568027 | 0,581110 | +0,013083 [0,011077; 0,015252] |
| 123 | 0,563400 | 0,573220 | +0,009820 [0,007298; 0,012503] |

Cả hai so sánh cùng seed nghiêng về E4 trên cùng quần thể SHHS. Hai seed không
phải hai quần thể độc lập. E3 được mô tả là tham chiếu chính của thiết kế ban đầu;
không còn khẳng định E3 là preprocessing tốt nhất khi chuyển quần thể. Phụ lục
thêm Bảng S11 để người đọc thấy cả hai seed cùng nhau, nhưng giữ riêng họ kiểm
định seed 123 cũ và CI mới của seed 42.

N3 recall seed 42 E4/E3 là 0,2728/0,2582. E4 tốt hơn về điểm tổng thể nhưng vẫn
bỏ sót nhiều N3. Phép chia hằng số dương giữ hình dạng sóng và tỷ lệ năng lượng
giữa các dải; không viết rằng phép chia 100 làm mất sóng chậm. Tương tác tối ưu
và BatchNorm chưa được xác lập bằng thực nghiệm nhân quả.

## 3. Calibration/EM đủ mười fold

| Chỉ số | Đối chứng E3 toàn bản ghi | Temperature | Temperature + EM |
|---|---:|---:|---:|
| Subject-mean macro-F1 | 0,558979 | 0,559468 | 0,489853 |
| Pooled macro-F1 | 0,598856 | 0,599418 | 0,502096 |
| N3 recall | 0,240989 | 0,242129 | 0 |
| N3 F1 | 0,383987 | 0,385453 | 0 |
| N2 recall | 0,730623 | 0,728696 | 0,728840 |

Temperature − raw: +0,000489, CI 95% [0,000021; 0,000970], mức thay đổi tuyệt
đối nhỏ. EM − temperature: −0,069615, CI [−0,079334; −0,059976]. Cả mười EM
hội tụ nhưng đặt prior N3 gần sàn số học; không còn dự đoán N3 và 98,83% N3
tham chiếu bị dự đoán thành N2. Không xem hội tụ là xác nhận giả định label shift,
không suy ra năm người adaptation không có N3.

Đối chứng ở bảng này dùng ngữ cảnh toàn bản ghi. Không ghép với E3 lịch sử từ
cửa sổ benchmark đã cắt để tính tác động calibration. Phần bổ sung nêu đúng
quá trình source-validation fit, adaptation không nhãn, đóng băng tham số và
chấm điểm sau suy luận. Tập xác thực nguồn đã từng được dùng chọn checkpoint;
đây không phải một bộ calibration độc lập với selection.

## 4. Các kết quả một fold đã đưa vào bản thảo

**Weighted TCN:** đối chứng không trọng số và can thiệp đều train mới với encoder
E3 fold 0 seed 123 cố định. N3 recall tăng 0,2432 → 0,3221; N2 recall giảm
0,7646 → 0,6708; subject-mean macro-F1 0,5431 → 0,5363. Báo cáo cả lợi ích và
đánh đổi; không gọi CI chứa 0 là bằng chứng tương đương.

**ADAST:** đối chiếu với source-only cùng backbone, khởi tạo, mẫu nguồn và ngân
sách 1.140 update mỗi nhánh. Subject-mean macro-F1 0,4755 → 0,4214; N3 recall
0,2657 → 0,2622; N1 recall ADAST chỉ 0,0077. Thiết lập này dùng năm người
adaptation và ngân sách giới hạn; chưa xếp hạng phương pháp với ensemble E3
mười fold hoặc đại diện mọi cấu hình ADAST.

Các kết quả nằm trong mục **Thí nghiệm bổ sung / Additional intervention results**.
Phụ lục S2 nêu giao thức, các Bảng S19–S21 tách calibration mười fold khỏi
weighted và ADAST một fold. Các protocol/hashes lịch sử vẫn giữ nguyên.

## 5. Việc còn lại, cần ngân sách tính toán riêng

1. Hoàn thiện orchestration/cache và huấn luyện cặp weighted/unweighted cho chín
   fold còn lại: dự kiến 18 TCN mới, encoder tương ứng đóng băng; cặp fold 0 chỉ
   được tái sử dụng sau khi kiểm tra giao thức và hiện vật khớp. Cấu hình đã có,
   module huấn luyện đã kiểm thử; runner chiến dịch đầy đủ chưa hoàn tất.
   Ước tính ban đầu 2–4 giờ CPU, chưa phải số đo chiến dịch thực tế.
2. Mở rộng ADAST/source-only theo fold và kiểm tra ngân sách thích nghi. Runner
   hiện chỉ hoàn tất thiết lập một fold; chưa có chiến dịch nhiều fold/seed đã chạy.
3. Khi chọn GPU, cần môi trường PyTorch CUDA, kiểm tra mô hình/dữ liệu và chạy
   kiểm chứng số học trước chiến dịch. Chưa có gói GPU đã kiểm thử để tuyên bố
   chạy nguyên lệnh là tái lập được trên máy khác.
4. Hoàn tất vai trò CRediT thực tế của Tri Nguyen và xác nhận bản thảo của bốn
   tác giả. Tên, email và khoa Hệ thống Thông tin đã được cập nhật; không tự gán
   vai trò đóng góp chưa được xác nhận.

Các lượt train nhiều fold tốn thời gian chưa được khởi chạy. Không còn tiến trình
train/inference được cố ý để chạy nền khi bàn giao.

## 6. Hiện vật

- `runs/teacher_revision_cpu_20261002/restored_checkpoints/restoration_manifest.json`
- `runs/teacher_revision_cpu_20261002/e4_seed42_extension/{aggregate_results,verification}.json`
- `runs/teacher_revision_cpu_20261003/calibration_10fold_seed123/{aggregate_results,verification}.json`
- `runs/teacher_revision_cpu_20261001/weighted_e3_fold0/{aggregate_results,verification}.json`
- `runs/teacher_revision_cpu_20261001/adast_fold0/{aggregate_results,verification}.json`
- `tmp/teacher_revision_20261003/qa.json`, `visual_review.json`, `delivery_verification.json`
- `Reports/output/pdf/SleepTCN_Scientific_Article_EN.pdf`
- `Reports/output/pdf/SleepTCN_BSPC_Ban_dich_Tieng_Viet.pdf`
- `Reports/output/pdf/SleepTCN_Supplement_EN.pdf`
- `Reports/output/source/SleepTCN_BSPC_Manuscript_Source.zip`
- `Reports/analysis/teacher_revision_20261003.json`: chỉ số cấp quần thể, ma trận
  nhầm lẫn gộp và CI của bốn thực nghiệm đã kiểm chứng; không có ID/dự đoán cấp người.

Tệp cấp người tham gia nằm trong thư mục bị gitignore và không đi vào ZIP bản thảo.
Bản trước khi sửa được giữ trong `tmp/teacher_revision_20261003/before`.
Gói ZIP đã được dựng thử trong thư mục riêng: số trang và toàn bộ text từng trang
của bản Anh/phụ lục khớp bản đã rà; các hash trong hai manifest bàn giao đều khớp.
