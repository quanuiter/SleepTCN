# Trạng thái dự án

> **Hồ sơ lịch sử v1.** Nội dung này phản ánh mốc E0--E3 trước giao thức v2. Trạng thái hiện hành
> và bước tiếp theo nằm trong `STATUS_V2.md` và `NEXT_STEPS.md`.

## Đã hoàn thành

- [x] Kiểm kê 153 cặp PSG/Hypnogram của 78 đối tượng và mã băm nguồn.
- [x] Kiểm định metadata EDF 153/153 bản ghi.
- [x] Sinh và kiểm định 153 NPZ `paper_raw_v1` và 153 NPZ `filtered_v2`.
- [x] Giữ đúng 298 epoch Movement/Unknown bằng nhãn `-1` ở vị trí thời gian gốc.
- [x] Khóa manifest 10-fold seed 42 theo đối tượng; train/validation/test không giao nhau.
- [x] Khóa bộ nạp, kiến trúc SleepCNN/BiLSTM/ResNet-1D/TCN, mặt nạ và chỉ số đánh giá.
- [x] Khóa phép dịch current/previous/next trong biên từng bản ghi và thứ tự 75 đặc trưng 15CNN.
- [x] Viết vòng huấn luyện, dừng sớm, checkpoint nguyên tử và tiếp tục từ epoch hoàn tất cuối.
- [x] Kiểm thử huấn luyện liên tục và huấn luyện tiếp tục cho trọng số giống tuyệt đối trên CPU.
- [x] Viết kho đặc trưng có mã băm và dự đoán có `subject_id`, `record_key`, `original_epoch_index`.
- [x] Nối đầy đủ E0, E1, E2, E3; E1 bắt buộc dùng lại 15CNN của E0.
- [x] Khóa tập test khỏi API huấn luyện; smoke mode không thể mở test.
- [x] Toàn bộ 47 kiểm thử tự động đạt.
- [x] Smoke run CPU bằng dữ liệu thật đạt cho E0–E3.
- [x] Bốn báo cáo kiểm định độc lập đạt: checkpoint, extractor, nhãn, chỉ số epoch và metrics đều khớp.

## Mốc Git

- `9ffe786`: tiền xử lý đã kiểm định.
- `dbcc103`: chia fold và hợp đồng mô hình CPU.
- `4e7113f`: trình chạy huấn luyện E0–E3, checkpoint, artifact và notebook GPU.

## Việc còn lại trước thí nghiệm chính

1. Tạo môi trường GPU Python 3.11 theo nhà cung cấp được chọn.
2. Chạy `check_environment.py --require-gpu` trong chính container/notebook đó.
3. Chạy smoke GPU E0–E3 ở fold 0, không mở test.
4. Đo thời gian, VRAM và dung lượng cache/checkpoint để lập ngân sách 10 fold.
5. Chỉ khi bốn smoke GPU và kiểm định artifact đều đạt mới bắt đầu thí nghiệm đầy đủ.

Chưa có kết quả mô hình chính thức. Các metrics trong `runs/smoke/` không được dùng trong khóa luận hoặc bài báo.
