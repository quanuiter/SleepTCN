# Trạng thái xử lý góp ý thầy Trí — 02-10-2026

> Hồ sơ theo thời điểm 02-10. Trạng thái mới được cập nhật trong
> [TEACHER_REVISION_STATUS_20261003_VI.md](TEACHER_REVISION_STATUS_20261003_VI.md).
> Nhận định thiếu checkpoint bên dưới đã được thay thế: đã khôi phục đủ E3/E4,
> hai seed, mười fold từ lịch sử Git và hoàn tất E4 seed 42. Giữ nội dung cũ để
> truy vết tiến trình, không dùng làm danh sách việc còn thiếu hiện tại.

Đã chạy và kiểm chứng các pilot khả thi trên CPU. Báo cáo số liệu/giao thức đầy đủ: [CPU_INTERVENTION_PILOTS_20261001_VI.md](CPU_INTERVENTION_PILOTS_20261001_VI.md). Các kết quả mới là thăm dò một fold, seed 123, trên cohort SHHS đã được xem trước đó; chưa cập nhật PDF/ZIP nộp bài bằng các con số này.

## 1. Preprocessing E3–E4

Đã sửa lập luận trong bản thảo: E3 là tham chiếu theo thiết kế lịch sử, không phải preprocessing tối ưu đã được chứng minh. Đối chiếu cùng seed 123 cho subject-mean macro-F1 E4 = 0,573220, E3 = 0,563400; chênh +0,009820, CI 95% [0,007298; 0,012503]. Không dùng E3 seed 42 để ghép với E4 seed 123 thành đối chiếu cùng điều kiện.

Đã kiểm tra quan hệ số học `100*x_E3 ≈ x_E4` trên cửa sổ tín hiệu lưu trữ; không dựng lời giải thích vật lý rằng phép chia 100 làm mất sóng chậm. Nguyên nhân chênh lệch ở các mô hình được huấn luyện riêng chưa được xác định nhân quả. Còn cần checkpoint E4 phù hợp để chạy kiểm tra bổ sung và activation/BN theo đúng nguồn gốc. GPU không tự giải quyết việc thiếu checkpoint.

## 2. Can thiệp N3

- **Temperature + EM:** đã chạy. EM đẩy prior N3 sát floor và không còn dự đoán N3; macro-F1 theo người giảm 0,5585 → 0,5023. Không xem hội tụ thuật toán là bằng chứng cải thiện.
- **Weighted TCN:** đã huấn luyện mới cả đối chứng và can thiệp với encoder cố định. N3 recall tăng 0,2432 → 0,3221, N3 F1 tăng 0,3864 → 0,4779; nhưng N2 recall giảm 0,7646 → 0,6708 và subject-mean macro-F1 giảm 0,5431 → 0,5363. CI của chênh lệch subject-mean macro-F1 chứa 0, không chứng minh tương đương/non-inferiority.

Như vậy đã chuyển từ chỉ chẩn đoán lỗi sang kiểm tra can thiệp thực tế. Tuy nhiên, chưa giải quyết được N3 mà không đánh đổi hiệu năng khác. Không nên viết “khắc phục N3”, “cải thiện toàn diện”, hoặc thay mô hình cuối chỉ vì recall N3 tăng.

## 3. Baseline UDA ngoài

Đã triển khai và chạy ADAST với source-only cùng backbone, cùng khởi tạo/mẫu nguồn và 1.140 updates mỗi nhánh. Dùng năm người adaptation không nhãn, suy luận toàn bản ghi trên đủ 180 người test rồi mới áp mask chấm điểm. Giữ model classes, similarity penalty, lịch loss và quy tắc inference upstream; preprocessing đồng nhất theo dự án. Vì vậy gọi là **pilot ADAST theo giao thức chung**, không gọi là tái lập nguyên bản hoặc benchmark SOTA hoàn chỉnh.

Subject-mean macro-F1 source-only/ADAST: 0,4755/0,4214; N3 recall 0,2657/0,2622. N1 recall ADAST chỉ 0,0077 trên SHHS và cũng rất thấp trên outer test nguồn. Trong thiết lập này adaptation không có lợi; không suy rộng thành kết luận ADAST hoặc UDA nói chung không hiệu quả. Không dùng kết quả một fold để tuyên bố vượt phương pháp ngoài bằng ensemble mười fold của E3.

## 4. Những việc còn lại trước khi tuyên bố đã xử lý đầy đủ

1. Khôi phục checkpoint E4 và chín fold encoder còn lại, hoặc huấn luyện lại trong chiến dịch mới có provenance riêng. Không có đủ hiện vật này để chạy campaign đầy đủ bằng các checkpoint lịch sử ngay bây giờ.
2. Mở rộng và chốt campaign nhiều fold/seed cho can thiệp và baseline ngoài nếu muốn đưa ra kết luận ở cùng mức với thực nghiệm chính. Runner hiện được khóa cho pilot fold 0; chưa xác nhận ổn định theo seed hoặc ngân sách adaptation khác. Cần thời gian chạy/lưu trữ; GPU là lựa chọn tăng tốc, không phải yêu cầu bắt buộc đã được chứng minh.
3. Khi tích hợp vào bài, tách rõ mục thực nghiệm thăm dò sau quan sát và giữ các kết quả âm/đánh đổi. Không trộn số liệu pilot với ensemble lịch sử, không chỉnh tham số theo SHHS test để tìm bảng đẹp hơn.
4. Vai trò CRediT thực tế của tác giả bổ sung còn cần người viết xác nhận; khoa/đơn vị đã được xác nhận là Hệ thống Thông tin, UIT–ĐHQG TP.HCM. Không tự gán vai trò khoa học khi chưa có thông tin.

## 5. Hiện vật để tiếp tục

Các thư mục sau nằm trong `runs` bị gitignore; không đưa file cấp người vào gói nộp:

- `runs/teacher_revision_cpu_20261001/recovered_e3_fold0`: calibration/EM, `aggregate_results.json`, `verification.json`.
- `runs/teacher_revision_cpu_20261001/weighted_e3_fold0`: hai checkpoint đã chọn, đủ 180 dự đoán, `aggregate_results.json`, `verification.json`, `recovery_replay_verification.json`.
- `runs/teacher_revision_cpu_20261001/adast_fold0`: hai checkpoint cuối, đủ 180 dự đoán, `aggregate_results.json`, `verification.json`.

Không có tiến trình train/inference còn được cố ý để chạy nền sau bàn giao. Không commit, push, nộp bài hoặc thay đổi các manifest lịch sử trong lượt này.
