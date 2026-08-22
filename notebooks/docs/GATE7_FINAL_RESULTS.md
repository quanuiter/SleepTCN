# Gate 7 — gói công bố và bản thảo khoa học

> Lưu ý trạng thái (2026-08-22): tài liệu này đóng băng kết luận tại thời điểm Gate 7, khi mới có seed 42
> và chưa có SHHS. Kết quả seed 123 và SHHS về sau không nhập ngược vào Gate 7; xem `STATUS_V2.md` và
> `MULTISEED_SENSITIVITY_RESULTS.md` để biết phạm vi hiện hành.

Ngày khóa: **2026-08-14**  
Nhánh: `run-in-docker`  
Trạng thái: **ĐẠT**

## Phạm vi

Gate 7 không huấn luyện lại mô hình, không mở lại test và không thay đổi E0–E6. Cổng này chỉ chuyển các artifact
đã khóa ở Gate 5–6 thành bảng, hình, ma trận bằng chứng–tuyên bố và bản thảo tiếng Việt có thể truy vết.

Trình tạo tái lập:

```powershell
python scripts/build_gate7_publication_package.py `
  --workspace D:\SleepTCN `
  --output-dir D:\SleepTCN\runs\v2\publication\gate7
```

Trình kiểm định độc lập:

```powershell
python scripts/validate_gate7_artifacts.py `
  --package-dir D:\SleepTCN\runs\v2\publication\gate7 `
  --output D:\SleepTCN\runs\v2\publication\gate7\gate7_validation_report.json
```

## Đầu ra đã tạo

Thư mục `runs/v2/publication/gate7/` chứa:

- bốn bảng CSV và bản tổng hợp `TABLES.md`;
- ba hình PNG: hiệu ứng chính, đánh đổi hiệu năng–tốc độ–tham số và Silhouette bắt cặp;
- `claim_evidence_matrix.json` cùng bản đọc `CLAIM_EVIDENCE_MATRIX.md`;
- `MANUSCRIPT_DRAFT_VI.md` gồm Tóm tắt, Đặt vấn đề, Phương pháp, Kết quả, Thảo luận, Hạn chế và Kết luận;
- `AUTHOR_CHECKLIST.md`, `publication_manifest.json` và `gate7_validation_report.json`.

Manifest ghi SHA-256 của năm đầu vào Gate 5–6 và 12 đầu ra chính, đồng thời lưu phiên bản Python, NumPy và
Matplotlib đã kết xuất hình. Báo cáo kiểm định xác nhận đủ sáu thí nghiệm, năm so sánh, bốn so sánh chính,
10 fold đặc trưng và tám phát biểu khoa học.

## Kết luận được phép sử dụng

- E3 có Macro-F1 out-of-fold mô tả cao nhất trong sáu cấu hình: **0,7904**.
- E3 tốt hơn E6: chênh lệch **0,0213**, CI 95% **[0,0122; 0,0307]**, p Holm **0,0012**.
- E1−E0 và E2−E1 chỉ tăng nhẹ về mô tả; chưa đủ bằng chứng sau hiệu chỉnh Holm.
- ResNet-1D–TCN nhanh hơn E0 khoảng **3,76 lần** trên giao thức V100 đã khóa, nhưng có **4,37 lần** số tham
  số và peak VRAM cao hơn **28,4%**.
- Silhouette E2 thấp hơn E1 trong **10/10 fold**; không được dùng kết quả này để tuyên bố embedding ResNet
  phân tách lớp tốt hơn.

## Ranh giới diễn giải

- Kết quả chỉ áp dụng in-domain trên Sleep-EDF Expanded và một training seed 42.
- Chưa đánh giá SHHS, domain shift, zero-shot, đa kênh hay giá trị lâm sàng.
- “Đơn giản hóa” chỉ có nghĩa là ít mô hình thành phần hơn và vận hành gọn hơn; không có nghĩa ít tham số hay
  ít VRAM hơn.
- Không thay các số liệu sinh tự động bằng kết quả chép tay. Khi sửa Tóm tắt/Kết luận phải đối chiếu ma trận
  bằng chứng–tuyên bố.

## Bước kế tiếp

Phần thực nghiệm chính trên Sleep-EDF đã khép kín. Công việc tiếp theo là biên tập học thuật: bổ sung trích dẫn,
đối chiếu phương pháp với paper gốc, chuẩn hóa thuật ngữ/hình/bảng theo nơi nộp và nhờ giảng viên duyệt phạm vi
tuyên bố. Nếu cần củng cố thực nghiệm, ưu tiên một chiến dịch nhiều seed được đăng ký trước; không trộn kết quả
mới vào chiến dịch seed 42 đã khóa.
