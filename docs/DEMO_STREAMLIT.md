# SleepTCN Explorer — demo bảo vệ khóa luận

Demo này được cố ý giữ gọn. Nó không thay thế báo cáo và không trình bày toàn bộ so sánh E0/E3/E6, kiểm định hay Gate 1–8. Mục tiêu là để hội đồng nhìn trực tiếp một đêm ngủ EEG và hiểu mối liên hệ giữa **giai đoạn ngủ**, **epoch 30 giây** và **đường sóng EEG**.

## Demo làm ba việc

1. Thống kê số epoch, thời lượng và tỷ lệ của W/N1/N2/N3/REM.
2. Hiển thị hypnogram bậc thang có vùng tô của toàn bộ đêm ngủ; trục trái là W/REM/N1/N2/N3.
3. Cho chọn một epoch để xem 30 giây EEG và giai đoạn ngủ tương ứng.

Sóng EEG luôn được vẽ từ cùng tín hiệu gốc của bản ghi. Khi đổi E3/E0, chỉ prediction, thống kê và hypnogram thay đổi; không đổi đoạn sóng đang quan sát.

## Hai nguồn bản ghi

- **Bản ghi mẫu:** dùng bốn bản ghi Sleep-EDF và prediction artifact test đã khóa. Chọn một trong hai model E3/E0; mọi giai đoạn trên màn hình là **dự đoán của model đã chọn**.
- **Tải EDF:** nhận EDF có `EEG Fpz-Cz`, 100 Hz và đơn vị µV; chọn E3 hoặc E0 trước khi chạy. Mọi giai đoạn ở màn hình này phải được gọi là **dự đoán**, không phải nhãn chuyên gia hay kết quả chẩn đoán.

Không có lựa chọn checkpoint, không có so sánh đồng thời nhiều model, không có bộ lọc lỗi và không có bảng thống kê nghiên cứu trong giao diện. Các phần đó thuộc báo cáo khóa luận.

## Chuẩn bị và chạy

Khuyến nghị Python 3.11:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[demo,test]"
python -m streamlit run demo/app.py
```

Chuẩn bị prediction artifact và checkpoint local đã xác minh SHA-256 (cần cho E3/E0):

```powershell
python scripts/prepare_demo_assets.py --ref run-in-docker --fold 0 --seed 123
```

`demo/assets/` bị Git ignore; không commit checkpoint, prediction hoặc dữ liệu người tham gia.

## Kịch bản trình bày 2 phút

1. Chọn một bản ghi mẫu và một model (mặc định E3).
2. Chỉ vào biểu đồ cột và bảng: mỗi giai đoạn chiếm bao nhiêu epoch và bao nhiêu phút.
3. Chỉ vào hypnogram: vùng bậc thang thay đổi theo thời gian cho thấy cấu trúc của cả đêm ngủ.
4. Lọc một giai đoạn, chẳng hạn N3, rồi kéo thanh chọn một epoch. Đọc dự đoán E3/E0 ở bên phải và xem đúng 30 giây EEG ở bên trái.
5. Chốt rằng đánh giá/so sánh đầy đủ được trình bày trong báo cáo, không nhồi vào demo trực quan này.

Demo là công cụ trực quan hóa nghiên cứu, không phải thiết bị y tế hoặc hệ thống chẩn đoán.
