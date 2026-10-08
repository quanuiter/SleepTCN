# SleepTCN Explorer — demo bảo vệ khóa luận

Demo này được cố ý giữ gọn. Nó không thay thế báo cáo và không trình bày toàn bộ so sánh E0/E3/E6, kiểm định hay Gate 1–8. Mục tiêu là để hội đồng nhìn trực tiếp một đêm ngủ EEG và hiểu mối liên hệ giữa **giai đoạn ngủ**, **epoch 30 giây** và **đường sóng EEG**.

## Demo làm ba việc

1. Thống kê số epoch, thời lượng và tỷ lệ của W/N1/N2/N3/REM.
2. Hiển thị hypnogram bậc thang có vùng tô của toàn bộ đêm ngủ; trục trái là W/REM/N1/N2/N3.
3. Cho chọn một epoch để xem 30 giây EEG và giai đoạn ngủ tương ứng.

Sóng EEG luôn được vẽ từ cùng tín hiệu gốc của bản ghi. Khi đổi E3/E0, chỉ prediction, thống kê và hypnogram thay đổi; không đổi đoạn sóng đang quan sát.

## Hai nguồn bản ghi

- **Bản ghi mẫu:** dùng bốn bản ghi Sleep-EDF và prediction artifact test đã khóa. Chọn một trong hai model E3/E0; mọi giai đoạn trên màn hình là **dự đoán của model đã chọn**.
- **Tải EDF:** nhận EDF có `EEG Fpz-Cz`, 100 Hz và đơn vị µV; chọn E3 hoặc E0 trước khi chạy. Có thể tải thêm tệp Hypnogram EDF để đối chiếu. Các biểu đồ chính vẫn hiển thị **dự đoán**; nhãn chuyên gia được ghi riêng trong phần đối chiếu, không phải kết quả chẩn đoán.

## Tải nhãn chuyên gia

1. Tải tệp EEG trước, chẳng hạn `SC4001E0-PSG.edf`.
2. Trong ô **Tải nhãn chuyên gia (tùy chọn)**, chọn tệp tương ứng, chẳng hạn `SC4001EC-Hypnogram.edf`.
3. Bấm **Phân tích bản ghi**. Ứng dụng chạy mô hình mà không nhận nhãn chuyên gia; nhãn chỉ được gắn vào kết quả sau suy luận để tính tỷ lệ khớp và đối chiếu từng đoạn.
4. Dùng bộ lọc **Dự đoán đúng** hoặc **Dự đoán sai** để tìm ví dụ. Có thể thêm, thay hoặc bỏ tệp nhãn sau khi chạy mà không phải suy luận lại.

Phiên bản này hỗ trợ chú giải Sleep-EDF Expanded: W, 1, 2, 3, 4, R, Movement time và ?. Giai đoạn 3 và 4 được gộp thành N3 theo giao thức năm lớp của dự án. Nhãn vận động, chưa chấm và khoảng trống chú giải được đặt là không hợp lệ, không đưa vào mẫu số đánh giá. Chú giải sau khi tín hiệu đã kết thúc không được dùng. Chưa hỗ trợ CSV hoặc tệp chú giải SHHS.

Ứng dụng kiểm tra mã bản ghi trong tên Sleep Cassette, yêu cầu thời điểm bắt đầu trong hai tiêu đề EDF khớp nhau và chú giải nằm đúng ranh giới 30 giây. Tệp sai thời gian, nhãn lạ, chú giải chồng lấn hoặc không có nhãn hợp lệ bị từ chối, không tự dịch hoặc làm tròn nhãn. Nếu đã đổi tên tệp, người dùng phải bảo đảm chúng thuộc cùng người và lần ghi; chỉ trùng thời gian không chứng minh được nguồn gốc.

Tỷ lệ khớp chỉ mô tả các đoạn có nhãn hợp lệ của bản ghi tải lên. Dù có nhãn, ứng dụng vẫn suy luận trên toàn bộ đoạn hoàn chỉnh, không cắt biên theo nhãn, nên không coi kết quả này là tái lập trực tiếp chỉ số trong khóa luận.

Nguồn mô tả tệp và chú giải: [Sleep-EDF Expanded trên PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/).

Với bản ghi mẫu, có thể bật **Đối chiếu với nhãn chuyên gia** để xem nhãn tham chiếu bên cạnh dự đoán tại cùng một đoạn. Bộ lọc cho phép tìm đoạn dự đoán đúng hoặc sai; đoạn không có nhãn hợp lệ không được đưa vào hai nhóm này. Đối chiếu tại một đoạn không thay thế đánh giá trên toàn bộ tập kiểm tra.

Không có lựa chọn tệp mô hình, so sánh đồng thời nhiều mô hình hoặc bảng thống kê nghiên cứu trong giao diện. Các phần đó thuộc báo cáo khóa luận. Kết quả lưu sẵn và suy luận trực tiếp được ghi rõ để tránh nhầm lẫn.

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

## Kịch bản trình bày 2 đến 3 phút

1. Chọn một bản ghi mẫu và một model (mặc định E3).
2. Chỉ vào biểu đồ cột và bảng: mỗi giai đoạn chiếm bao nhiêu epoch và bao nhiêu phút.
3. Chỉ vào hypnogram: vùng bậc thang thay đổi theo thời gian cho thấy cấu trúc của cả đêm ngủ.
4. Bật đối chiếu với chuyên gia. Chọn một đoạn dự đoán đúng, rồi một đoạn dự đoán sai; giải thích giới hạn của mô hình trên ví dụ cụ thể.
5. Chuyển sang Tải EDF, chọn tệp hợp lệ đã chuẩn bị và bấm Phân tích bản ghi. Nêu rõ đây là suy luận trực tiếp, khác với kết quả lưu sẵn của bản ghi mẫu. Thời gian hiển thị chỉ tính suy luận, không gồm nạp mô hình và tiền xử lý.
6. Chốt rằng đánh giá và so sánh đầy đủ được trình bày trong báo cáo.

## Kiểm tra trước buổi bảo vệ

- Chạy thử trên đúng máy trình diễn, không phụ thuộc Internet sau khi đã chuẩn bị môi trường, mô hình và dữ liệu.
- Chuẩn bị một EDF được phép sử dụng, đúng kênh, tần số và đơn vị; không đưa dữ liệu người tham gia lên dịch vụ công khai.
- Kiểm tra cả E3 và E0, bộ lọc đúng/sai, giai đoạn không có đoạn phù hợp và tệp hỏng.
- Khi chưa tải nhãn chuyên gia, ứng dụng không hiển thị độ chính xác đối chiếu. Có hoặc không có nhãn đều không áp dụng cắt biên theo nhãn như giao thức thực nghiệm.
- Đoạn cuối chưa đủ 30 giây bị bỏ và ứng dụng thông báo số mẫu tín hiệu bị bỏ.
- Chuẩn bị video ngắn hoặc ảnh minh họa dự phòng, ghi rõ đó là bản ghi lại nếu cần sử dụng.

Demo là công cụ trực quan hóa nghiên cứu, không phải thiết bị y tế hoặc hệ thống chẩn đoán.
