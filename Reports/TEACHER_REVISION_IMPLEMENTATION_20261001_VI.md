# Kết quả thực hiện chỉnh sửa theo góp ý thầy Trí

> Hồ sơ ngày 01-10. Các thí nghiệm bổ sung đã được thực hiện sau đó; xem
> `TEACHER_REVISION_STATUS_20261003_VI.md` và `TEACHER_RESPONSE_20261003_VI.md`.
> Những câu “chưa chạy” và “thiếu checkpoint” bên dưới mô tả thời điểm ban đầu,
> không phải tình trạng bản thảo hiện tại.

Ngày 01-10-2026. Trạng thái: **đã hoàn tất phần biên tập, kiểm tra đầu vào và công cụ pilot; chưa hoàn tất các thực nghiệm mới**. Không commit/push, không nộp bài, không thay checkpoint/dự đoán/giao thức lịch sử.

## Đã thực hiện

- Sửa paper tiếng Anh và bản dịch tiếng Việt hiện hành: abstract đã nêu E4 so với E3 cùng seed 123; bổ sung CI/đối chiếu cùng seed vào Results; giới hạn contribution là lợi ích E3 so với đối chứng CNN–BiLSTM; làm rõ E3 là tham chiếu chính, không phải preprocessing đã chứng minh tối ưu.
- Bỏ cách quy toàn bộ lợi ích E3–E2 cho phép chia 100: đây là gói lọc/clip/fixed-scale so với raw. Phân biệt phép đổi thang E3/E4 với các mô hình được huấn luyện riêng. Không gán cơ chế mất sóng chậm hoặc đặc tính vật lý của thiết bị SHHS khi chưa có đo kiểm.
- Bổ sung mô tả N3 của E4 seed 123 từ hiện vật đã lưu: recall 0,2591, F1 0,4068; E3 cùng seed là 0,2410 và 0,3840. Đây là tái kiểm tra dự đoán đã có, không phải chạy mô hình/can thiệp mới. E4 tốt hơn tổng thể nhưng vẫn bỏ sót N3.
- Đồng bộ highlights: không nhấn quá mức chênh lệch bộ nhớ; nêu rõ kết quả E4 cùng seed là hậu nghiệm. Không thay số đo tài nguyên trong bảng.
- Thêm Tri Nguyen Ho Duy (`trinhd@uit.edu.vn`) và Tri Nguyen (`tringuyen@uit.edu.vn`) sau hai tác giả hiện tại; giữ nguyên tác giả liên hệ. Đồng bộ main EN/VI, supplement, running headers và PDF metadata; bỏ lời cảm ơn riêng cho người đã chuyển thành đồng tác giả.
- Đặt seed trước khởi tạo sequence model; kiểm thử cùng seed/different ambient RNG cho trọng số ban đầu giống nhau, đổi campaign seed thì có trọng số khác. Đây là sửa cho lượt chạy tương lai; chưa chứng minh là nguyên nhân của kết quả cũ.
- Kiểm thử tính nhất quán của ResNet eval: chia input cho 100 và nhân trọng số convolution stem không bias với 100 cho đầu ra gần như cũ, kể cả với running statistics BN không mặc định. Đây là kiểm tra số học trên mô hình thử nghiệm, không phải thí nghiệm cải thiện hay kết quả từ checkpoint huấn luyện đầy đủ.
- Thêm module CPU `src/sleeptcn/calibration.py`, CLI `scripts/run_calibration_pilot.py` và kiểm thử: temperature calibration, EM prior adjustment, ba nhánh raw/calibrated/calibrated+EM; từ chối trường nhãn đích/cửa sổ adaptation khai báo phụ thuộc nhãn; lưu mã băm input/code và không ghi đè output.
- Chuẩn bị utility tính class weights từ train labels cho nhánh weighted TCN; **chưa nối vào historical runner hoặc huấn luyện weighted TCN**.
- Viết giao thức pilot `docs/TEACHER_REVISION_EXECUTION_V1.md`: calibration theo fold, thứ tự trước ensemble, selection nguồn, dữ liệu adaptation không cần nhãn, metric N3 và các đối chứng. Không tự gọi temperature scaling là tái lập bias-corrected calibration.

## Kiểm tra hiện vật trên CPU đã chạy

Công cụ: `scripts/audit_teacher_revision_inputs.py`; output aggregate-only ở `tmp/teacher_revision_20261001/input_audit_final.json`. Tệp này không chứa mã định danh đối tượng; không xuất participant-level artifacts sang gói nộp. Liên kết mã băm run manifest → inventory/protocol → seed 123 cũng được xác minh.

| Kiểm tra | Kết quả |
|---|---|
| EDF, cặp tín hiệu E3/E4 | 153/153 bản ghi khớp nhãn, epoch và hash nguồn |
| SHHS, cặp tín hiệu E3/E4 | 200/200 bản ghi khớp nhãn, epoch và hash nguồn; clip_fraction E3 bằng 0 |
| Quan hệ số học | `100*x_E3 ~= x_E4` đạt mọi mẫu đã lưu, tolerance `2e-5 + 3e-7*abs(x_E4)` microvolt |
| Sai số tuyệt đối lớn nhất EDF/SHHS | 0,00001812 / 0,00002670 microvolt, phù hợp float32 |
| E3/E4 SHHS seed 123 | 360/360 file hash khớp; argmax, nhãn/epoch, confusion và subject-mean tái tính khớp manifest |
| Subject-mean macro-F1 | E3 0,56339956; E4 0,57321999; E4−E3 0,00982043 |

Kiểm tra tín hiệu chỉ bao phủ các cửa sổ benchmark đã lưu, không phải mọi mẫu liên tục trước cắt. Không bootstrap lại CI hoặc tạo dự đoán mới trong lượt này. CI/p trong manuscript lấy từ phân tích bắt cặp đã lưu. Kết quả này xác nhận khác biệt đầu vào, không xác lập cơ chế tối ưu hóa/BN hoặc cơ chế sinh lý.

## Tài nguyên thực sự còn thiếu

Thông tin review trước về ổ E: không truy cập được đã hết đúng ở phiên thực hiện này. Hiện có full EDF processed (153 bản ghi mỗi biến thể), SHHS processed và các dự đoán SHHS lưu trữ ở E:. Kiểm tra phải dùng cả file bị gitignore: `rg --files` mặc định không đủ để kết luận không có checkpoint.

- `runs/v2/full`: 0 checkpoint `.pt`, 0 NPZ dự đoán nguồn. 42 checkpoint trong `runs/smoke` và 2 ở cây smoke khác chỉ phục vụ smoke, không thay thế mô hình đầy đủ.
- Inventory SHHS chính: 0/200 checkpoint có ở đường dẫn đã ghi; inventory component: 0/180; inventory seed 123 gồm E4: 0/240. Các inventory có phần trùng nhau; **không cộng** thành số checkpoint riêng biệt. Đây là tình trạng truy cập tại đường dẫn, không kết luận dữ liệu đã bị mất.
- Runtime PyTorch `2.5.1+cpu`, `cuda_available=False`. Điều này xác nhận runtime chưa dùng CUDA, không phải kết luận máy không có phần cứng GPU.

| Công việc tiếp theo | Đầu vào cần khôi phục/chuẩn bị | Tài nguyên tính toán |
|---|---|---|
| E4 seed 42 trên SHHS | 10 encoder + 10 TCN checkpoint E4 seed 42 và metadata/hash, inventory/protocol mở rộng riêng | CPU chạy được; GPU tăng tốc inference, không bắt buộc train lại |
| Calibration + EM thật | Xác suất/logits + nhãn validation nguồn đúng từng fold, checkpoint/split provenance; xác suất adaptation chọn cửa sổ không nhãn | CPU đủ cho fitting/EM; cần inference thêm nếu chưa có đầu ra đủ chuẩn |
| Weighted TCN | Encoder checkpoint/cache full đủ train/validation mỗi fold; runner/config chiến dịch mới và đối chứng retrain | GPU nên có để train; chưa có runner chiến dịch mới |
| ADAST source-only + UDA | Audit/pin implementation, adapter dữ liệu và selection rule, dữ liệu nguồn + đích adaptation không nhãn | Môi trường CUDA và lưu trữ; chưa tích hợp/chạy |
| Xác nhận phương án đã chọn | Holdout chưa mở, protocol sampling/cỡ mẫu và nhánh baseline tương ứng | Tùy inference/training; không thay bằng thêm seed |

NPZ SHHS hiện được cắt bằng nhãn first/last true sleep. Bỏ trường `y` khỏi file **không** làm quy trình chọn cửa sổ thành label-free. Vì vậy không chạy EM/UDA trên chúng rồi tuyên bố adaptation hoàn toàn không nhãn. Không chạy EM chưa calibration để thay thế nhánh đã chốt khi thiếu validation nguồn.

## Kiểm chứng mã và tài liệu

Bộ test hiện hành chạy trên CPU bằng `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`; 182 test đạt, 29 cảnh báo (dependency deprecations, collection warning và phép Wilcoxon trên mẫu thử nhỏ), không có test fail. Kết quả và QA cuối cùng được lưu trong `tmp/teacher_revision_20261001`. Các test calibration/EM dùng dữ liệu tổng hợp và không chứng minh N3 trên SHHS sẽ cải thiện. Bộ test không thay thế tái lập đầy đủ/resume dài hạn.

Main EN/VI được build lại bằng MiKTeX đã có, supplement cũng build lại. Kiểm tra giữ nguyên toàn bộ bảng khoa học, thứ tự cite/ref/label, không có undefined references hoặc overfull; xem toàn bộ trang bằng ảnh render và các trang chỉnh sửa ở kích thước đầy đủ. Không thêm biểu đồ mới khi chưa được xem trước. ZIP nộp chỉ thay nguồn/metadata đã sửa, không kèm công cụ pilot hay dữ liệu hạn chế. Mã băm experimental configs/runs lịch sử giữ nguyên.

## Xác nhận tác giả còn thiếu trước khi nộp

Thứ tự đã được tác giả xác nhận: Quân → Sơn → Tri Nguyen Ho Duy → Tri Nguyen. Tác giả cũng xác nhận hai thầy cùng Khoa Hệ thống Thông tin, nên EN/VI đã dùng chung affiliation đó. Còn cần xác nhận CRediT của Tri Nguyen. Vai trò Supervision/Writing–review của Tri Nguyen Ho Duy dựa trên lời cảm ơn hướng dẫn đã có và góp ý manuscript được cung cấp; vẫn cần đồng tác giả duyệt. Chưa tự gán đóng góp của Tri Nguyen hoặc đổi tác giả liên hệ.

Bản hiện hành **chưa được đánh dấu sẵn sàng Submit**: còn xác nhận tác giả/declarations và các thực nghiệm bổ sung nếu nhóm theo mục tiêu nghiên cứu mới của thầy. Bản bàn giao readiness cũ đã có cảnh báo để tránh coi trạng thái lịch sử là trạng thái hiện tại.
