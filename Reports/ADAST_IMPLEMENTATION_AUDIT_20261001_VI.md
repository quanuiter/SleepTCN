# Kiểm tra implementation ADAST và khả năng chạy CPU

**Cập nhật 02-10-2026:** sau kiểm tra thành phần dưới đây, đã triển khai và hoàn tất cặp ADAST/source-only fold 0, seed 123 theo giao thức chung; 1.140 update/nhánh, đánh giá đủ 180 người SHHS và kiểm chứng passed. Subject-mean macro-F1 source-only/ADAST: 0,475542/0,421359. Xem mục 5 của `CPU_INTERVENTION_PILOTS_20261001_VI.md` cho chi tiết và giới hạn; kết quả này không xác lập ADAST nói chung kém hơn. Các mục nói “chưa có campaign” dưới đây là trạng thái tại lượt kiểm tra thành phần ban đầu, không phải trạng thái cuối.

Đã tải kho chính thức https://github.com/emadeldeen24/ADAST vào vùng `runs/teacher_revision_cpu_20261001/adast_upstream`, pin commit `e0fb503544ddd38f71027c09e3401b900f3dabc3`. Các file nguồn upstream không bị sửa. License ở commit này là Apache-2.0; không đưa dữ liệu người tham gia hoặc bản sao repo vào gói nộp báo.

## Đã thực sự chạy

`scripts/check_adast_cpu_compatibility.py` đã chạy các model class gốc và hàm similarity penalty gốc trên PyTorch 2.5.1 CPU. Đầu vào gồm tám epoch nguồn có nhãn và tám epoch adaptation từ cache toàn bản ghi không chứa nhãn. Hai bước cập nhật tương ứng trọng số loss của round 0/1 đều cho loss và gradient hữu hạn; đặc trưng có kích thước `(8,3712)`, logits `(8,5)`.

Đây là kiểm tra tích hợp các thành phần bằng vòng cập nhật cục bộ, **không phải chạy nguyên trainer, không phải benchmark độ chính xác, và không xác nhận hội tụ trên toàn dữ liệu**. Chưa có kết quả so sánh ADAST/source-only. Output tổng hợp: `runs/teacher_revision_cpu_20261001/adast_cpu_compatibility.json`.

## Những điểm ảnh hưởng đến đối chứng ngang hàng

1. **Nhãn đích trong quá trình chạy:** [trainer](https://github.com/emadeldeen24/ADAST/blob/e0fb503544ddd38f71027c09e3401b900f3dabc3/trainer/ADAST.py) gọi đánh giá validation đích mỗi epoch, nhưng hàm được kiểm tra trả về mô hình cuối; không thấy dùng điểm validation đích để chọn checkpoint trong hàm này. Cần bỏ luồng log có nhãn đích khỏi lần chạy UDA của dự án. Không được mô tả upstream là đã chứng minh có chọn mô hình theo target validation khi code không thể hiện điều đó.

2. **Sinh pseudo-label:** [training_evaluation.py](https://github.com/emadeldeen24/ADAST/blob/e0fb503544ddd38f71027c09e3401b900f3dabc3/trainer/training_evaluation.py) lấy trung bình logits hai classifier để sinh pseudo-label. Hàm có đọc/tích lũy nhãn được loader trả về, nhưng pseudo-label được sinh từ đầu ra mô hình. Khi tích hợp phải dùng loader adaptation chỉ có tín hiệu, loại luồng nhãn thật này.

3. **Quy tắc inference khác quy tắc pseudo-label:** cùng file dùng maximum theo phần tử giữa logits hai classifier lúc đánh giá. Phải giữ hoặc công bố rõ thay đổi; không âm thầm thay bằng trung bình rồi gọi là tái lập nguyên bản.

4. **Số bước train bị giới hạn bởi loader ngắn hơn:** vòng train dùng `zip(source_loader, target_loader)`, target pseudo-loader `drop_last=True`. Với 4.989 epoch adaptation và batch 128, có 38 batch đầy đủ mỗi epoch; trong khi train nguồn fold 0 có 157.200 epoch. Do đó không thể coi một epoch ADAST là một lượt qua toàn bộ nguồn. Đối chứng source-only cần kiểm soát số bước tối ưu hóa/số mẫu nguồn đã xem. Nếu đổi sang lặp lại target loader để đi hết nguồn, phải ghi nhận đây là điều chỉnh giao thức.

5. **Lịch train gốc:** [configs.py](https://github.com/emadeldeen24/ADAST/blob/e0fb503544ddd38f71027c09e3401b900f3dabc3/config_files/configs.py) đặt hai vòng self-training, mỗi vòng 15 epoch. Trainer chỉ gọi scheduler ở round đầu. Cần lưu chính xác số update, lịch learning rate và quy tắc chọn model trong adapter mới.

6. **Preprocessing không đồng nhất với dự án:** [prepare_shhs.py](https://github.com/emadeldeen24/ADAST/blob/e0fb503544ddd38f71027c09e3401b900f3dabc3/data_preprocessing/prepare_shhs.py) chọn cửa sổ dựa trên nhãn; `downsample.py` resample riêng từng epoch. Dự án dùng resample liên tục và cần adaptation không phụ thuộc nhãn. Có thể đối chiếu backbone ADAST dưới preprocessing chung, nhưng phải gọi đúng là implementation được điều chỉnh cho giao thức chung.

7. **Chi tiết attention:** [models.py](https://github.com/emadeldeen24/ADAST/blob/e0fb503544ddd38f71027c09e3401b900f3dabc3/models/models.py) khai báo `value_conv` nhưng forward dùng trực tiếp `x` cho value. Kiểm tra CPU giữ nguyên hành vi này. Không tự sửa khi mục tiêu là giữ kiến trúc upstream.

## Checklist tích hợp được xác định sau audit ban đầu

- Viết và kiểm tra adapter dữ liệu theo subject split; source train/validation và adaptation không nhãn tách biệt.
- Tạo source-only của cùng backbone/classifier; chốt ngân sách số update và quy tắc sử dụng attention cho inference.
- Chốt trước cách giữ quy tắc logits gốc, lịch train, selection nguồn hoặc số epoch cố định; ghi rõ phần điều chỉnh.
- Có checkpoint, predictions và thống kê riêng cho hai nhánh; so sánh trên cùng cohort/epoch và báo cáo N3 cùng macro-F1.

Các bước này đã thực hiện ở mức pilot một fold bằng `scripts/run_adast_cpu_pilot.py` và kiểm chứng `scripts/verify_adast_cpu_pilot.py`. Source-only giữ dual-head/similarity regularization, dùng source attention; ADAST dùng target attention khi suy luận trên SHHS. Chọn checkpoint cuối theo lịch upstream cố định, không theo nhãn đích. Chưa mở rộng nhiều fold/seed hoặc ngân sách adaptation lớn hơn, chưa benchmark công bằng với ensemble mười fold. Không đưa code upstream hay artifact người tham gia vào ZIP nộp.

CLI upstream hiện còn phụ thuộc `torchvision` (import trong dataloader) và phần xuất Excel cần `openpyxl`; môi trường hiện tại chưa có chúng. `umap` phục vụ đồ thị tùy chọn cũng chưa có. Chưa cài thêm hoặc thay môi trường PyTorch đang dùng. Việc thiếu các gói này không ngăn kiểm tra model/loss đã chạy, nhưng kiểm tra đó không thay thế lượt chạy trainer đầy đủ. Nhu cầu thời gian/GPU của campaign UDA chưa được benchmark; không khẳng định GPU là điều kiện toán học bắt buộc.
