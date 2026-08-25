# Kiến trúc phần mềm và phạm vi refactor

Kiến trúc hiện tại tách các hợp đồng dữ liệu, đánh giá và provenance khỏi phần điều phối huấn luyện.
Mục tiêu là giảm logic trùng lặp, duy trì đường dẫn artifact và bảo toàn kết quả đã khóa trong các
chiến dịch Sleep-EDF và Gate 8.

## Phân lớp phần mềm

```text
src/sleeptcn/
  io/             serialization, hashing, manifest và kiểm toán dữ liệu
  evaluation/     prediction table, metrics, bảng và validator publication
  workflows/      layout, checkpoint, provenance và protocol Gate 8/SHHS
  dataset.py      đọc bản ghi Sleep-EDF và thông tin epoch
  preprocessing.py tiền xử lý tín hiệu và tạo artifact NPZ
  models.py       các kiến trúc CNN, ResNet-1D, BiLSTM và TCN
  engine.py       vòng lặp huấn luyện, validation và checkpoint
  experiment.py   điều phối các cấu hình E0–E6
  gate8.py        điều phối ablation C/P/N
```

Các module `io` cung cấp primitive dùng chung cho SHA-256, JSON nguyên tử, NPZ xác định, đường dẫn
portable và manifest. `evaluation` quản lý schema của prediction, lưu metrics theo role và kiểm tra
gói publication. `workflows` giữ các hợp đồng không phụ thuộc vào một lệnh gọi cụ thể, gồm layout của
run, marker checkpoint, cách dựng sequence model, protocol Gate 8 và protocol SHHS.

## Nguyên tắc bảo toàn kết quả

- Split, seed, metric, schema artifact và đường dẫn output được giữ nguyên.
- Các checkpoint chỉ được nạp sau khi marker stage và SHA-256 được xác minh.
- Vector thay thế trong Gate 8 chỉ được tính từ epoch train hợp lệ của từng fold.
- Serialization mới dùng format `npz_zip_stored_v1`; thay đổi format phải đi kèm phiên bản và manifest
  mới.
- Các helper thuần NumPy và filesystem được kiểm thử độc lập với phần huấn luyện PyTorch.

## Provenance và tái lập

`workflows.provenance` định nghĩa tập file tham gia `runner_code_sha256` cho experiment runner và Gate 8.
Các validator publication, manifest và dữ liệu dẫn xuất ghi nhận hash của nguồn đầu vào và output. Nhờ
đó, việc thay đổi một module tham gia kết quả sẽ làm thay đổi provenance tương ứng.

Các script trong `scripts/` là lớp giao tiếp dòng lệnh. Logic kiểm toán và kiểm tra schema nằm trong
package để có thể gọi trực tiếp từ test hoặc từ một quy trình tự động khác.

## Phạm vi còn lại

`experiment.py`, `gate8.py` và `shhs_zero_shot.py` vẫn chứa phần điều phối huấn luyện và suy luận do đây
là ranh giới gắn với checkpoint, thiết bị và dữ liệu. Việc tiếp tục tách các module này chỉ nên thực hiện
sau khi có characterization test chứng minh output fixture giữ nguyên.

Mọi mở rộng seed, adaptation, fine-tuning hoặc cohort mới phải được đăng ký trong protocol riêng và
không được sửa ngược các artifact đã khóa.
