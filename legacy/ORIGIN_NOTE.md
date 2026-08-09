# Nguồn của bản sao legacy

Các tệp trong `source_snapshot` được sao chép từ working tree của dự án cũ tại thời điểm thiết lập workspace, không phải checkout sạch từ một commit.

- Nhánh tại thời điểm sao chép: `minimax-promt`.
- Commit HEAD được ghi trong `SOURCE_MANIFEST.csv` để cung cấp ngữ cảnh.
- SHA-256 trong manifest mới là định danh chính xác của nội dung đã sao chép.
- `PaperZleepAnylist/prepare_sleepedf.py` đang có thay đổi chưa commit trong dự án nguồn; trạng thái này được ghi riêng trong cột `source_git_status`.
- `PipelineEEG/weight.rar` là tệp chưa theo dõi trong dự án nguồn và không được sao chép vào workspace mới.

Mọi tệp legacy mang trạng thái `legacy_unverified` và không được import trực tiếp vào thí nghiệm mới.

