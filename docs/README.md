# Tài liệu SleepTCN

Thư mục này chứa tài liệu hiện hành. Kết quả máy đọc nằm trong `runs/v2/`, báo cáo LaTeX và PDF
nằm trong `Reports/`, còn notebook chỉ là lớp điều phối minh họa.

## Đọc trước

1. [STATUS_V2.md](STATUS_V2.md) — trạng thái, kết quả chính và giới hạn kết luận.
2. [REPRODUCIBILITY_PACKAGE.md](REPRODUCIBILITY_PACKAGE.md) — hợp đồng môi trường, dữ liệu và audit.
3. [REPORT_GATE1_8_AUDIT.md](../Reports/REPORT_GATE1_8_AUDIT.md) — kiểm định nội dung báo cáo.
4. [CLAIM_EVIDENCE_MATRIX.md](../runs/v2/publication/gate8/CLAIM_EVIDENCE_MATRIX.md) — biên phát biểu khoa học.

## Theo nhiệm vụ

| Nhu cầu | Tài liệu chính |
|---|---|
| Cấu trúc source | [SOURCE_ARCHITECTURE.md](SOURCE_ARCHITECTURE.md) |
| Chạy GPU/Docker | [DOCKER_GPU_RUNBOOK.md](DOCKER_GPU_RUNBOOK.md) |
| Tái lập SHHS | [SHHS_V1_RUNBOOK.md](SHHS_V1_RUNBOOK.md) |
| Tiền xử lý | [PREPROCESSING_SPEC.md](PREPROCESSING_SPEC.md) |
| Chia fold | [SPLIT_SPEC.md](SPLIT_SPEC.md) |
| Thí nghiệm E0–E6 | [EXPERIMENT_PROTOCOL_V2.md](EXPERIMENT_PROTOCOL_V2.md) |
| Gate 8 ablation | [GATE8_CONTEXT_ABLATION_PROTOCOL.md](GATE8_CONTEXT_ABLATION_PROTOCOL.md) |
| Kết quả chi tiết | Các file `*_RESULTS.md`; số liệu máy đọc tương ứng nằm trong `runs/v2/` |

## Nguồn sự thật

- `STATUS_V2.md` là trạng thái hiện hành duy nhất.
- Protocol và runbook là tài liệu chuẩn để tái lập, không phải backlog.
- `runs/v2/publication/gate8/` là gói công bố cuối; Gate 7 chỉ còn là bước lịch sử trong quy trình.
- `Reports/` là báo cáo và evidence; không dùng tài liệu diễn giải để sửa artifact đã khóa.
- Tài liệu cũ đã xóa vẫn có thể truy nguyên từ lịch sử Git của nhánh nghiên cứu.
