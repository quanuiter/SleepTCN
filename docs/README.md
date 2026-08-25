# Cổng tài liệu hiện hành

Đây là điểm vào để đọc repository sau khi Gate 1–8 đã đóng. Tài liệu trong
`notebooks/docs/` vẫn được giữ để tái lập và truy nguyên, nhưng không phải file
nào trong đó cũng là trạng thái hiện tại.

## Đọc trước

1. [STATUS_V2.md](../notebooks/docs/STATUS_V2.md) — trạng thái, kết quả và giới hạn hiện hành.
2. [REPRODUCIBILITY_PACKAGE.md](../notebooks/docs/REPRODUCIBILITY_PACKAGE.md) — hợp đồng artifact, lock và audit.
3. [REPORT_GATE1_8_AUDIT.md](../Reports/REPORT_GATE1_8_AUDIT.md) — audit narrative của gói kết quả.
4. [NEXT_STEPS.md](../notebooks/docs/NEXT_STEPS.md) — phụ lục quyết định cho nghiên cứu mới.

## Theo nhiệm vụ

| Nhu cầu | Tài liệu chính |
|---|---|
| Chạy GPU/Docker | [DOCKER_GPU_RUNBOOK.md](../notebooks/docs/DOCKER_GPU_RUNBOOK.md) |
| Chạy SHHS external validation | [SHHS_V1_RUNBOOK.md](../notebooks/docs/SHHS_V1_RUNBOOK.md) |
| Protocol preprocessing | [PREPROCESSING_SPEC.md](../notebooks/docs/PREPROCESSING_SPEC.md) |
| Protocol split | [SPLIT_SPEC.md](../notebooks/docs/SPLIT_SPEC.md) |
| Protocol experiments | [EXPERIMENT_PROTOCOL_V2.md](../notebooks/docs/EXPERIMENT_PROTOCOL_V2.md) |
| Gate 8 ablation | [GATE8_CONTEXT_ABLATION_PROTOCOL.md](../notebooks/docs/GATE8_CONTEXT_ABLATION_PROTOCOL.md) |
| Kết quả seed/SHHS | Các report `*_RESULTS.md` trong `notebooks/docs/` |
| Bài báo và bảng/hình | `Reports/` và `runs/v2/publication/` |

## Quy tắc nguồn sự thật

- `STATUS_V2.md` là nguồn duy nhất cho trạng thái hiện hành.
- `NEXT_STEPS.md` không phải backlog đang chờ.
- `Reports/` và các report kết quả là evidence; không dùng chúng để thay đổi protocol đã khóa.
- Các runbook có nhãn **legacy** chỉ được dùng để truy nguyên hoặc tái lập lịch sử.
- Các runbook đã đóng được đặt dưới `notebooks/docs/archive/`; không dùng chúng làm lệnh vận hành mới.
- Ma trận đầy đủ nằm tại [DOCUMENT_MATRIX.md](DOCUMENT_MATRIX.md).
