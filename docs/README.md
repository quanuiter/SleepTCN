# Hệ thống tài liệu SleepTCN

Thư mục này tập hợp các protocol, kết quả và giới hạn diễn giải của nghiên cứu. Số liệu có cấu trúc được
lưu trong `runs/v2/`, còn báo cáo tổng hợp và bản thảo nằm trong `Reports/`. Notebook chỉ giữ vai trò
minh họa và kiểm tra môi trường.

## Tài liệu nền tảng

1. [STATUS_V2.md](STATUS_V2.md) — trạng thái, kết quả chính và giới hạn kết luận.
2. [REPRODUCIBILITY_PACKAGE.md](REPRODUCIBILITY_PACKAGE.md) — hợp đồng môi trường, dữ liệu và audit.
3. [REPORT_GATE1_8_AUDIT.md](../Reports/REPORT_GATE1_8_AUDIT.md) — kiểm định nội dung báo cáo.
4. [CLAIM_EVIDENCE_MATRIX.md](../runs/v2/publication/gate8/CLAIM_EVIDENCE_MATRIX.md) — biên phát biểu khoa học.

## Phân loại tài liệu

| Nhu cầu | Tài liệu chính |
|---|---|
| Kiến trúc phần mềm | [SOURCE_ARCHITECTURE.md](SOURCE_ARCHITECTURE.md) |
| Chạy GPU/Docker | [DOCKER_GPU_RUNBOOK.md](DOCKER_GPU_RUNBOOK.md) |
| Tuning ResNet-1D v3 | [RESNET_TUNING_V3.md](RESNET_TUNING_V3.md) |
| Tái lập SHHS | [SHHS_V1_RUNBOOK.md](SHHS_V1_RUNBOOK.md) |
| Tiền xử lý | [PREPROCESSING_SPEC.md](PREPROCESSING_SPEC.md) |
| Chia fold | [SPLIT_SPEC.md](SPLIT_SPEC.md) |
| Thí nghiệm E0–E6 | [EXPERIMENT_PROTOCOL_V2.md](EXPERIMENT_PROTOCOL_V2.md) |
| Gate 8 ablation | [GATE8_CONTEXT_ABLATION_PROTOCOL.md](GATE8_CONTEXT_ABLATION_PROTOCOL.md) |
| Kết quả chi tiết | Các file `*_RESULTS.md`; số liệu máy đọc tương ứng nằm trong `runs/v2/` |

## Nguyên tắc sử dụng

- `STATUS_V2.md` là tài liệu chính thức về trạng thái hiện hành.
- Protocol và runbook mô tả thiết kế, điều kiện tái lập và giới hạn của từng phân tích.
- `runs/v2/publication/gate8/` là gói kết quả công bố cuối; các bảng và hình Gate 7 đã được hợp nhất vào
  gói này.
- `Reports/` chứa báo cáo tổng hợp và các biên bản kiểm toán; mọi diễn giải phải phù hợp với ma trận bằng
  chứng trong gói Gate 8.
- Các tài liệu lịch sử vẫn có thể được truy nguyên từ lịch sử Git khi cần đối chiếu.
