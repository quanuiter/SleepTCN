# Ma trận tài liệu và kế hoạch hợp nhất

Tài liệu này là inventory cho đợt refactor documentation. Nó không thay đổi
protocol, artifact hay kết quả khoa học. Các runbook lịch sử đã đóng được chuyển
vào `notebooks/docs/archive/`; đường dẫn hiện hành và nguồn sự thật vẫn được giữ
rõ ràng để không làm hỏng liên kết trong report và runbook.

## Quy ước trạng thái

- **Giữ**: tài liệu vẫn là nguồn cần đọc sau khi dọn cấu trúc.
- **Gộp**: nội dung sẽ được đưa vào một tài liệu current/protocol khác.
- **Lưu trữ**: giữ nguyên để truy nguyên lịch sử, nhưng không được trình bày như trạng thái hiện tại.
- **Sinh tự động**: không chỉnh tay; chỉ sửa generator hoặc input contract.

## Nguồn sự thật sau khi hợp nhất

| Nội dung | Nguồn hiện hành | Đích sau refactor |
|---|---|---|
| Trạng thái và phạm vi đã đóng | `notebooks/docs/STATUS_V2.md` | Giữ làm current status duy nhất |
| Quyết định có mở nghiên cứu mới hay không | `notebooks/docs/NEXT_STEPS.md` | Phụ lục quyết định, không phải current status |
| Hợp đồng reproducibility | `notebooks/docs/REPRODUCIBILITY_PACKAGE.md` | Giữ, sau đó chuyển vào `docs/reproducibility/` |
| Protocol preprocessing/split/experiment | Các file `*_SPEC.md`, `EXPERIMENT_PROTOCOL_V2.md` | Gộp theo `docs/protocols/` |
| Lệnh vận hành Docker/GPU/SHHS | Các file `*_RUNBOOK.md` | Gộp thành runbook vận hành và runbook external validation |
| Kết quả và audit | `Reports/`, `GATE*_FINAL_RESULTS.md`, `SHHS_*_RESULTS.md` | Giữ report chính; archive bản mô tả trùng |

## Inventory

| File | Vai trò | Hành động |
|---|---|---|
| `README.md` | Entry point | Giữ, rút gọn link và nguồn sự thật |
| `docs/README.md` | Cổng tài liệu current/legacy | Giữ làm navigation chính |
| `notebooks/README.md` | Giải thích thư mục notebook | Giữ, cập nhật link current status |
| `notebooks/docs/STATUS_V2.md` | Current status và giới hạn | Giữ làm nguồn duy nhất |
| `notebooks/docs/NEXT_STEPS.md` | Điều kiện mở nghiên cứu mới | Giữ như decision appendix |
| `notebooks/docs/REPRODUCIBILITY_PACKAGE.md` | Reproducibility contract | Giữ, chuyển nhóm sau |
| `notebooks/docs/PREPROCESSING_SPEC.md` | Protocol preprocessing | Gộp vào `docs/protocols/preprocessing.md` |
| `notebooks/docs/PREPROCESSING_ABLATION_SPEC.md` | Protocol ablation | Gộp vào protocol ablation |
| `notebooks/docs/SPLIT_SPEC.md` | Protocol split | Gộp vào `docs/protocols/splits.md` |
| `notebooks/docs/EXPERIMENT_PROTOCOL_V2.md` | Protocol experiment | Gộp vào `docs/protocols/experiments.md` |
| `notebooks/docs/archive/TEST_GATE_RUNBOOK.md` | Test-gate operation lịch sử | Lưu trữ; không dùng làm current runbook |
| `notebooks/docs/DOCKER_GPU_RUNBOOK.md` | GPU operation | Giữ làm runbook GPU chính |
| `notebooks/docs/archive/GATE8_DOCKER_GPU_RUNBOOK.md` | Gate 8-specific GPU operation lịch sử | Lưu trữ; dùng `DOCKER_GPU_RUNBOOK.md` cho vận hành GPU hiện hành |
| `notebooks/docs/SHHS_V1_RUNBOOK.md` | SHHS preprocessing/zero-shot operation | Gộp vào runbook external validation |
| `notebooks/docs/GATE6_PROTOCOL_AND_RUNBOOK.md` | Gate 6 protocol + operation | Tách protocol, archive phần operation trùng |
| `notebooks/docs/archive/TRAINING_RUNNER.md` | Training CLI notes lịch sử | Lưu trữ; lệnh hiện hành nằm trong runbook GPU |
| `notebooks/docs/VALIDATION_AUDIT_10FOLD.md` | Validation evidence contract | Gộp vào reproducibility/validation contract |
| `notebooks/docs/GATE8_CONTEXT_ABLATION_PROTOCOL.md` | Gate 8 protocol | Giữ trong protocol ablation |
| `notebooks/docs/GATE5_STATISTICAL_RESULTS.md` | Gate 5 result narrative | Giữ nếu paper dùng; không phải status source |
| `notebooks/docs/GATE6_FINAL_RESULTS.md` | Gate 6 result narrative | Giữ nếu paper dùng; archive nếu report chính đủ |
| `notebooks/docs/GATE7_FINAL_RESULTS.md` | Gate 7 result narrative | Giữ evidence index hoặc archive bản trùng |
| `notebooks/docs/GATE8_FINAL_RESULTS.md` | Gate 8 result narrative | Giữ evidence index hoặc archive bản trùng |
| `notebooks/docs/MULTISEED_SENSITIVITY_RESULTS.md` | Seed 42/123 results | Giữ làm evidence |
| `notebooks/docs/SHHS_ZERO_SHOT_RESULTS.md` | SHHS zero-shot results | Giữ làm evidence |
| `notebooks/docs/SHHS_COMPONENT_EXTENSION_RESULTS.md` | SHHS component extension | Giữ làm evidence, ghi rõ hậu nghiệm |
| `notebooks/docs/SHHS_E3_E2_PAIRED_RESULTS.md` | SHHS E3-E2 result | Giữ làm evidence, ghi rõ hậu nghiệm |
| `Reports/REPORT_GATE1_8_AUDIT.md` | Audit narrative | Giữ làm audit chính |
| `Reports/BUILD.md` | Build/report operation | Giữ, rút gọn link |
| `Reports/SHHS_SEED123_E4_EXTENSION.md` | Extension report | Giữ làm evidence |
| `Reports/paper/BUILD.md` | Paper build operation | Giữ trong nhóm publication |
| `tests/README.md` | Test instructions | Giữ, bổ sung test groups |
| `runs/v2/analysis/fold_00_seed42_validation_summary.md` | Generated result summary | Sinh tự động; không chỉnh tay |

## Quy trình di chuyển

1. Cập nhật link và đánh dấu deprecated trước khi di chuyển file.
2. Tạo các tài liệu đích, giữ nguyên nội dung normative và provenance.
3. Chạy link check toàn repository.
4. Chỉ sau khi link check pass mới chuyển file lịch sử vào `archive/`.
5. Không xóa nội dung lịch sử; chỉ chuyển file đã đánh dấu legacy vào `archive/` sau khi link check pass.

## Đợt đã hoàn tất

- `GATE8_DOCKER_GPU_RUNBOOK.md`, `TRAINING_RUNNER.md` và `TEST_GATE_RUNBOOK.md`
  đã chuyển vào `notebooks/docs/archive/` vì không còn là hướng dẫn vận hành hiện hành.
- `DOCKER_GPU_RUNBOOK.md` vẫn là runbook GPU chính; `docs/README.md` là cổng
  điều hướng duy nhất cho tài liệu current/legacy.
- Kiểm tra 36 file Markdown và toàn bộ liên kết tương đối: **0 broken links**.
