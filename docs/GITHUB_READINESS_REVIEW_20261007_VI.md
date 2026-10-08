# Review mã nguồn và phạm vi đưa lên GitHub — 07-10-2026

> Các phát hiện kỹ thuật dưới đây đã được xử lý trong lượt tiếp theo. Xem
> [kết quả sửa và kiểm tra ngày 08-10-2026](GITHUB_FIXES_20261008_VI.md).
> Phần còn lại được giữ như snapshot review ban đầu, không phải trạng thái lỗi hiện tại.

## Kết luận

Mã tính toán đã có nhiều tối ưu hợp lý và các test chọn lọc đều đạt. Tuy nhiên, **chưa nên push toàn bộ worktree hiện tại**: khả năng dùng từ checkout sạch, ghi tiến độ trên Windows và quy trình đóng gói paper còn vấn đề cụ thể dưới đây. Không cần train lại để xử lý các vấn đề này.

Đây là review và đề xuất phạm vi, chưa phải phê duyệt phát hành. Chưa stage, commit, push, xóa tệp hoặc sửa mã nghiên cứu trong lượt review này. Các kết quả, checkpoint và bằng chứng lịch sử phải được giữ nguyên.

## Các phát hiện cần xử lý

### P1 — Test ADAST phụ thuộc tệp riêng ngoài mã nguồn

- `tests/test_adast_fullsource_completion.py:21–24` import trực tiếp từ `runs/adast_development_20261004/payload`; hai test module loss-ablation và small-confirmation cũng làm tương tự.
- Trong bản sao chỉ có mã nguồn, cấu hình và metadata cần thiết, không có các payload riêng: pytest thu thập được 357 test rồi gặp **3 lỗi import**. Trong workspace đầy đủ có 395 test được thu thập.
- Hậu quả: người clone repo không thể chạy bộ test đầy đủ bằng hướng dẫn thông thường.
- Đề xuất: cung cấp adapter/metric helper như mã nguồn có thể import; dùng fixture tổng hợp cho unit test; đánh dấu riêng các integration test cần dữ liệu có quyền truy cập. Không chữa bằng cách đưa dữ liệu EEG hoặc toàn bộ `runs/` lên GitHub, cũng không âm thầm bỏ các test lỗi.
- ADAST upstream cần hướng dẫn lấy đúng phiên bản và giữ giấy phép đi kèm; không tự gán giấy phép mới cho mã của bên khác.

### P1 — Bộ build chính chưa tạo đúng toàn bộ paper hiện tại

- `scripts/rebuild_bspc_delivery.ps1:33–42` build paper Anh, phụ lục và các báo cáo Việt cũ; thiếu `Reports/paper_vi_translation/main.tex` và PDF bản dịch BSPC hiện hành.
- `Reports/REPORT_MANIFEST.sha256:93–104` còn tham chiếu bốn tệp trong `tmp/` chưa được Git theo dõi. Checkout sạch sẽ không có các tệp này.
- Vòng lặp tại `scripts/rebuild_bspc_delivery.ps1:65–82` tính lại mọi hash trong manifest đóng gói mà không so với giá trị cũ. Nó không sửa manifest thí nghiệm gốc, nhưng có thể thay các hash tham chiếu bằng chứng lịch sử trong manifest đóng gói mà không báo thay đổi.
- Đề xuất: thống nhất một entrypoint build đủ Anh/Việt/phụ lục; đưa phần kiểm chứng công khai cần thiết vào thư mục ổn định; tách manifest phát hành khỏi hồ sơ vận hành riêng. Kiểm tra các hash bằng chứng cũ trước khi tái tạo manifest phát hành. Tạo ZIP qua tệp tạm rồi thay thế sau khi thành công.
- Đây là lỗi khả năng dựng lại bằng entrypoint chung; không có nghĩa các PDF vừa bàn giao sai số liệu.

### P1 — Hash nguồn paper có thể thay đổi khi đi qua Git

- `.gitattributes:1–13` chưa quy định EOL cho `.tex` và một số nguồn paper; máy hiện tại bật `core.autocrlf=true`.
- `Reports/paper_en/main.tex` có working-tree EOL hỗn hợp. `git hash-object` có và không áp dụng bộ lọc Git cho ra hai hash khác nhau, chứng minh bytes bị chuẩn hóa khi đưa vào Git.
- Hậu quả: manifest hash theo bytes local có thể không khớp bản checkout, dù nội dung hiển thị không đổi.
- Đề xuất: chọn chính sách bytes/EOL rõ ràng cho tài liệu phát hành, kiểm tra round-trip qua Git và dựng lại manifest công khai. Không chạy chuẩn hóa toàn repo lên code/config đã gắn với bằng chứng thí nghiệm.

### P2 — Ghi progress chung vẫn có thể làm dừng job khi Windows khóa tệp

- `src/sleeptcn/revision_weighted_campaign.py:13–20` dùng một tên tệp tạm cố định và gọi `replace()` một lần; lỗi khóa tệp được truyền ra ngoài `CampaignBudget.publish()`.
- Bản resume riêng đã xử lý tình huống cho lượt bị dừng, nhưng không thay thế helper chung của những runner khác.
- Đề xuất: writer tiến độ phiên bản mới có tên tạm riêng, retry hữu hạn và thông báo lỗi rõ ràng; tách lỗi cập nhật trạng thái khỏi ghi kết quả khoa học. Test lỗi khóa tệp bằng mock. Không được bỏ qua lỗi ghi checkpoint/kết quả thật.

### P2 — Cache hash không tránh được việc đọc lại tệp

- `scripts/run_adast_fullsource_local_evaluation.py:60` dùng `ledger.setdefault(key, sha(path))`.
- Python tính `sha(path)` trước khi gọi `setdefault`, kể cả khi key đã tồn tại. Một số lời gọi ngay bên dưới còn tính hash ở đối số trước đó.
- Hậu quả: I/O dư thừa trên các tệp lớn; không làm sai phép so sánh hash.
- Đề xuất: kiểm tra key trước khi tính hash trong phiên bản runner kế tiếp; vẫn đối chiếu expected hash mỗi lần. Chỉ cache trong giai đoạn input bất biến, không dùng cache để bỏ kiểm tra tệp có thể bị sửa. Chưa đo mức tăng tốc, không gán con số tiết kiệm giả định.

### P2 — Hướng dẫn và phân chia môi trường chưa theo kịp mã mới

- `README.md:7` còn trạng thái 25-08 và đoạn SHHS vẫn nói chưa thích nghi miền; `tests/README.md:4` còn số 139 test.
- `pyproject.toml` pin PyTorch 2.5.1, còn giao thức ADAST Colab yêu cầu 2.11.0+cu130. Đây là hai môi trường lịch sử khác nhau, cần mô tả riêng, không nâng đồng loạt để làm chúng giống nhau.
- Runner replay hiện hành phụ thuộc pointer/proof và thư mục chiến dịch cụ thể. Cần phân biệt rõ “phát lại chiến dịch đã làm” với “chạy một thí nghiệm mới từ dữ liệu riêng”. Không tháo kiểm tra nguồn gốc chỉ để chạy được trên máy khác.
- Chưa thấy LICENSE cấp repo hoặc CI. Cần người sở hữu chọn giấy phép nếu muốn cho phép tái sử dụng; CI nên chạy nhóm test không đòi EEG/checkpoint riêng.

## Mức tối ưu đã thấy

- CPU validation dùng inference mode và tái sử dụng feature encoder cho hai attention trên cùng batch.
- Inference nạp các model trước vòng lặp bản ghi; không nạp lại checkpoint cho từng bản ghi.
- GPU full-source dùng memory map và lấy dữ liệu theo batch thay vì sao chép toàn bộ nguồn cho từng fold.
- Bootstrap dùng tính toán theo khối và cùng lượt lấy mẫu người cho bốn hệ thống.
- Khởi tạo model đã đặt seed trước khi xây model; test hồi quy cho thay đổi này đạt.

Ưu tiên lúc này là tính ổn định, khả năng dựng lại và đóng gói. Chưa có lý do từ review này để thay kiến trúc, giảm kiểm chứng hoặc train lại. Việc tách module dài và hợp nhất script có thể làm sau; phải giữ bản lịch sử dùng cho chứng minh kết quả.

## Kiểm tra thực hiện

| Kiểm tra | Kết quả |
|---|---|
| Parse cú pháp Python trong src/scripts/tests/demo | 261 tệp, không lỗi cú pháp |
| Thu thập test trên workspace hiện tại | 395 test |
| Test chọn lọc về preprocessing, seed, calibration, demo, inference, verification, resume, hashing, statistics và campaign | 150 passed, 19,09 giây |
| Thu thập test trong bản sao chỉ chứa nguồn, không payload riêng | 357 test được thu thập, 3 lỗi import |
| Quét mẫu token/private-key trong văn bản ứng viên ngoài tmp | Không thấy mẫu khớp; không phải chứng nhận bảo mật toàn bộ lịch sử Git |
| Notebook ipynb hiện có | 3 notebook, không output hoặc execution count |

Không chạy toàn bộ 395 test; không train CPU/GPU; không benchmark lại mô hình. Phạm vi kiểm tra là mã nguồn hiện tại, không phải toàn bộ object/lịch sử Git hoặc remote GitHub.

## Đề xuất đưa lên GitHub

| Nhóm | Phạm vi đề xuất |
|---|---|
| Mã nguồn | `src/`, `demo/`, các script train/evaluate/verify/analyze/build có liên quan; giữ dependency helper kể cả khi tên chứa CPU/pilot |
| Kiểm thử | `tests/`, sau khi unit test không còn yêu cầu payload riêng |
| Giao thức và môi trường | Config đã dùng, requirements/pyproject, hướng dẫn môi trường CPU và Colab tách biệt; thay đường dẫn riêng bằng hướng dẫn ánh xạ khi thích hợp |
| Hướng dẫn | README, docs và notebook sạch; cập nhật trạng thái, lệnh chạy, dữ liệu cần tự xin quyền và phạm vi tái lập |
| Nguồn paper | TeX/BibTeX/BST, nguồn và ảnh figure đã duyệt của Anh/Việt/phụ lục; không đưa build rác kèm theo |
| Kết quả công khai | 15 JSON/CSV tổng hợp đã phân loại trong `Reports/analysis/`; báo cáo khoa học/phản hồi thầy sau rà thông tin riêng; manifest công khai độc lập với tmp |
| Bản bàn giao hiện hành | Ba PDF: paper Anh, bản dịch Việt BSPC, phụ lục Anh; một source ZIP BSPC, tổng khoảng 1,32 MB |

Bốn bản bàn giao cụ thể:

- `Reports/output/pdf/SleepTCN_Scientific_Article_EN.pdf`
- `Reports/output/pdf/SleepTCN_BSPC_Ban_dich_Tieng_Viet.pdf`
- `Reports/output/pdf/SleepTCN_Supplement_EN.pdf`
- `Reports/output/source/SleepTCN_BSPC_Manuscript_Source.zip`

Không xóa hoặc thay thế kết quả khoa học lịch sử chỉ vì có kết quả mới. Các metadata/split đã tracked từ trước cần rà quyền công bố riêng, không coi “đã tracked” đồng nghĩa “được phép công khai”.

## Không đưa lên trong lượt này

- EEG, nhãn/ID người tham gia, dự đoán từng người, checkpoint, optimizer/RNG, cache dữ liệu.
- Payload/ZIP upload Colab, backup fold, storage pointer, process identity, log vận hành, file `*.private.json`, đường dẫn notebook/tài khoản riêng.
- Toàn bộ `tmp/`, render PNG để QA, bản giải nén ZIP, thư viện tạm, `.venv`, `.miktex`, cache Python/pytest.
- LaTeX `.aux/.log/.out/.blg` và PDF build trùng với PDF bàn giao. `.bbl` chỉ thêm có chủ đích nếu gói nộp bài yêu cầu, không gom tất cả build artifacts.
- Config nháp `configs/adast_development_followup_draft_20261004.json`.
- Script dọn ổ đĩa/đo Iris dùng một lần: `audit_cleanup_20261007.py`, `cleanup_recoverable_20261007.ps1`, `verify_cleanup_20261007.py`, `diagnose_iris_gradients.py`, `benchmark_iris_tcn.py` trong `scripts/`.

Snapshot đầu review có 640 tệp tracked, 34 tệp tracked thay đổi và 5.254 tệp untracked. Có **303 tệp sinh tự động khoảng 50,6 MiB đang được Git theo dõi**, cần gỡ khỏi index sau xác nhận nhưng **giữ nguyên trên ổ đĩa**. Không chạy `git add .`; `.gitignore` mới không tự gỡ những tệp đã tracked. Không rewrite lịch sử Git trong phạm vi này.

## Danh sách chi tiết và thứ tự sau xác nhận

Danh sách từng tệp nằm trong `tmp/github_review_20261007/`: `file_decisions.csv` là snapshot đầy đủ; các CSV con chia nguồn, kết quả, chờ rà, giữ local và đề xuất gỡ tracking. Đây là danh sách review, không phải lệnh stage tự động. Các nhãn `AFTER_FIXES`/`REVIEW` thể hiện bước cần làm trước khi đưa vào commit.

1. Sửa các điểm chặn và hướng dẫn, giữ nguyên code/hash lịch sử; tạo phiên bản mới khi sửa helper đã gắn với proof.
2. Rà thông tin riêng/giấy phép; cập nhật ignore và gỡ tracking đúng danh sách, không xóa tệp local.
3. Kiểm tra source-only checkout, chạy test phù hợp, build đủ ba PDF và source ZIP, kiểm tra manifest sau Git round-trip và QA PDF.
4. Kiểm tra staged diff và danh sách cuối; chia commit nguồn/test và paper/kết quả để dễ review.
5. Chỉ push sau xác nhận của người dùng về phạm vi và đích push. Hiện branch local là `main`; không đổi sharing, không nộp bài, không tự chọn giấy phép.
