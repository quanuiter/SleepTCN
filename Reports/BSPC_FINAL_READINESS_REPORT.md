# Bàn giao bản sửa chuẩn bị nộp BSPC

Ngày kiểm tra cuối: **06-09-2026**. Baseline: `1710c49fe529f7d498598ad25ff108163f3e6f06`.
Các thay đổi hiện nằm trong working tree; chưa tạo commit/release hoặc nộp bài.

**Kết luận:** đã hoàn tất phần chỉnh nội dung, đồng bộ ENG/VI, ghi nhận các xác nhận tác giả, build và
kiểm tra tài liệu trong phạm vi được giao. Bài được giữ đúng định vị nghiên cứu thực nghiệm về lựa chọn
pipeline, đánh đổi vận hành và cấu trúc lỗi chuyển miền. Trước khi bấm Submit, tác giả vẫn cần push/tag
đúng revision GitHub, giữ căn cứ ethics/NSRR trong hồ sơ và đối chiếu Guide for Authors/upload hiện hành.
Không bảo đảm chấp nhận hoặc gọi kiểm tra tài liệu là chứng minh khoa học “100%”.

## 1. Tài liệu bàn giao

| Tài liệu | Đường dẫn tương đối với `Reports/` | Trang |
|---|---|---:|
| Bài tiếng Anh | `output/pdf/SleepTCN_Scientific_Article_EN.pdf` | 13 |
| Supplement tiếng Anh | `output/pdf/SleepTCN_Supplement_EN.pdf` | 6 |
| Bài tiếng Việt | `output/pdf/SleepTCN_Scientific_Article_VI.pdf` | 13 |
| Báo cáo nghiên cứu đầy đủ tiếng Việt | `output/pdf/SleepTCN_Gate1_8_SHHS_Report.pdf` | 41 |

Tổng cộng 73 trang đã được render và kiểm tra trực quan. Những trang thay đổi ở lượt sửa cuối
được render/kiểm tra lại. Bản sao bàn giao khớp SHA-256 với PDF build tương ứng. Không phát hiện
chữ/bảng bị cắt, chồng lấn hoặc tham chiếu hiển thị thiếu trong lượt kiểm tra này.

Nguồn tiếng Anh được đóng gói tại `output/source/SleepTCN_BSPC_Manuscript_Source.zip`.
ZIP chứa đúng bảy file: `main.tex`, `supplement.tex`, `references.bib`, `highlights.txt`, `README.md`
và hai PDF hình tiếng Anh dưới `figures/`. Đã build từ thư mục độc lập và kiểm tra 7/7 tên file/hash
trong ZIP. Text trích từ hai PDF của fresh build khớp text của bản bàn giao.

Gói này không phải archive tái lập thí nghiệm: không chứa raw data, participant-level predictions,
subject IDs hoặc tài liệu kiểm toán nội bộ. Giữ hai PDF ENG cùng thư mục để link sang supplement
hoạt động. Nếu hệ thống nộp chỉ build `main.tex`, cần cung cấp `supplement.aux` mới theo hướng dẫn
trong README hoặc dùng quy trình hỗ trợ của tạp chí; chưa kiểm tra bộ build phía publisher.

## 2. Những sửa đổi khoa học và biên tập chính

- E1 được mô tả là thay cấu hình mô hình chuỗi **và training recipe**; E2 là thay gói C/P/N 75 chiều
  bằng encoder ResNet-1D 128 chiều chỉ dùng epoch hiện tại, cùng khác biệt huấn luyện/chọn encoder.
- Đã bổ sung bảng cấu trúc, training và preprocessing; làm rõ context, offline inference, cửa sổ
  lấy từ nhãn thật và whole-record transductive normalization của E6.
- Phân biệt pooled macro-F1 và subject-mean macro-F1, CI và Wilcoxon, family Holm chính/mở rộng.
  E3−E2 trên EDF có CI gộp dương nhưng không có Wilcoxon–Holm significance; không diễn giải sai
  thành CI chứa 0 hay suy nhóm người tạo ra lợi ích gộp.
- Giữ lợi ích vận hành 3,76 lần forward speed-up và thời gian training/validation đã quan sát,
  đồng thời báo cáo đúng chi phí tham số/bộ nhớ và khác biệt training recipe.
- Làm nổi bật lợi ích tổng thể E3 trên SHHS cùng lỗi bỏ sót N3 tái diễn ở E0/E3. E3 precision N3
  cao hơn E0, nhưng recall vẫn thấp. E6 thất bại trong pipeline đã thử, không loại trừ biên độ
  hoặc mọi cách chuẩn hóa/thích nghi không nhãn.
- Oracle dùng nhãn thật, không cộng tuyến tính và không phải hiệu năng một remedy đã đạt.
  Chênh lệch EDF OOF–SHHS ensemble được gọi đúng là mô tả, không cô lập domain shift.
- Bổ sung bằng chứng seed123 đã có vào supplement, kể cả đối chiếu SHHS mở rộng và vùng nhãn;
  không nâng thành cohort độc lập hoặc tự tạo kết quả mới.
- Giữ context ablation/silhouette ở vai trò hỗ trợ; related work có SleepInceptionNet, ADAST và
  bối cảnh tiêu chí chấm N3 của Davidson et al. Không quy cơ chế sinh lý từ các trích dẫn đó.
- Abstract ENG còn khoảng **212 token tách theo khoảng trắng** từ PDF, gồm các nhãn phần;
  keywords còn sáu mục. Bốn highlights dài 73/71/67/68 ký tự, không tính dấu đầu dòng.
- Ngô Nhật Quân là first/corresponding author, email đã xác nhận; Phạm Thái Sơn đứng thứ hai.
  Cả hai tác giả đã xác nhận đọc và duyệt bản cuối; CRediT vai trò đã được ghi, không ghi tỷ lệ phần trăm.

Chi tiết từng claim, nguồn và giới hạn: `BSPC_CLAIM_EVIDENCE_LEDGER.md`.
Lịch sử thay đổi: `BSPC_REVISION_LOG.md`. Các báo cáo hiện hành và `note.md` đã được rà lại;
biên bản review cũ được giữ làm lịch sử, có ghi rõ không thay thế kết quả bàn giao này.

## 3. Kiểm tra đã thực hiện

| Kiểm tra | Kết quả và phạm vi |
|---|---|
| Build bốn tài liệu | Thành công; log cuối không có LaTeX undefined references/citations, overfull/underfull box hoặc BibTeX warning |
| Kiểm PDF | 73 trang, kiểm dấu tiếng Việt, bảng/hình, tên tác giả, tham chiếu và trang cuối; đã kiểm lại trang sửa cuối |
| Gói nguồn độc lập | Fresh build ENG 13 trang và supplement 6 trang; không cần asset ngoài bundle; text khớp bản bàn giao |
| Tests | `python -m pytest -q -p no:cacheprovider`: 150 passed, 19 warnings |
| Dữ liệu khoa học khóa | Không sửa `configs/`, `src/`, `data/`, `tests/`, `runs/` hoặc các audit JSON số liệu |
| Primary SHHS | 540/540 ensemble hashes và confusion matrices từng recording/tổng gộp khớp nguồn đã lưu |
| Bằng chứng vùng seed123 | EDF campaign/20 E0/E3 prediction hashes và SHHS extension/360 E0/E3 ensemble hashes khớp nguồn summary |
| Git whitespace | `git diff --check` đạt; cảnh báo chuyển LF/CRLF của Git không phải thay đổi số liệu |

Các test warnings liên quan API deprecated, helper class không thuộc test collection và mẫu nhỏ trong
test Wilcoxon tổng hợp. Chúng không phải kết quả bất định mới của paper. MiKTeX còn thông báo chưa
kiểm tra cập nhật; không cập nhật hệ thống chỉ để tắt thông báo đó.

`texcount` không chạy được vì máy thiếu Perl. Vì vậy số 212 ở trên là phép đếm minh bạch từ PDF,
không tuyên bố đã kiểm word count theo định nghĩa riêng của BSPC. Không lấy số trang hoặc số token
gồm bảng/header để khẳng định đáp ứng giới hạn bài báo.

Phép đếm khoảng trắng từ PDF trước bibliography cho 5.954 token, **bao gồm** title/abstract,
bảng, số và header. Đây chỉ là mốc biên tập để tác giả kiểm lại theo cách tính của journal;
không đồng nhất với 5.954 từ main text hoặc khẳng định bài dưới 5.000 từ.

## 4. Hash: đã làm gì và chưa làm gì

Không sửa hash trong run manifest để ép khớp file hiện hành. Snapshot lịch sử
`configs/shhs_zero_shot_v1.json` khớp protocol hash gốc; `configs/shhs_v1_protocol.json` là hồ sơ
mở rộng sau chạy và được giữ riêng. Lịch sử N3 audit có cả thay metadata lẫn sửa tỷ lệ dẫn xuất E3;
không gọi thay đổi đó là metadata-only. Xem `SHHS_PROTOCOL_PROVENANCE.md` và `ARTIFACT_INDEX.md`.

Publication manifests ghi hash mới của **tài liệu đã sửa và file bàn giao**, không viết lại lịch sử
thí nghiệm. `REPORT_MANIFEST.sha256` dùng đường dẫn từ `Reports/`;
`paper_en/SUBMISSION_MANIFEST.sha256` dùng đường dẫn từ `Reports/paper_en/`.
Các manifest này là inventory kiểm tra byte, không tự chứng minh gói tái lập đầy đủ hoặc quyền chia sẻ.

Chưa tái chạy raw EDF/SHHS → preprocessing → training → predictions; chưa kiểm toàn bộ 5.400
fold-prediction files của primary SHHS hoặc toàn checkpoint chain. Không chạy lại bootstrap,
calibration hay adaptation. Không được đổi các giới hạn này thành “fully reproducible”.

## 5. Những mục cần chốt trước Submit

Làm theo `BSPC_AUTHOR_CONFIRMATIONS_VI.md`; các xác nhận tác giả đã được đưa vào manuscript. Các bước dưới đây
là đối chiếu hồ sơ và journal-upload, không phải yêu cầu chạy thêm thí nghiệm.

1. Đối chiếu CRediT đã ghi với đóng góp thực tế; việc đọc và duyệt bản cuối đã được xác nhận.
2. Giữ căn cứ cho xác nhận ethics và quyền truy cập SHHS/NSRR trong hồ sơ nội bộ; kiểm tra acknowledgement
   bắt buộc của NSRR trước khi upload. Manuscript không chứa dữ liệu hạn chế hoặc subject IDs.
3. Đối chiếu khoản hỗ trợ sinh viên khoảng 6.000.000 VND và declaration không có xung đột lợi ích với hồ sơ của hai tác giả.
4. Push các thay đổi hiện tại lên GitHub và ghi commit/tag đúng với bộ PDF nộp; không đưa raw SHHS, subject IDs hoặc predictions cá thể vào kho.
5. Rà lần cuối declaration AI; hiện tác giả xác nhận chỉ dùng OpenAI Codex và cả hai đã duyệt bản cuối.
6. Mở [Guide for Authors BSPC](https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control/publish/guide-for-authors)
   trong trình duyệt của tác giả. Trang trả HTTP 403 trong môi trường kiểm tra; cần đối chiếu loại bài,
   word limit, review mode, title page, template, line numbers, highlights/cover letter và supplement.
   Chưa có cơ sở coi 5.000 từ hoặc một review mode cụ thể là yêu cầu đã được xác minh.
7. Sau khi chốt revision repository và journal upload, giữ lại PDF/ZIP cùng publication hashes tương ứng.

## 6. Có cần thêm thí nghiệm không?

Không cần thí nghiệm mới để sửa những lỗi diễn giải đã xác định. Bản sửa không đặt điều kiện rằng bài
phải trở thành nghiên cứu adaptation hoặc SOTA mới phù hợp định vị. Tuy nhiên biên tập viên/phản biện
vẫn có thể yêu cầu kiểm chứng thêm; chỉnh câu chữ không xóa giới hạn một hướng chuyển cohort/hai seed.

Nếu tác giả muốn tăng mức kiểm chứng, các lựa chọn chưa thực hiện gồm demographics đúng mẫu,
phân bố lỗi theo subject, kiểm tra ảnh hưởng ensemble, baseline cùng protocol hoặc cohort độc lập.
Nếu muốn claim khắc phục N3, cần adaptation/calibration study riêng với tập chọn biện pháp và test
tách biệt, báo cáo recall cùng precision/false-positive N3. Đây là các việc cần phê duyệt kế hoạch
và nguồn lực, không tự chạy trong lượt sửa tài liệu.
