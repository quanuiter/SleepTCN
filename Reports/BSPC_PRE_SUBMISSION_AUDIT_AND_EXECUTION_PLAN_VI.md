# SleepTCN — kiểm định trước nộp BSPC và kế hoạch thực thi

Ngày kiểm định: 05-09-2026. Tác giả chính theo xác nhận của người dùng: **Ngô Nhật Quân**.

Ghi chú trạng thái 06-09-2026: tài liệu này được giữ làm audit đầu vào. Các sửa đổi và phạm vi xác minh mới (bao gồm việc đã truy cập được prediction SHHS) được ghi tại `BSPC_REVISION_LOG.md`, `SHHS_PROTOCOL_PROVENANCE.md` và `BSPC_FINAL_READINESS_REPORT.md`; không dùng trạng thái thiếu dữ liệu trong audit cũ để thay cho trạng thái hiện tại.

Bản nguồn được kiểm tra tại commit `1710c49fe529f7d498598ad25ff108163f3e6f06`. Các số dòng dưới đây thuộc bản này; khi thực thi phải tìm lại theo nội dung, không áp dụng máy móc theo số dòng.

Trạng thái tài liệu này: **báo cáo review và kế hoạch, chưa phải biên bản xác nhận sẵn sàng nộp**. Trong lượt review không chỉnh manuscript, số liệu, protocol hoặc checkpoint. Không kế hoạch nào bảo đảm được chấp nhận, hoặc thay thế việc tác giả xác nhận các thông tin chỉ tác giả biết. Mục tiêu là để model thực thi biết chính xác phải sửa gì, lấy bằng chứng ở đâu, khi nào phải dừng và cách kiểm tra kết quả.

## 1. Kết luận biên tập

**Bài có định vị phù hợp với BSPC theo hướng nghiên cứu thực nghiệm về lựa chọn pipeline xử lý EEG, đánh giá có kiểm soát trong phạm vi xác định và phân tích lỗi khi chuyển cohort. Không cần tự biến bài thành một đề xuất kiến trúc SOTA hoặc một phương pháp domain adaptation mới.** Phạm vi chính thức của tạp chí bao gồm xử lý tín hiệu y sinh, machine learning, đánh giá thực nghiệm và benchmark. Đây là căn cứ về scope, không phải bảo đảm mức độ mới hoặc quyết định phản biện. [Phạm vi BSPC](https://shop.elsevier.com/journals/biomedical-signal-processing-and-control/1746-8094).

Tuy nhiên, **chưa nên bấm Submit bản hiện tại**. Có các lỗi kiểm chứng được trong mô tả contrast, diễn giải thống kê, giới hạn suy luận và tính đầy đủ của Methods/Supplement. Những lỗi này quan trọng hơn việc giảm thêm vài trăm từ. Có thể xử lý phần lớn mà không huấn luyện lại.

Những lo ngại trong đánh giá trước về baseline, khả năng khái quát và mức kiểm chứng kết luận có cơ sở để cân nhắc. Tuy nhiên, nếu dùng việc chưa có kiến trúc mới hoặc chưa thử adaptation làm điều kiện loại trừ mặc định, thì tiêu chuẩn đó chưa khớp định vị này. Ngược lại, chỉ đổi tên bài thành “empirical study” cũng chưa đủ: cần chỉ rõ bằng chứng mới so với các nghiên cứu pipeline/preprocessing/transfer đã có, và không tuyên bố đã chứng minh nguyên nhân hoặc cách khắc phục khi mới mô tả hiện tượng.

Ba tầng đóng góp nên giữ:

1. So sánh các **cấu hình pipeline xác định rõ**, cùng subject splits, với bất định và đánh đổi vận hành được báo cáo minh bạch.
2. Đối chiếu kết quả trong Sleep-EDF với chuyển sang SHHS1, tách khác biệt tổng thể khỏi lỗi theo lớp.
3. Chẩn đoán hậu nghiệm: N3 bị dự đoán thành N2 xuất hiện ở cả E0 và E3; kết quả E6 và phân tích vùng nhãn không cung cấp một biện pháp khắc phục đã được kiểm chứng, nhưng giới hạn những gì có thể kết luận từ các lựa chọn đã thử.

### 1.1. Những nhận xét cũ không còn đúng với bản hiện tại

| Nhận xét cũ | Tình trạng đã kiểm tra ngày 05-09-2026 |
|---|---|
| ENG 19 trang, abstract khoảng 373 từ | PDF ENG hiện 12 trang; abstract trích từ PDF khoảng 223 token phân tách bằng khoảng trắng, đã dưới mục tiêu biên tập 250 từ. Không dùng lại số đếm cũ. |
| 9 keywords | Hiện có 6 keywords hiển thị; metadata PDF còn danh sách khác, cần đồng bộ. |
| Chưa có SleepInceptionNet và ADAST | Cả hai đã được bổ sung; việc cần làm là đối chiếu đóng góp, không thêm citation trùng. |
| E2 vẫn được gọi là encoder-only ở mọi nơi | Phần lớn đã sửa thành gói feature/context, nhưng E1 vẫn bị gọi là architecture-only dù training recipe khác. Cũng cần khai báo khác biệt huấn luyện encoder ở E2. |
| Không tìm được protocol khớp hash | Snapshot `configs/shhs_zero_shot_v1.json` hiện khớp giá trị hash được lưu trong hồ sơ. Không đồng nghĩa đã tái chạy prediction hoặc chứng minh thời điểm khóa chỉ bằng hash. |
| note.md còn toàn bộ ngôn ngữ “breakthrough/mechanism” | Nhiều chỗ đã hạ mức khẳng định. Vẫn phải sửa những phát biểu còn sai, không tái báo lỗi đã được xử lý. |

Phần trước References của ENG có khoảng 6.055 token phân tách bằng khoảng trắng sau trích PDF, **bao gồm** front matter, bảng và các thành phần bố cục. Đây không phải word count thuần main text theo quy định tạp chí. Chưa có cơ sở dùng số này để tuyên bố vi phạm giới hạn 5.000 từ.

### 1.2. Mức độ xác minh

- Đã đọc nguồn ENG main/supplement, các phần báo cáo Gate 1–8/SHHS, bản VI, các BUILD/review/provenance liên quan; đối chiếu code, config, aggregate artifact và bibliography.
- Đã render và kiểm tra toàn bộ 67 trang: ENG 12, supplement 4, paper VI 12, báo cáo đầy đủ 39. PDF skill được dùng để kiểm tra bố cục, không suy luận layout chỉ từ LaTeX.
- 13 mục `SUBMISSION_MANIFEST.sha256` và 39 mục `REPORT_MANIFEST.sha256` đều khớp tại thời điểm kiểm tra. Đây là kiểm tra các mục được liệt kê, không chứng nhận package chứa mọi dependency.
- PDF bàn giao khớp bản PDF build tương ứng. Log hiện có không thấy citation/reference chưa xác định hoặc cảnh báo overfull/underfull qua kiểm tra mẫu lỗi.
- Tính lại độc lập macro-F1, N3 precision/recall/F1, tỷ lệ N3→N2 và oracle từ confusion counts đã lưu; các kết quả chính khớp. Có một sai khác làm tròn nhỏ ở oracle gộp hai kênh, nêu ở T06.
- Chạy toàn bộ test: `150 passed, 19 warnings`, khoảng 8,88 giây. Các cảnh báo không làm test thất bại; test xanh không chứng minh toàn bộ nghiên cứu không có leakage hoặc diễn giải đúng.
- **Không có các prediction/raw data tại ổ E: trong môi trường kiểm tra.** Không tái huấn luyện, không tái sinh prediction, không kiểm chứng lại toàn bộ CI từ các prediction ngoài repository. Các số CI đối chiếu với aggregate artifact hiện có.

## 2. Quy ước thực thi và ranh giới an toàn

- **P0:** phải xử lý trước nộp, hoặc thu hẹp/xóa claim liên quan nếu không thể cung cấp bằng chứng. Không được đánh dấu hoàn thành bằng cách viết một disclaimer không giải quyết mâu thuẫn.
- **P1:** rất nên làm để củng cố khả năng qua phản biện; có thể chọn không làm nếu ghi rõ lý do và giữ claim đủ hẹp.
- **P2:** cải tiến tùy chọn hoặc nghiên cứu tiếp theo; không tự nâng thành điều kiện bắt buộc của BSPC.
- Trạng thái từng việc: `TODO / IN_PROGRESS / PASS / BLOCKED_EXTERNAL / WAIVED_WITH_REASON`. Mỗi PASS cần đường dẫn output và kiểm tra kèm theo.
- Không sửa protocol lịch sử, prediction, nhãn, checkpoint hoặc kết quả để làm câu chuyện đẹp hơn. Mọi tái phân tích tạo output mới có version, provenance và nhãn hậu nghiệm.
- Không dùng 180 SHHS test subjects để fit/tune threshold, calibration hoặc model. Đây là cohort đã được xem kết quả; một phân tích mới trên đó không trở thành confirmatory chỉ vì script mới có hash.
- Không tự thêm/bớt tác giả, tự gán CRediT, tự xác nhận đạo đức hoặc public registration. Không tự publish repository, tạo release công khai, đăng preprint hay nộp bài.
- Không public raw SHHS, subject IDs hoặc dữ liệu giới hạn bởi DUA. Aggregate metrics và mã cũng phải được kiểm tra điều khoản trước công bố.
- Chỉnh file bằng patch, giữ thay đổi của người dùng. Không checkout/reset để khôi phục lịch sử lên working tree đang làm việc.
- Không coi diễn giải trong tài liệu này là nguồn số liệu thay cho artifact. Các giá trị bên dưới là mốc đối chiếu, phải đọc lại nguồn khi triển khai.

## 3. Bản đồ nguồn và nơi phải đồng bộ

Mọi đường dẫn trong tài liệu là tương đối với root repository, trừ khi nói khác.

| Nhóm | Nguồn / nơi chỉnh | Vai trò |
|---|---|---|
| Manuscript ENG | `Reports/paper_en/main.tex`, `references.bib` | Bản nộp chính; ưu tiên sửa trước |
| Supplement ENG | `Reports/paper_en/supplement.tex` | Chi tiết phương pháp, bảng đầy đủ, robustness/supporting analyses |
| Metadata/narrative ENG | `Reports/paper_en/highlights.txt`, `note.md`, `BUILD.md` | Không được còn claim trái với main |
| Paper VI | `Reports/paper/main.tex`, `references.bib`, `BUILD.md` | Đồng bộ sự thật và giới hạn suy luận; không cần dịch từng câu cơ học |
| Báo cáo đầy đủ | `Reports/main.tex`, `cover.tex`, `sec1_introduction.tex` đến `sec9_reproducibility.tex`, các file `sec*b_shhs_seed123_extension.tex` | Giữ chi tiết kỹ thuật; không ép độ dài bằng journal paper |
| Báo cáo kiểm định | `Reports/BUILD.md`, `REPORT_GATE1_8_AUDIT.md`, `PAPER_REFRAMING_REVIEW_VI.md`, `SHHS_PROTOCOL_PROVENANCE.md`, `SHHS_E6_PER_CLASS_REANALYSIS.md`, `SHHS_SEED123_E4_EXTENSION.md` | Phân biệt trạng thái hiện tại và ghi chép lịch sử |
| EDF metrics | `runs/v2/analysis/gate5_paired_results_seed42.json`, `gate5_paired_results_seed123.json`, `multiseed_sensitivity_seed42_seed123.json`, `runs/v2/gate8/analysis_seed42.json` | Nguồn số liệu EDF, không lấy từ notebook cũ |
| SHHS aggregate | `Reports/SHHS_E0_E3_E6_N3_AUDIT.json`, `SHHS_E3_E2_PAIRED_AUDIT.json`, `POSTHOC_E3_E0_AUDIT.json`, `docs/SHHS_*RESULTS.md` | Bằng chứng hiện có; phân biệt aggregate với prediction gốc |
| Boundary diagnostics | `runs/v2/analysis/transition_regions_edf_seed42_t2.json`, `transition_regions_shhs_t2.json`, hai file seed123 tương ứng | Hậu nghiệm, từ nhãn tham chiếu |
| Design/code | `configs/experiments_v2.json`, `configs/*shhs*.json`, `configs/gate8_context_ablation.json`, `src/sleeptcn/{experiment,models,features,metrics,statistics,preprocessing,shhs_preprocessing,shhs_transition_analysis}.py` | Kiểm tra thực sự đã thay gì và metric được tính thế nào |
| Splits / validation | `data/splits/sleepedf_sc_10fold_seed42_v2.json`, `data/manifests/*v2.json`, các test gate và run manifest SHHS ngoài repository | Subject isolation, provenance, tính toàn vẹn dữ liệu |
| Delivery | 4 PDF trong `Reports/output/pdf/`, 2 manifest `.sha256` | Chỉ cập nhật sau sửa, build và QA |

Không tự sửa JSON lịch sử chỉ vì trường mô tả trong đó dùng từ quá mạnh. Tạo interpretation errata liên kết snapshot. Đối với tài liệu báo cáo được duy trì hiện tại thì phải sửa đồng bộ. Nếu giữ một review cũ làm lịch sử, thêm trạng thái “superseded” và liên kết bản mới, không để người đọc hiểu đó là kết luận hiện hành.

## 4. Kế hoạch công việc chi tiết

### T00 — Đóng băng baseline review và tạo sổ thay đổi [P0]

**Đầu vào:** repository hiện tại và tài liệu này. **Phụ thuộc:** không.

1. Đọc `AGENTS.md` nếu đã xuất hiện; kiểm tra `git status --short`, `git rev-parse HEAD`. Không giả định working tree vẫn như ngày audit.
2. Đọc đầy đủ main ENG, supplement và config/code được dẫn ở T02–T04 trước khi sửa câu.
3. Tạo `Reports/BSPC_REVISION_LOG.md`: ID công việc, trạng thái, file đã đổi, nguồn kiểm tra, test, phần chưa xác minh.
4. Tạo `Reports/BSPC_CLAIM_EVIDENCE_LEDGER.md`: mỗi claim có population, model/seed, aggregation, evidence class, nguồn JSON/key, cách tính, vị trí xuất hiện ở ENG/VI/report.
5. Lưu giá trị hash baseline và phân biệt file khoa học bất biến với tài liệu dự kiến chỉnh. Không tạo commit/tag hoặc publication bên ngoài nếu chưa được phép.

**PASS:** có log và ledger; mọi giới hạn truy cập được ghi đúng; chưa đụng dữ liệu/manifest lịch sử.

### T01 — Chốt định vị, câu hỏi nghiên cứu và phạm vi “controlled” [P0]

**Vị trí:** main ENG 160–204, 911–989, 1027–1055; `note.md`; cả hai BUILD; mở đầu/kết luận VI và báo cáo.

1. Giữ định vị empirical pipeline evaluation, không gán yêu cầu phải phát minh ResNet/TCN mới.
2. Xóa tiền đề “one component changes at a time” như mô tả chung toàn bộ E0–E6. Nêu chính xác cái được giữ: subject partitions, đánh giá, nguồn dữ liệu; các khác biệt khác công khai trong design matrix.
3. Đổi câu hỏi “kiến trúc riêng lẻ có vượt trội?” thành “các cấu hình/thay thế pipeline đã đánh giá có tạo lợi ích đo được và ổn định không?”.
4. Đổi “phân bổ effort/label budget tối ưu” thành bằng chứng mô tả giúp đề xuất hướng kiểm chứng. Nghiên cứu chưa định lượng chi phí thu nhãn hay tối ưu annotation budget.
5. Title hiện có thể giữ nếu giải thích controlled ở đúng phạm vi. Phương án hẹp hơn, không bắt buộc: `Pipeline Choices and Cross-Cohort Error Patterns in Single-Channel EEG Sleep Staging: An Evaluation on Sleep-EDF and SHHS1`.
6. Nếu đổi title, đồng bộ title hiển thị, PDF metadata, supplement, cover letter và nội dung liên quan; không chỉ đổi `\title`.

**Câu định vị gợi ý:**

> This study evaluates specified sleep-staging pipeline configurations under shared subject partitions and characterizes their cross-cohort error patterns. It does not propose a new backbone architecture, establish architecture-only causal effects, or validate a target-domain adaptation method.

**PASS:** title, abstract, ba câu hỏi, contributions và conclusion cùng nói về một bài; không hứa chứng minh điều không được thí nghiệm.

### T02 — Sửa E1: encoder được giữ nhưng training recipe không được giữ [P0, lỗi đã xác minh]

**Vị trí:** main ENG 122, 166–171, 177, 187, 351, 379, 975, 1033; highlights; `Reports/paper_en/BUILD.md`; các bản VI/report. **Nguồn:** `configs/experiments_v2.json`, `src/sleeptcn/experiment.py` khoảng 523–565.

| Thuộc tính sequence model | E0 BiLSTM | E1 TCN |
|---|---|---|
| Encoder và cached features | CNN15, dùng chung với E1 | Tái sử dụng E0 encoder/cache |
| Optimizer | Adam | Adam |
| Learning rate | 0.01 | 0.0005 |
| Batch size | 4 recordings | 8 recordings |
| Maximum epochs | 1000 | 300 |
| Early-stopping patience | 10 | 30 |

1. Đối chiếu config với effective run configuration/checkpoint metadata nếu truy cập được. Nếu khác nhau, báo mâu thuẫn trước khi chọn nguồn đúng; không lấy default hiện hành thay cho lịch sử chạy.
2. Thay toàn bộ claim `changed only the sequence model`, `isolates the sequence model`, `isolated architectural contrast` bằng mô tả cấu hình và training recipe.
3. Câu thay trực tiếp cho Methods/Introduction:

> E1 replaces the E0 BiLSTM sequence-model configuration with a TCN configuration while reusing the E0 encoder and cached features. The sequence-model training settings also differ, as detailed in Table S[training]; therefore, E1–E0 does not isolate an architecture-only effect.

4. Nhãn ngắn trong bảng: `Sequence-model and training-recipe replacement; encoder/features reused`.
5. Bản VI: `E1 thay cấu hình mô hình chuỗi BiLSTM bằng TCN và tái sử dụng encoder/đặc trưng của E0; lịch huấn luyện mô hình chuỗi cũng khác, nên đây không phải phép cô lập tác động kiến trúc.`
6. Không phủ định giá trị thực nghiệm: đây vẫn là so sánh cấu hình thực tế có ích. Chỉ khi tác giả muốn giữ causal claim architecture-only mới cần nhánh thí nghiệm T20-B.

**PASS:** không còn mô tả E1–E0 architecture-only trong các tài liệu hiện hành; bảng training có đủ khác biệt; claim runtime/accuracy gọi đúng cấu hình.

### T03 — Hoàn chỉnh E2 và bảng thiết kế E0–E6 [P0]

**Vị trí:** main ENG Methods khoảng 328–384; supplement 12–40. **Nguồn:** config, `experiment.py`, `models.py`, `features.py`.

1. Giữ câu đã sửa đúng: `E2 replaces the E1 feature/context package with a current-epoch ResNet-1D encoder.`
2. Thêm khác biệt encoder training, không chỉ 75→128 chiều và bỏ P/N:

| Encoder | CNN15 của E0/E1 | ResNet của E2–E6 |
|---|---|---|
| Input/context | Các nhánh C/P/N theo thiết kế | Current epoch |
| Output chuyển cho sequence model | 15 × 5 softmax outputs = 75 | 128-dimensional representation |
| Optimizer | Adam | AdamW |
| Learning rate | 0.001 | 0.001 |
| Weight decay | Xác minh effective value của run | 0.0001 |
| Batch size | 64 epochs | 64 epochs |
| Maximum epochs / patience | 1000 / 10 | 40 / 8 |
| Encoder selection | Validation weighted loss | Validation macro-F1 |

3. Không gọi softmax outputs là “calibrated posteriors” nếu chưa đo probability calibration. Không mô tả 15 CNN thành 15 binary heads: kiểm tra đúng 5-way output mỗi nhánh.
4. Sửa `shared selection rules` thành mô tả cụ thể: splits được giữ; selection của encoder khác; selection sequence checkpoint theo validation macro-F1 nếu run xác nhận.
5. Tạo bảng design matrix đầy đủ E0–E6: signal operations theo đúng thứ tự; feature/context; encoder; encoder training/selection; sequence model; sequence training/selection; frozen/shared stage; inference context; model weights có dùng target hay không; loại contrast.
6. E5 là identifier của identity audit, không phải model thua nên bỏ. Trình bày riêng 153/153 EDF recordings E4/E5 bitwise identical, clipping fraction zero.
7. Mô tả E3–E2 là contrast preprocessing trong cấu hình đã định, không tự rút ra hiệu quả độc lập của từng operation. E3–E6 cũng phải đọc đúng toàn bộ khác biệt source preprocessing/training.

**PASS:** người khác dựng được các cấu hình từ main+supplement, không phải đoán optimizer, selection hoặc context; không có bảng nào nói “all other components fixed” sai.

### T04 — Khôi phục chi tiết Methods đủ tái lập [P0]

**Vị trí:** main 270–414; supplement; report `sec3_dataset`, `sec4_methodology`, `sec5_preprocessing`, `sec9_reproducibility`.

1. Lập bảng layer-by-layer lấy từ code: kernel, stride, channel, pooling, residual blocks, dropout, normalization, TCN dilation, padding/cropping, BiLSTM hidden size/directions. Bổ sung mọi hyperparameter mà main đang hứa “provided in the supplement” nhưng supplement chưa có.
2. Ghi rõ hai giai đoạn train, cách tạo cache, thời điểm freeze, cách đảm bảo encoder của outer fold không nhìn validation/test ngoài phạm vi cho phép. Kiểm tra cả labels lẫn fitted preprocessing/features.
3. CNN15 class-specialization: đối chiếu công thức trọng số theo training counts; hiện code dùng `w_target = 0.5 / (n_c/N)` và `w_other = 0.5 / ((N-n_c)/N)`. Giải thích đó là loss weighting cho output đa lớp, không tự đổi thành binary objective.
4. Context C/P/N: ghi cách dịch trong từng recording, edge replication và central-epoch target; không nối context qua ranh giới recording hoặc subject.
5. Sequence training/inference trên recording, padding và mask valid labels; không gọi benchmark 100 epochs là training window nếu actual training dùng toàn recording.
6. TCN receptive field theo code hiện tại là 253 epochs, đối xứng khoảng ±126 epochs; ghi rõ lý thuyết và biên padding. Kiểm lại bằng test/code trước chèn. Đây không phải causal online model.
7. Preprocessing: đơn vị điện thế và conversion; sampling rate/resampling; filter order/cutoff; zero-phase hay causal; thứ tự filtering, clipping, z-score, cropping; cách xử lý NaN/flat record, epsilon; training statistics so với per-record statistics. Không chỉ ghi “standard preprocessing”.
8. E6 mean/std lấy toàn bộ recording không nhãn ở cả source/target theo implementation; nói rõ có bao gồm phần ngoài đoạn được chấm/trim hay không. Không đổi preprocessing để làm Methods thuận miệng.
9. Wake trimming và unknown/movement mask: mô tả rule, nhãn dùng để tìm sleep interval, thời điểm áp dụng; số epoch trước/sau loại bỏ. Đây là evaluation convention có sử dụng reference labels, không được mô tả là một bước triển khai không cần nhãn hoàn toàn.
10. SHHS inference: 10-fold model probability ensemble, alignment class order, mean probability rồi argmax; model selection không dựa target test.
11. Runtime: GPU/CPU, software versions, precision, batch/window, warm-up, repetitions, synchronization, inference-only versus preprocessing/feature extraction, peak allocated/reserved memory. Kiểm tra `scripts/benchmark_model_complexity.py` và artifact, không suy ra thiết lập từ máy review.
12. Dừng sớm khiến training wall-clock khác cả recipe lẫn model; nêu “observed training-and-validation wall-clock” chứ không quy toàn bộ tốc độ cho TCN.

**PASS:** có bảng kiến trúc + bảng train và đoạn xử lý dữ liệu đủ chi tiết; mọi lời hứa dẫn sang supplement đều có nội dung tương ứng; ngoại tuyến được khai báo nhất quán.

### T05 — Sửa mâu thuẫn CI, kiểm định và estimand [P0, lỗi đã xác minh]

**Vị trí:** main ENG 392–393, 448–479; Abstract Results; bảng liên quan trong VI/report. **Nguồn:** `src/sleeptcn/statistics.py`, `metrics.py`, `gate5_paired_results_seed42.json`.

Các số đối chiếu EDF seed 42:

| Contrast | Δ pooled macro-F1 | Bootstrap CI 95% | Holm p của subject-wise Wilcoxon | W/L |
|---|---:|---|---:|---:|
| E1−E0 | 0.004811 | [−0.000915, 0.010193] | 0.1022 | 51/27 |
| E2−E1 | 0.003251 | [−0.002370, 0.008808] | 0.1036 | 44/34 |
| E3−E2 | 0.006962 | [0.000305, 0.014520] | 0.8989 | 37/41 |
| E3−E6 | 0.021319 | [0.012179, 0.030698] | 0.001185 | 49/29 |

1. Sửa câu “remaining ... intervals containing zero”: sai với E3−E2. Thay bằng:

> E3–E6 was the only pre-specified contrast with both a positive pooled-metric bootstrap interval and a Holm-significant subject-level Wilcoxon result. E1–E0 and E2–E1 had intervals including zero. E3–E2 had a positive pooled-metric interval but a non-significant subject-level Wilcoxon result.

2. Sửa abstract `Other contrasts were small and non-significant` để chỉ rõ “none of the other pre-specified contrasts was significant under the subject-level Wilcoxon–Holm analysis”, tránh phủ định CI dương.
3. Định nghĩa `pooled macro-F1 = macro-F1(sum of subject confusion matrices)` và `subject-mean macro-F1 = mean(per-subject macro-F1)`. Không dùng cùng ký hiệu delta mà giấu estimand.
4. Bootstrap resample subjects, giữ các recording cùng subject và paired models cùng index; kiểm tra số lần bootstrap, seed, percentile/BCa đúng artifact. Không bootstrap epochs độc lập để làm CI hẹp.
5. Wilcoxon là paired signed-rank trên subject differences, không phải kiểm định “subject-averaged rank quantity” hoặc kiểm định mean. Nêu assumptions/zero handling theo implementation; không suy ra mean gain chỉ từ p-value.
6. Pooled macro-F1 không phải weighted average của per-subject macro-F1. Xóa kết luận hiện tại rằng chênh lệch hai phép suy luận chứng minh gain do một nhóm subject có nhiều epochs. Chưa có influence analysis xác nhận điều đó.
7. Câu thay: `The pooled and subject-wise analyses summarize different aspects of performance. Their differing results do not establish a uniform subject-level benefit and do not, by themselves, identify which subjects drive the aggregate change.`
8. Ghi các bootstrap CI là marginal 95% nếu chưa chỉnh multiplicity; Holm áp vào family p-values được chỉ định, không tự gọi CI là family-wise simultaneous interval.
9. Sign test/W–T–L nếu giữ: ghi descriptive/secondary và có hay không multiplicity adjustment; không chọn test có p nhỏ nhất làm kết luận chính.
10. Kiểm tra macro-F1 khi một subject không có class: code dùng 5 classes cố định và zero-denominator convention. Ghi rõ, không đổi averaging rule giữa cohort/region để tạo ranking mong muốn.

**PASS:** abstract, bảng, Results và Discussion không còn mâu thuẫn; mọi metric/CI/p đều chỉ rõ aggregation và family; không diễn giải non-significant thành equivalence.

### T06 — Kiểm toán số liệu và oracle, tạo một nguồn sự thật [P0; rounding nhỏ P2]

**Nguồn:** aggregate JSON trong mục 3; SHHS prediction gốc nếu cung cấp; code metrics. **Output:** ledger có đường dẫn JSON/key và test numeric consistency.

Mốc kiểm tra từ confusion counts SHHS seed 42, N3 support = 22.806, toàn bộ scored epochs = 169.012:

| Model | Pooled macro-F1 | N3 precision | N3 recall | N3 F1 | N3→N2 count | N3→N2 / true N3 |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 0.576107813 | 0.900741 | 0.261027800 | 0.404759477 | 16.480 | 72.261685% |
| E3 | 0.609888602 | 0.944044 | 0.258177673 | 0.405467755 | 16.674 | 73.112339% |
| E6 | 0.573238197 | 0.905186 | 0.200517408 | 0.328307847 | 17.764 | 77.891783% |

1. Tính accuracy, Cohen's kappa, per-class precision/recall/F1, macro-F1 trực tiếp từ counts bằng một đường tính độc lập, không chỉ gọi lại cùng hàm rồi gọi là independent verification.
2. Kiểm tra class order W/N1/N2/N3/REM, axis true/pred, số subjects/epochs và excluded labels trước đối chiếu.
3. EDF: 195.469 scored epochs, 298 ignored, tổng 195.767. Kiểm tra tổng lớp và tất cả model dùng cùng masks.
4. SHHS paired summary dùng subject-mean khác pooled table: E3−E0 ≈ 0.041219; E3−E6 ≈ 0.02735875. Không thay bằng chênh pooled chỉ vì dễ lấy từ bảng.
5. Oracle chỉ sửa đúng cell đã định trong E3 confusion matrix rồi tính lại metric; không diễn giải là một classifier thực hiện được:
   - EDF E3 pooled macro-F1 = 0.7904430932802019.
   - SHHS E3 pooled macro-F1 = 0.609888602120497.
   - Sửa N3→N2: 0.7443519081917789; tỷ phần chênh lệch mô tả = 74.4724%, làm tròn 74.5%.
   - Sửa N2→REM: 0.6596215406428867; 27.5446%.
   - Sửa cả hai: 0.7946825493852532; 102.3480%, làm tròn một chữ số là **102.3%, không phải 102.4%** ở `Reports/sec6_results.tex` khoảng 553.
6. Không cộng hai tỷ lệ oracle riêng rồi cho rằng phải bằng oracle đồng thời; macro-F1 phi tuyến. “Hơn 100%” là vượt chênh hai score, không phải phục hồi hơn 100% bệnh nhân hoặc chứng minh domain gap đã được giải quyết.
7. Lưu full precision ở artifact, render rounding ở presentation. Không lấy số đã làm tròn trong PDF làm đầu vào tính delta/ratio.
8. Runtime đối chiếu: 3.76× speed-up, 4.37× parameters, 28.4% more peak memory. Nếu muốn dùng percent latency reduction, tính `100*(1 - 1/speedup)` từ raw timing; không viết “3.76× lower latency”.

**PASS:** mọi số trong main/supplement/VI/report/highlights có nguồn và rounding thống nhất. Nếu không có raw artifact, phân biệt “recomputed from stored aggregate counts” với “recomputed from predictions”.

### T07 — Thu hẹp diễn giải E6, cơ chế biên độ và khuyến nghị adaptation [P0]

**Vị trí:** main ENG 249–250, 711–720, 946–989, 1038–1050; report `sec4_methodology.tex`, `sec7_discussion.tex`, `sec8_conclusion.tex`; note/BUILD/review VI.

1. Giữ khai báo E6: mean/std được tính từ **toàn bộ recording đích không nhãn**. Thuật ngữ thống nhất: `label-free target-record normalization with a transductive dependency`. Model weights source-only không làm phép chuẩn hóa toàn recording thành inductive thuần.
2. Xóa suy luận hiện có rằng E6 không cải thiện thì absolute amplitude không đủ giải thích lỗi. E6 là một pipeline được train riêng với preprocessing riêng, không phải phép can thiệp amplitude-only lên cùng một model đóng băng.
3. Câu thay cho đoạn E6:

> Under the evaluated E6 pipeline, SHHS1 N3 recall was 0.2005, compared with 0.2582 for E3. The tested record-wise z-scoring pipeline therefore did not remedy N3 under-detection. This comparison does not isolate the role of absolute amplitude or rule out amplitude- or montage-related contributors.

4. Nếu giữ boundary recall 0.0721/0.0733, ghi đúng region và seed, không nói chúng “bằng nhau” theo nghĩa equivalence.
5. Không đổi kết luận thành “mọi chuẩn hóa không nhãn đều không hiệu quả”, “bắt buộc phải có target labels”, hoặc “cần thay đổi đặc trưng vật lý”. Không thử một phương pháp không đồng nghĩa loại trừ cả họ phương pháp.
6. `Preprocessing was the stronger development axis` → `The observed post-hoc preprocessing contrast was larger than the evaluated sequence-model and feature/context contrasts on this SHHS1 sample.` Thêm không phải factorial effect, phép so sánh effect-size có kiểm định trực tiếp, hay bằng chứng về hiệu quả đầu tư nguồn lực.
7. Bỏ “roughly seven times the largest architectural contrast”: E1 và E2 không phải pure architecture contrasts. Nếu còn ratio mô tả thì ghi đúng mẫu số, seed, estimand, và giải thích không xác lập thứ hạng tổng quát.
8. Đổi “N2–N3 and N2–REM should be prioritised for labelled adaptation” thành `N3→N2, and secondarily N2→REM in E3, are candidate targets for future adaptation studies.` Không biến oracle thành bằng chứng một chiến lược lấy nhãn tốt hơn random sampling.
9. Thay phát biểu unconditional về prior shift làm classifier miscalibrated bằng conditional statement. Predicted-to-true class ratio là marginal emission diagnostic, không đo probability calibration. Prior correction phụ thuộc assumptions; chưa fit/validate thì không nói nó đủ hoặc không đủ.
10. Xóa khẳng định wake-trimming policy “alone moves” score bằng các architecture contrasts khi không có trim ablation tương ứng; giữ nhận xét có thể thay đổi class composition và khả năng so sánh giữa nghiên cứu.

**PASS:** mỗi câu nguyên nhân, remedy, optimality đều có thí nghiệm hỗ trợ hoặc được chuyển thành hypothesis/future work; không đánh đồng “không cải thiện trong E6” với “cơ chế đã bị loại trừ”.

### T08 — Tách cross-cohort score difference khỏi hiệu ứng domain shift thuần [P0]

**Vị trí:** đoạn transfer gap/oracle, Discussion, Limitations; bảng EDF/SHHS. **Nguồn:** OOF setup và SHHS ensemble code.

1. Nêu EDF score từ out-of-fold predictions: mỗi subject do model chưa train trên subject đó dự đoán. SHHS score dùng ensemble các outer-fold models.
2. Chênh 0.7904−0.6099 ≈ 0.1805 là khác biệt quan sát giữa hai cohort/evaluation settings, không phải ước lượng riêng tác động của montage, tuổi, sampling rate hay “domain shift” với mọi thứ khác giữ nguyên.
3. Class prevalence, cohort composition, scoring convention và ensemble cũng khác. Không giải thích toàn bộ gap bằng duy nhất montage hoặc N3 amplitude.
4. Gắn caveat vào oracle 74.5%: phần của **observed pooled-score difference**, không “74.5% causal domain-shift loss”.
5. Tuyệt đối không tạo EDF 10-model ensemble từ các model đã train trên chính subject test để làm hai bên “giống nhau”. Việc đó gây leakage.
6. Tùy chọn T20-A: nếu có prediction từng fold trên SHHS, báo sensitivity individual-fold model versus ensemble, vẫn cùng subjects; không coi 10 folds là 10 independent cohort replications.

**PASS:** Methods và Limitations khai báo evaluation asymmetry; oracle/cross-cohort claims đúng phạm vi.

### T09 — Mô tả chọn cohort, demographics và masking [P0 về mô tả; P1 về bảng demographics mới]

**Vị trí:** main Dataset/External evaluation; report `sec3_dataset.tex`; supplement. **Nguồn:** `scripts/select_shhs_subjects.py`, `configs/shhs_v1_protocol.json`, split/metadata manifests.

1. SHHS selection: kiểm tra trường quality `overall_shhs1`, tập giá trị 3/4/5/6/7, bộ lọc đủ điều kiện, sort deterministic bằng SHA-256 và selection seed 42. Mô tả để người khác hiểu cohort không được chọn theo score model.
2. Khai báo 220 subjects được phân vai: 5 adaptation, 15 validation, 180 test, 20 reserve; policy no replacement theo protocol. Giải thích technical pilot 10 người dùng 5 adaptation + 5 validation, và test không trộn vào pilot.
3. Phân biệt “180 locked subjects” với “mọi phân tích đều pre-specified trước mở test”: chỉ câu đầu đúng cho nhiều extension hậu nghiệm.
4. Nêu EDF SC 78 subjects/153 recordings; không nhầm SC-20. Dataset chính thức có tuổi 25–101, nên không viết nhóm EDF này mặc nhiên là “young adults”. [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/).
5. Tạo demographics aggregate cho chính 78 EDF và 180 SHHS: số subjects/recordings, age mean±SD và range hoặc median/IQR, sex counts, scoring standard, montage, sampling rate, epoch counts/class proportions, missingness. AHI/clinical characteristics chỉ thêm nếu truy cập hợp lệ và thực sự dùng.
6. Không dùng full-cohort published average để đại diện 180 người được chọn. Không công bố từng subject ID. Nếu thiếu metadata, để task blocked và bỏ so sánh tuổi định lượng/khẳng định trẻ hơn cho tới khi có số đúng.
7. R&K→five-class mapping và SHHS scoring/mapping lấy từ data documentation và code; ghi việc gộp stage 3/4, unknown/movement exclusions, không tự giả định cả hai cùng AASM scoring.
8. Làm rõ recording-wise trimming có dựa reference labels, boundary analysis cũng dựa reference labels; “label-free transfer” mô tả fitting/inference weights/statistics, không có nghĩa evaluation không dùng reference labels.

**PASS:** bảng cohort không bịa số; tất cả điều kiện chọn và role allocation được mô tả; tuyên bố demographic không vượt số liệu của mẫu thực tế.

### T10 — Bổ sung bảng bằng chứng trung tâm vào main/supplement [P0]

**Vị trí:** main transfer Results, supplement; hiện một số bảng quan trọng nằm trong `\iffalse` và không xuất hiện ở PDF.

1. Đưa một bảng ngắn E0/E3/E6 N3 precision, recall, N3→N2 vào main; đây là bằng chứng trung tâm của framing cross-model failure. Có thể dùng ba hàng của T06, làm tròn 4 chữ số/1 chữ số phần trăm. Không cần đưa tất cả confusion matrices vào main.
2. Bổ sung bảng paired SHHS E3−E2, vì claim preprocessing lớn dựa vào contrast này. Ghi `post-hoc on the already evaluated cohort`, Δ subject-mean ≈ 0.04750, CI [0.03724, 0.05792], W/L 147/33, p theo đúng audit và đúng family nếu trình bày.
3. E1/E2 SHHS: secondary, không trộn thành primary campaign. E1−E0 ≈ 0.00651; E2−E1 ≈ −0.01280; giữ CI/family từ artifact.
4. Supplement cần bảng paired extension seed 123, không chỉ bảng score: main đang viện dẫn Holm significance nhưng supplement chưa trình bày đủ.

| Seed 123 extension contrast | Δ subject-mean macro-F1 | CI 95% | Holm p trong family bốn contrast |
|---|---:|---|---:|
| E4−E2 | 0.0323 | [0.0236, 0.0410] | 7.40×10^-13 |
| E3−E4 | −0.0098 | [−0.0125, −0.0073] | 1.89×10^-10 |
| E3−E6 | 0.0131 | [0.0023, 0.0239] | 0.0085 |
| E3−E0 | 0.0331 | [0.0225, 0.0438] | 5.19×10^-9 |

5. Nguồn bảng trên: `Reports/SHHS_SEED123_E4_EXTENSION.md` và linked artifact. Trình bày family của extension tách family hai primary SHHS contrasts; không so sánh p như cùng một phép chỉnh.
6. Seed 42 EDF E3 cao nhất trong các model thử; seed 123 EDF E4 0.788673 > E3 0.788265. Seed 123 SHHS E4 cũng cao hơn E3. Sửa bold/ranking và mọi “best in campaign” không gắn seed/scope.
7. Không khôi phục nguyên xi khối `\iffalse`: nội dung cũ chứa claim đã bị bỏ. Chỉ tái dựng bảng từ nguồn hợp lệ, caption/footnote mới.

**PASS:** người đọc paper+supplement thấy được số liệu hỗ trợ các claim trung tâm mà không cần mở báo cáo nội bộ; primary/secondary/post-hoc và families rõ ràng.

### T11 — Làm cho boundary diagnostic có thể hiểu và tái lập [P0 cho định nghĩa; P1 cho seed 123]

**Vị trí:** supplement bảng vùng nhãn hiện được gọi S11; main diagnostic paragraph; report. **Nguồn:** `scripts/analyze_transition_regions_edf.py`, `scripts/analyze_transition_regions_complete.py`, transition analysis code và JSON.

1. Thêm định nghĩa anchor: epoch đầu mang nhãn mới tại một cặp epoch liên tiếp có nhãn hợp lệ; invalid-label gaps không tự tạo transition liên tục.
2. Radius-one neighbourhood là union của `{j−1, j, j+1}` quanh anchor. Stable interior hiện dùng distance ≥3 theo code. Distance 2 là vùng trung gian, **không** thuộc một trong hai nhóm đang so sánh.
3. Báo support: EDF near = 43.237, stable = 130.693, intermediate = 21.539; SHHS near = 44.744, stable = 103.195, intermediate = 21.073. Kiểm lại tổng bằng scored masks.
4. Giải thích persistent transition rule ≥3 same-label epochs mỗi phía nếu báo loại đó. Tách all-transition neighbourhood khỏi riêng N2–N3 transition; không gọi cả hai là cùng tập.
5. Reference-label neighbourhood không phải detector online, thời điểm chuyển sinh lý đo trực tiếp, hay evidence transition gây lỗi. Class composition giữa near/stable khác, pooled metric khác không tự chứng minh độ khó nhân quả.
6. Support N3 trong từng vùng và cách xử lý subject không có true N3 phải rõ. Không im lặng bỏ subject làm tăng mean recall.
7. Đã tìm thấy seed 123 aggregate ở local `runs/v2/analysis/transition_regions_edf_seed123_t2.json` và `transition_regions_shhs_seed123_t2.json` cùng `.md/.csv`; lúc audit chưa thấy được theo dõi bởi Git. Kiểm tra script version, inputs, masks, source hashes, ngày tạo và prediction provenance trước sử dụng. Có file không tự chứng minh tính hợp lệ.
8. Sau xác minh, đưa vào supplement như post-hoc seed sensitivity, không “independent validation cohort”. Nêu cả kết quả thuận và không thuận, không chọn riêng E3 improvement.

Các recall N3 để đối chiếu (E0 / E3):

| Cohort và region | Seed 42 | Seed 123 |
|---|---|---|
| EDF near transition | 0.6121 / 0.6326 | 0.6174 / 0.6385 |
| EDF stable | 0.9216 / 0.9316 | 0.9162 / 0.9285 |
| SHHS near transition | 0.0633 / 0.0733 | 0.0639 / 0.0656 |
| SHHS stable | 0.4008 / 0.3900 | 0.3885 / 0.3660 |

9. Diễn giải: stable-region N3 deficit cũng tồn tại ở SHHS; do đó một mô tả chỉ dựa boundary không bao trùm hết lỗi. Không suy ra nguyên nhân sinh lý từ bảng.
10. Nếu thiếu prediction gốc để kiểm tra seed123, vẫn hoàn thành định nghĩa phương pháp; ghi seed123 evidence chưa được independently reverified và quyết định có đưa vào dựa mức provenance thực có.

**PASS:** region definitions, supports, exclusions và evidence class đầy đủ; seed123 chỉ được đưa vào sau validation có ghi log.

### T12 — Làm rõ context ablation và calibration diagnostics [P0 về claim; P1 về số liệu bổ sung]

1. Supplement context ablation hiện thiếu protocol. Đọc `configs/gate8_context_ablation.json` và implementation trước viết.
2. Nêu group masking bằng training means từ valid training data, áp dụng train/validation/test; sequence model được retrain ở mỗi điều kiện, FULL reuse E1 theo thiết kế. Không mô tả như inference-only removal nếu không đúng run.
3. Mask giữ dimensionality nên kết quả nói về thông tin incremental trong setup này, không chứng minh P/N hoàn toàn vô ích, không phải lượng phần trăm thông tin vật lý độc lập.
4. Silhouette: ghi space, scaling, sampling, distance, class labels, seed; không coi silhouette cao là chứng minh classifier tốt hoặc cơ chế chuyển miền. Giữ supplementary.
5. Trường `preregistered_before_gate8_training` trong JSON là metadata nội bộ; không dùng làm bằng chứng đăng ký công khai. Sửa “đã đăng ký” trong `Reports/sec6_results.tex` khoảng 87 nếu không có registration thực.
6. `src/sleeptcn/shhs_transition_analysis.py` có mã ECE và multiclass Brier; chưa tìm thấy output tương ứng trong aggregate đã đọc. Trước viết “chưa thực hiện calibration analysis”, tìm toàn bộ output hợp lệ: tồn tại code ≠ đã chạy; không thấy trong một JSON ≠ không tồn tại ở nơi khác.
7. Nếu có valid output, đánh giá xem có liên quan câu hỏi không rồi bổ sung descriptive calibration, kèm binning/ECE definition/Brier scaling/subject aggregation và provenance. Nếu cần chạy mới từ prediction probabilities, ghi post-hoc, không fit calibration trên test.
8. Statements “không thể tính IQR/per-subject distribution vì chưa archive vectors” phải sửa thành thiếu trong aggregate hiện có; nếu prediction gốc còn thì các đại lượng có thể khôi phục. Không invent availability, cũng không tuyên bố bất khả thi quá mức.

**PASS:** context ablation được mô tả đúng intervention; không có claim calibration suy ra từ class margins; trạng thái “có code/có output/đã kiểm chứng” phân biệt rõ.

### T13 — Rà related work theo đúng định vị và đối chiếu citation–claim [P0 cho citation sai; P1 cho định vị]

**Vị trí:** Related work, Discussion về N3/montage/age; ba `references.bib`.

1. Giữ SleepInceptionNet và ADAST đã có. Đọc primary paper, không chỉ abstract/model-generated summary.
2. Tạo bảng làm việc, có thể rút gọn đưa vào supplement hoặc một đoạn Related work: nghiên cứu, cohorts, single-channel derivation, epoch/sequence context, preprocessing factors, target labels, target weight adaptation, paired subject uncertainty, cross-class error diagnostic, khác biệt thực sự với bài này.
3. Những đối chiếu cần ưu tiên:
   - SleepInceptionNet khảo sát preprocessing/representation cho single-channel sleep staging trên MESA; vì vậy không tuyên bố ý tưởng so sánh preprocessing là mới tự thân. Nêu bài hiện tại khác ở pipeline contrasts, paired analysis và EDF→SHHS cross-model error audit. [Haghayegh et al., 2023](https://www.jmir.org/2023/1/e40211/).
   - Nghiên cứu của Van Der Donckt và cộng sự cho thấy hướng đánh giá pipeline truyền thống trên nhiều bộ dữ liệu đã có chỗ trong BSPC. Đây là tiền lệ về loại đóng góp, đồng thời buộc làm rõ thêm bài hiện tại cung cấp gì. Không so score trực tiếp nếu protocol khác. [Primary preprint](https://arxiv.org/abs/2207.07753), [bài BSPC](https://www.sciencedirect.com/science/article/pii/S1746809422008837).
   - ADAST học thích nghi miền không nhãn với cập nhật mô hình; khác source-trained weights cố định cộng per-record normalization của nghiên cứu này. Không gọi E6 hoàn toàn “không adaptation” nếu đang dùng adaptation theo nghĩa rộng; viết `no learned/weight adaptation`. [ADAST](https://arxiv.org/abs/2107.04470).
   - Davidson et al. 2025 về tiêu chí N3, tuổi/giới và slow-wave amplitude trên SHHS là tài liệu trực tiếp liên quan mà bibliography hiện chưa có. Nó cung cấp bối cảnh về scoring, **không** chứng minh nguyên nhân N3 error của E0/E3. [Is it time to revisit the scoring of slow wave (N3) sleep?](https://academic.oup.com/sleep/article/48/10/zsaf063/8074201).
   - So sánh implementation E0 với ZleepAnlystNet gốc: objective, optimizer, validation/checkpoint scheduling, context và splits. Gọi `reimplementation/internal control`, không ngụ ý tái lập nguyên bản tuyệt đối nếu có khác biệt. [Primary Scientific Reports paper](https://www.nature.com/articles/s41598-024-60796-y).
4. Citation Ohayon 2004 chủ yếu về thay đổi sleep parameters/stage proportions theo tuổi; kiểm tra primary evidence trước dùng hỗ trợ một câu cụ thể về điện thế slow waves. Nếu câu về amplitude không có bằng chứng phù hợp, đổi claim hoặc dẫn nguồn đo amplitude trực tiếp.
5. Citation về slow-wave topography không tự chứng minh chênh voltage của đúng montage Fpz–Cz và C4–A1. Tìm primary montage-comparison evidence; PhysioNet dẫn van Sweden et al. 1990 về alternative electrode placement, có thể đọc để kiểm tra tính phù hợp trước thêm.
6. Mỗi đoạn novelty cần phân biệt “chưa được kiểm tra trong protocol/mẫu này” với “chưa ai nghiên cứu”. Không viết `first` nếu chưa có systematic search đủ mạnh.
7. Kiểm tra title, authors, year, DOI, volume/pages; bảo vệ acronym trong BibTeX bằng braces nếu style làm mất EEG/TCN/SHHS. Không đổi citation key chỉ vì key cũ đặt tên không đẹp nếu metadata đúng.
8. Đồng bộ metadata của các references dùng chung trong ba bib; không bắt cả ba file chứa cùng mọi bài nếu nội dung báo cáo khác.

**PASS:** có novelty paragraph cụ thể và khiêm tốn nhưng không tự phủ định đóng góp; mọi citation hỗ trợ đúng claim; không bảng SOTA xếp hạng bằng số lấy từ protocol khác.

### T14 — Hoàn tất provenance, không “sửa cho hash khớp” [P0]

**Vị trí:** main Reproducibility; `Reports/SHHS_PROTOCOL_PROVENANCE.md`; BUILD; `REPORT_GATE1_8_AUDIT.md`; manifests. **Phụ thuộc:** T00 và xác định mọi artifact mới được dùng.

Phải tách ba vấn đề:

1. **Integrity:** byte của một file hiện tại có khớp hash được ghi không?
2. **Provenance:** file đó có đúng là input/config của run đang báo cáo, và có bằng chứng về chronology không?
3. **Reproducibility:** người có quyền dữ liệu có đủ code, environment, split, command, model/artifact để tái chạy hoặc tái tính không?

Một file hash khớp chỉ giải quyết (1). Việc xác định snapshot và run manifest có thể củng cố (2); không tự chứng minh (3), public preregistration hoặc thời điểm tạo.

Các mốc hiện tại:

| Artifact | SHA-256 / tình trạng |
|---|---|
| `configs/shhs_zero_shot_v1.json` | `165d7cdf614ff071da7bd5ca94eb4e52dd8bee1ce5eafb712c2c8a0d0550fe93` — snapshot khớp giá trị protocol được lưu trong hồ sơ |
| `configs/shhs_v1_protocol.json` | `9541e2334cdae98b5d36b95a2656993cdaaffb35a3160dad70af11d14b653fe9` — expanded post-run audit, không thay snapshot trên |
| `Reports/SHHS_E0_E3_E6_N3_AUDIT.json` | `3333b5f12788e592323f05a3a1514fdd3fe724832fcc34e0419669c57223b0db` — hash file hiện tại |
| Inline hash N3 audit trong `Reports/BUILD.md:55`, `REPORT_GATE1_8_AUDIT.md:20` | Còn giá trị cũ `6b12447f27cf71f8a7b7c100919ab5438dc69fc3efd9f4d7a1439c3f29b6496b` |
| Original SHHS run manifest được hồ sơ tham chiếu | Expected SHA `f9cd5ebbd20f26b188b5dc13ac6e417ff8ef0fa8dcae78760cfcb27940bf58cf` — chưa đọc trực tiếp được vì nằm ngoài repository |

Các bước:

1. Recompute hai protocol SHA từ bytes và so với baseline. Giữ nguyên lịch sử, không format JSON hoặc đổi newline snapshot cho đẹp.
2. Khi người dùng cung cấp run manifest ngoài ổ E:, kiểm tra hash manifest, đọc trường protocol hash, checkpoint/prediction/input hashes và đối chiếu từng file thực có. Báo thiếu/mismatch trước, không rehash để “pass”.
3. Truy lịch sử bằng `git log --follow`, `git show <commit>:<path>` hoặc archive bản read-only vào vị trí mới. Không overwrite working snapshot. Commit date là dấu vết, không một mình chứng minh public registration hoặc rằng đã khóa trước xem test.
4. Sửa inline hash cũ: nếu mô tả file hiện tại thì cập nhật đúng hash mới; nếu là biên bản tại thời điểm cũ thì gắn version/date, giữ hash cũ với nhãn historical và link current. Ghi vì sao bytes thay đổi, ví dụ bổ sung metadata, sau đối chiếu diff thực tế.
5. Đọc chronology seeds: `experiments_v2.json` liệt kê `[42,123,2025]`, trong bài chỉ chạy 42/123 và gọi seed123 post-protocol. Tạo bảng planned/run/reported và thời điểm quyết định từ evidence. Không tự xóa 2025 hoặc chế lý do chưa chạy. Hỏi tác giả khi không xác định được.
6. `pre-specified` chỉ dùng khi có evidence quyết định trước analysis liên quan. `pre-registered` chỉ khi có registration thực; local JSON flag không đủ. Nếu chronology không xác minh được, dùng `specified in the archived protocol` và khai báo giới hạn thay vì nâng thành confirmatory.
7. Tạo `ARTIFACT_INDEX.md` hoặc phần tương đương: file, vai trò, hash, source run/commit, public/restricted, available/missing, command tái tạo, claim sử dụng. Không chép subject IDs ra public.
8. Tài liệu availability phải có repository/reviewer archive URL hoặc kế hoạch cung cấp thực, version/tag/DOI nếu tác giả cấp. Không tự điền URL chưa tồn tại; không gọi “fully reproducible” khi prediction/environment thiếu.
9. Đưa SHA dài/chronology chi tiết sang supplement/BUILD; main chỉ giữ statement chính xác và đường dẫn artifact index.
10. Sau toàn bộ revision/build, tạo manifest mới của package hiện hành; lưu ý hash manifest không bao gồm chính nó. Manifest báo cáo và manuscript có scope rõ; cập nhật dependency assets mới, không chỉ main.tex/PDF.

**Câu availability khi chưa có raw/predictions để rerun:**

> The archived protocol snapshot matches the protocol hash recorded in the available audit documentation. Code and aggregate results are provided in [verified archive]. Raw SHHS recordings are subject to NSRR access conditions. This release does not independently regenerate the historical predictions; the artifact index distinguishes verified files from externally held inputs and predictions.

Chỉ đổi `available audit documentation` thành `original run manifest` sau khi trực tiếp kiểm tra file đó. Không để `[verified archive]` chưa điền trong bản nộp.

**PASS:** identity, chronology và replay được khai báo riêng; inline hash không mâu thuẫn; mọi artifact dùng cho claim có trạng thái truy cập; không thay lịch sử để hợp văn bản.

### T15 — Xác nhận authorship, ethics, funding và AI disclosure [P0, cần tác giả]

**Vị trí:** main ENG 94–108 và 1068–1096; title page, supplement, paper VI và report cover khi phù hợp.

1. Giữ **Ngô Nhật Quân đứng đầu** theo yêu cầu rõ của người dùng. Bản ENG đang ghi `Ngo Nhat Quan`; hỏi Quân muốn dùng dạng có dấu hay transliteration nhất quán với hồ sơ học thuật. Trong submission metadata, phân biệt family name `Ngo` với given names `Nhat Quan`, không đảo thành `Quan Ngo` ngoài ý muốn.
2. First author không tự đồng nghĩa corresponding author. Hiện manuscript đánh dấu cả hai và email `23521258@gm.uit.edu.vn`; yêu cầu Quân xác nhận email và vai trò corresponding trước nộp.
3. Phạm Thái Sơn hiện đứng thứ hai. Không tự loại bỏ hoặc thêm người; kiểm tra thứ tự đã được tất cả đồng tác giả đồng ý.
4. CRediT hiện gán Sơn `Supervision, Methodology, Validation, Writing – review and editing`, Quân nhiều vai trò, và khẳng định hai tác giả đã duyệt. Đây là thông tin phải do người thật xác nhận. Report cover có giảng viên Nguyễn Hồ Duy Trí, không tự suy ra người nào phải/không phải coauthor hoặc supervisor.
5. Thu xác nhận vai trò bằng danh sách thực tế từng người đã làm; chỉ giữ statement `Both authors approved...` sau khi thực sự được duyệt. ORCID chỉ điền số thật.
6. Ethics: xác nhận secondary de-identified analysis, quyền NSRR/DUA của nhóm, yêu cầu xét duyệt/exemption tại cơ sở. Không tự bịa mã IRB, không tự kết luận “public data nên chắc chắn không cần ethics”. Nếu cần, hỏi đơn vị quản lý nghiên cứu và dùng đúng xác nhận được cấp.
7. Funding/conflict: xác nhận `no specific grant` và `no competing interests`; SHHS source grants/NSRR acknowledgement đối chiếu hướng dẫn và DUA đang áp dụng, không đoán đủ grant IDs từ bản cũ.
8. AI disclosure: nghiên cứu đã được hỗ trợ review/chỉnh/biên tập bằng AI theo trao đổi này; tác giả phải mô tả đúng các công cụ thực dùng, mục đích và trách nhiệm kiểm tra. Không tự liệt kê model/version không biết.
9. Chính sách Elsevier hiện cho phép hỗ trợ bằng AI dưới giám sát của tác giả và yêu cầu minh bạch phù hợp; công cụ không là tác giả. Hỗ trợ substantive writing/restructuring không nên bị mô tả như chỉ kiểm tra chính tả. AI tham gia phương pháp nghiên cứu phải được khai báo tại Methods ở mức tái lập được. [Chính sách AI của Elsevier](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals).
10. Draft declaration để tác giả điền/xác nhận, không tự coi đã chính xác:

> During the preparation of this manuscript, the authors used [tool/service names] to assist with [actual purposes, such as English-language editing, manuscript organization, and consistency checks]. The authors reviewed and verified the resulting text and take full responsibility for the content of the manuscript.

Nếu AI còn dùng viết code/phân tích/generating figures thì mô tả thêm đúng thực tế, không giấu trong “language editing”. Không tạo scientific figure/data bằng image generation.

11. Xác nhận có preprint/thesis/conference version hoặc manuscript đang xét nơi khác không; disclosure và cover letter phản ánh sự thật. Không tự coi bản VI nội bộ đã là prior publication.

**PASS:** có xác nhận của tác giả về tất cả statements cá nhân/pháp lý/đạo đức; không placeholder trong bản nộp; không model nào tự ký thay đồng tác giả.

### T16 — Kiểm tra BSPC author guide hiện hành và chuẩn bị bộ submission [P0]

**Giới hạn audit:** trang scope BSPC truy cập được, nhưng Guide for Authors trên ScienceDirect trả HTTP 403 trong môi trường này. Vì vậy review **chưa xác minh được** quy định riêng hiện hành về word limit, abstract format, review anonymity và loại file chấp nhận. Không dùng blog để biến các mức 250/5.000 thành luật bắt buộc.

1. Mở trực tiếp [BSPC Guide for Authors](https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control/publish/guide-for-authors) và hệ thống submission bằng tài khoản tác giả nếu cần. Lưu ngày truy cập và phần quy định liên quan vào `Reports/BSPC_SUBMISSION_REQUIREMENTS_CHECK.md`; nếu vẫn không truy cập được, nhờ tác giả cung cấp PDF/screenshot/text guide.
2. Điền checklist có nguồn cho: article type; abstract length/structured hay unstructured; keywords; manuscript length và phần được tính; review mode; title page; source/PDF/Word/LaTeX support; line/page numbering; figures/resolution; supplementary files; references/DOI; declarations; data statement; cover letter; highlights; graphical abstract bắt buộc hay tùy chọn.
3. Không tuyên bố `article` class hoặc single-column preprint tự động bị loại. Chỉ đổi template khi guideline yêu cầu hoặc có lợi rõ; không hy sinh tính đúng để làm giống mẫu.
4. Nếu double-anonymous được yêu cầu, tách title page và kiểm tra author/email/acknowledgement, metadata PDF, repository links có lộ danh tính theo policy. Nếu không phải, không xóa thông tin tác giả vô cớ. [Hướng dẫn chuẩn bị file Editorial Manager](https://www.elsevier.support/publishing/answer/how-do-i-prepare-my-files-for-submission-in-editorial-manager).
5. Hướng dẫn highlights chung của Elsevier nêu 3–5 bullet, mỗi bullet ≤85 ký tự tính cả khoảng trắng và chuẩn bị dạng editable Word document; kiểm tra yêu cầu upload cụ thể của BSPC. `highlights.txt` hiện chỉ nên coi là nguồn nội bộ cho tới khi định dạng được xác nhận. [Elsevier Highlights](https://www.elsevier.com/researcher/author/tools-and-resources/highlights).
6. Kiểm tra các bullet sau có khớp bản sửa và giới hạn ký tự, có thể dùng làm nguồn thay highlights hiện tại:
   - `Pipeline choices were evaluated with subject-wise paired comparisons.`
   - `The evaluated pipelines showed speed–memory trade-offs.`
   - `Deep-sleep under-detection recurred across pipelines on an external cohort.`
   - `Whole-record normalization did not resolve the observed deep-sleep deficit.`
   Không quảng cáo một remedy/adaptation chưa được kiểm chứng; tránh quá nhiều mã E0/E3/E6 trong highlights dành cho người đọc rộng.
7. Soạn cover letter ngắn: câu hỏi xử lý tín hiệu, empirical contribution, protocol strengths, giới hạn thật, fit với journal; không dùng “breakthrough”, “first-ever”, SOTA hoặc xác suất chấp nhận. Corresponding author duyệt trước khi gửi.
8. Tách **journal submission files** khỏi **reproducibility archive**. Không upload mặc định `note.md`, audit nội bộ, report lớp học, source path ổ E: hoặc review này như manuscript content.
9. Tạo source bundle có main, bib, supplement và mọi figure dependency với path portable; không phụ thuộc `../figures` nằm ngoài zip. Dựng thử từ một thư mục sạch của bundle.
10. Không tự chọn mô hình xuất bản có phí hoặc thanh toán APC; các lựa chọn tài chính và submission do tác giả quyết định.

**PASS:** mọi yêu cầu riêng journal có nguồn/ảnh chụp/ngày xác nhận; không còn assumption về hard word limit hoặc anonymity; bộ file tự đủ để hệ thống dựng đúng.

### T17 — Abstract, bố cục main và supplement [P0 cho nhất quán; P1 cho rút gọn]

1. Không “sửa abstract 373 từ” vì bản đó đã cũ. Mục tiêu biên tập tạm thời 220–240 từ cho đến khi xác nhận T16; đếm lại sau sửa đúng nội dung T02/T05/T07.
2. Abstract cần: empirical question; 78 EDF/180 SHHS và paired analysis; E1 là configuration replacement; E2 bundle; E6 transductive; kết quả chính có seed/estimand rõ; conclusion không causal/remedy claim.
3. Nếu thiếu chỗ, bỏ chi tiết thứ yếu chứ không bỏ caveat quan trọng. Có thể chuyển parameters/memory ra main và chỉ giữ speed trade-off; hoặc rút bớt mã model không cần để hiểu kết quả.
4. Mô tả `E3 attained the highest ...` phải scope `in the primary seed-42 campaign`; seed123 E4 cao hơn. `no stable gain` không đồng nghĩa chứng minh equivalence.
5. Main ưu tiên: cohort/protocol, design matrix, primary paired EDF results, operational trade-off, main SHHS paired results, central N3 table, boundary diagnostic ngắn, focused discussion/limitations.
6. Supplement giữ: layer/training tables; per-class/CM; secondary/post-hoc contrasts; seed123 đầy đủ; region definitions/supports; context masking và silhouette; chi tiết timing/provenance.
7. Có thể rút khoảng 800–1.200 từ lặp lại giữa Discussion/Conclusion/provenance nếu kiểm đếm cho thấy phù hợp. Đây là đề xuất biên tập, không hạn mức đã xác minh của BSPC. Không chuyển phương pháp thiết yếu ra ngoài tất cả file.
8. Supplement title hiện không trùng main: sửa cùng title hoặc `Supplementary material for: [exact main title]`; đồng bộ tác giả/version nếu cần.
9. Bỏ việc caption có `Table 1: S1...`; đặt numbering LaTeX `\renewcommand{\thetable}{S\arabic{table}}`, tương tự figure, rồi dùng labels/references. Sau thêm bảng, kiểm tra lại mọi main cross-reference, không hard-code số S11/S12 cũ.
10. Figure silhouette/context trong supplement hiện có tiếng Việt. Tạo English-only derivatives từ source plotting/data, hoặc thêm language parameter; không ghi đè figure VI đang dùng chung. Không sửa scientific charts bằng generative image editing.
11. Table sizing: tránh `\resizebox{\textwidth}{!}` phóng lớn bảng ít cột hoặc làm bảng nhiều cột quá nhỏ. Ưu tiên font/column wrapping/đưa bớt cột; main Table 6 cần kiểm tra khả năng đọc.
12. Loại bỏ các khối `\iffalse` lỗi thời sau khi bảo đảm bảng cần giữ đã chuyển đúng; Git giữ lịch sử. Không để model sau vô tình bật lại claim `calibrated posteriors`, `not noise`, `structurally predetermined` thiếu bằng chứng.
13. Cập nhật PDF metadata keywords cho khớp 6 visible keywords hoặc danh sách mới đã chốt. Header/title/date thống nhất, không mặc định đổi ngày nghiên cứu thành ngày edit.

**PASS:** main tập trung vào đóng góp thực nghiệm; supplement đọc độc lập, hoàn toàn tiếng Anh với bảng/hình đánh số đúng; không nén font để đạt số trang giả tạo.

### T18 — Đồng bộ từng claim qua toàn bộ báo cáo [P0]

**Phụ thuộc:** T01–T17 đã chốt nội dung khoa học. Đồng bộ không có nghĩa mọi tài liệu dài bằng nhau hoặc chứa cùng wording.

1. Với mỗi dòng ledger T00, liệt kê toàn bộ nơi xuất hiện; duyệt title, abstract, Methods, caption, Results, Discussion, Conclusion, highlights, note và BUILD.
2. Chạy tìm kiếm toàn repository text hiện hành, không chỉ main ENG:

```powershell
rg -n -i 'isolat|only the sequence|one component|architectural contrast|cô lập|chỉ thay' Reports docs README.md
rg -n -i 'pre.?register|đăng ký|confirmatory|pre.?specified|post.?hoc|hậu nghiệm' Reports docs configs
rg -n -i 'amplitude|biên độ|mechanism|cơ chế|remedy|khắc phục|bắt buộc|breakthrough' Reports docs
rg -n -i 'seven times|lower latency|speed.up|nhanh hơn|parameter.efficient|fully reproducible' Reports docs
rg -n '6b12447f|165d7cdf|9541e233|3333b5f1|102[.,]4|102[.,]3' Reports docs
rg -n -i 'best|highest|tốt nhất|cao nhất|zero.shot|inductive|transductive' Reports docs
```

3. Đây là danh sách ứng viên để đọc theo ngữ cảnh, **không phải lệnh replace-all**. Ví dụ `does not isolate` là câu đúng cần giữ; một snapshot lịch sử không được sửa bằng global replacement.
4. Đặc biệt sửa các file BUILD đang coi chỉ E1 là architecture-isolated và đang mặc định adaptation là thí nghiệm tiếp theo bắt buộc. Nêu các lựa chọn follow-up theo T20 và định vị tác giả.
5. Report `sec6_results.tex`: sửa statistical interpretation, “đã đăng ký” nếu chỉ local protocol, seed123 bold/ranking và oracle rounding.
6. Report `sec4_methodology.tex`: selection của CNN15 khác ResNet; E6 hypothesis không được dùng như suy luận loại trừ cơ chế.
7. Report `sec7_discussion.tex`/`sec8_conclusion.tex`: không suy ra phải có labels/must adapt; giữ cross-model error observation và phạm vi post-hoc.
8. `PAPER_REFRAMING_REVIEW_VI.md`: cập nhật trạng thái provenance hoặc ghi superseded; sửa câu lỗi ngữ pháp còn sót; không tiếp tục khuyên adaptation như điều kiện journal đã xác lập.
9. Đồng bộ bản dịch thuật ngữ: pooled/gộp; subject-mean/trung bình theo đối tượng; signed-rank; evidence secondary/thứ cấp; post-hoc/hậu nghiệm; whole-record/toàn bộ bản ghi; configuration/cấu hình; class confusion/nhầm lẫn giữa lớp; physiological transition khác reference-label boundary.
10. Nếu một phần chỉ có ở báo cáo đầy đủ, vẫn dùng cùng fact/limitation. Không ép report nội bộ 39 trang thành 12 trang chỉ vì paper ngắn hơn.
11. Sau sửa text, rebuild cả bốn PDF. Không bàn giao ENG mới nhưng VI/report PDF vẫn chứa kết luận cũ.

**PASS:** search sweep đã được đọc và phân loại; ledger không có claim hiện hành mâu thuẫn giữa các tài liệu; bản build/PDF cập nhật đồng bộ.

### T19 — Sửa bằng chứng clipping đúng cohort [P0 nếu giữ attribution cụ thể]

**Vị trí:** main ENG khoảng 719–720 và các câu “E3−E4 therefore isolates high-pass” liên quan.

1. Hiện bài viện dẫn identity audit E4/E5 trên EDF để suy ra clipping không có tác động trên SHHS. Suy luận qua cohort như vậy không đủ.
2. Có số audit SHHS riêng trong `configs/shhs_v1_protocol.json`: zero nonzero-clipping recordings trong 200 processed recordings, gồm 5/15/180 roles. Kiểm tra đúng variant, clipping thresholds, stage trước/sau filter và source manifest.
3. Nếu corroborated, dẫn **SHHS-specific numeric preprocessing audit**, nói rõ 200 processed và test subset 180; không dùng “153 EDF records” làm bằng chứng trực tiếp SHHS.
4. Nếu external data không có để kiểm tra lại, khai báo nguồn là archived numeric audit; không ghi “we independently verified all raw SHHS recordings” trong lượt revision.
5. Chỉ khi chứng minh các operations khác thực sự giống nhau trong run mới quy contrast cho operation còn lại, và vẫn nói đây là so sánh pipeline được train tương ứng, không thử mọi tham số filter.

**PASS:** mọi attribution filter/clipping dùng bằng chứng đúng cohort và đúng stage, hoặc được thu hẹp về bundled preprocessing contrast.

## 5. Có phải làm thêm thí nghiệm không? Các nhánh quyết định T20

**Khuyến nghị cho thời điểm sắp nộp:** hoàn thành P0, ưu tiên tái phân tích/khôi phục bằng chứng sẵn có và định vị related work. Chưa có căn cứ đặt “phải train baseline mới” hay “phải fine-tune SHHS” thành điều kiện mặc định để bài thuộc BSPC. Nhưng nếu giữ các claim rộng, phải cung cấp đúng thí nghiệm tương ứng.

### T20-A — Phân tích bổ sung không cần train mới [P1, ưu tiên cao]

**Mục tiêu:** củng cố độ minh bạch và độ ổn định của đóng góp thực nghiệm hiện tại.

1. Xác minh và đưa seed123 boundary diagnostic đã có vào supplement (T11).
2. Khi có prediction gốc: khôi phục per-subject distributions, median/IQR, support và paired bootstrap CI cho N3 recall/precision hoặc các chênh lệch region quan trọng. Chọn trước primary descriptive readouts của analysis mới, lưu config và công khai post-hoc.
3. Định nghĩa FPR N3 = `FP_N3 / (FP_N3 + TN_N3)`, không nhầm với `1−precision` hoặc N3→N2 rate. Nếu dùng FPR N3 specifically among N2, ghi denominator chỉ true N2 và gọi đúng conditional confusion rate.
4. Nếu muốn giải thích pooled/subject discordance, phân tích subject influence và sensitivity theo support; không chọn threshold phân nhóm sau xem outcome mà gọi confirmatory. Bảng support/paired scatter thường đủ hơn nhiều test mới.
5. Nếu có SHHS prediction từng fold, sensitivity individual models versus ensemble; không dùng folds như independent subjects.
6. Nếu có probabilities, có thể bổ sung descriptive ECE/Brier/calibration curves sau kiểm tra code/output T12. Không triển khai calibration fitting trên test.
7. Giữ seed42/123 tách; không gộp 360 subject-seed pairs như 360 người độc lập.

**Dừng khi:** thiếu predictions/probabilities. Ghi missing inputs cụ thể, không tự chế metric từ aggregate không đủ thông tin. Những việc đọc aggregate/viết Methods vẫn tiếp tục được.

### T20-B — Kiểm tra architecture-only nếu muốn giữ causal architecture claim [P2, chỉ bắt buộc khi giữ claim đó]

1. Giữ encoder, cached features, data split, loss, optimizer schedule, stopping rule, batch handling và evaluation cố định; hoặc thiết kế validation-only tuning budget tương đương cho hai sequence models.
2. Không giả định ép cùng learning rate là cách công bằng duy nhất. Chốt trước estimand: thay backbone trong một recipe cố định, hay so best configuration dưới equal search budget. Hai câu hỏi khác nhau, phải báo đúng.
3. Dùng source training/validation; không tune trên EDF outer test hoặc SHHS test. Lưu search space, budget, seeds, selection rule trước chạy mới.
4. Báo cả score và runtime/memory trong setup so sánh, uncertainty theo subject; gọi additional experiment, không thay số cũ như chưa từng tồn tại.
5. Nếu không làm, hoàn thành T02 và bỏ architecture-only claim là phương án hợp lệ cho empirical paper.

### T20-C — Baseline độc lập hoặc holdout mới [P1/P2 theo nguồn lực, không mặc định]

1. Nếu reviewer nội bộ thấy contribution quá đặc thù E0/E3, lựa chọn baseline single-channel có implementation/license/checkpoint phù hợp, chạy cùng data/split/preprocessing/evaluation, selection trên validation. Chọn dựa câu hỏi robustness, không chỉ vì leaderboard mới nhất.
2. Phân biệt external baseline chuẩn từ tác giả, reimplementation và pretrained model có khả năng overlap dữ liệu; kiểm tra data contamination.
3. Có thể ưu tiên unseen target holdout hoặc cohort khác để kiểm tra lặp lại N3 pattern, nếu mục tiêu là external generalization chứ không mới kiến trúc.
4. Một holdout mới phải thực sự chưa dùng trong lựa chọn hypothesis/pipeline; nêu sampling, eligibility, size rationale và plan trước mở labels/results. 20 reserve không tự trở thành cohort đủ lực thống kê; không tự lấy ra dùng nếu trái locked protocol.
5. Đừng thêm nhiều seed như giải pháp thay cho nhiều cohort. Seed sensitivity đo variability của training trong setup hiện có, không loại bỏ montage/age/scorer confounding.

### T20-D — Adaptation/calibration nhỏ nếu muốn tuyên bố remedy [P2 với định vị hiện tại]

1. Chỉ thực hiện sau khi tác giả chọn mục tiêu bổ sung; hiện tại có thể để future work.
2. Tách fit/calibration set, validation selection set và test; tận dụng 5 adaptation/15 validation chỉ khi đúng quyền và protocol. 180 test đã phân tích không trở thành confirmatory test mới cho hypothesis sinh sau khi xem nó; báo exploratory evaluation hoặc dùng truly unseen holdout để xác nhận.
3. Pre-specify frozen baseline, prior/logit calibration, limited fine-tuning variants, label budget, objective và stopping; không thử vô hạn rồi chỉ báo biến thể tốt.
4. Fit priors/threshold trên adaptation/validation, không dùng true class proportions của 180 test để dựng “label-free correction”. Oracle dùng test labels chỉ là diagnostic ceiling.
5. Readouts ít nhất overall/subject macro-F1, N3 recall, precision/FPR, N2 degradation, boundary/stable performance và subject-cluster CI. Tăng N3 recall bằng dự đoán mọi epoch thành N3 không phải remedy.
6. Nếu muốn claim “targeted labeling tốt hơn random”, phải thêm đối chứng matched label budget và sampling strategy, không suy từ confusion counts.
7. Báo cả thất bại và chi phí; không buộc adaptation thành công để được giữ phần diagnostic hiện tại.

## 6. Thứ tự thực thi và gói đầu ra bắt buộc

### Giai đoạn A — Xác định sự thật trước viết lại

- [ ] T00 ledger/log/baseline.
- [ ] T02–T06 design, Methods, statistics, numeric audit.
- [ ] T09/T11/T12/T19 dataset, regions, ablation, clipping.
- [ ] T14 provenance; liệt kê external inputs cần tác giả cung cấp.
- [ ] Gửi câu hỏi T15/T16 cho tác giả; tiếp tục các phần không phụ thuộc câu trả lời.

### Giai đoạn B — Chốt câu chuyện có thể chứng minh

- [ ] T01 định vị; T07–T08 inference limits.
- [ ] T10 bảng trung tâm và seed extension.
- [ ] T13 related work/citation audit.
- [ ] Chọn hoặc miễn có lý do từng nhánh T20; không âm thầm mở rộng sang GPU campaign.

### Giai đoạn C — Chỉnh toàn bộ gói tài liệu

- [ ] T17 main/supplement/abstract/layout.
- [ ] T18 đồng bộ ENG/VI/full report/notes/BUILD/docs hiện hành.
- [ ] T15 authorship/declarations được con người xác nhận.
- [ ] T16 format, portable source bundle, cover letter/highlights được xác nhận.

### Giai đoạn D — Kiểm định cuối

- [ ] Numeric consistency và test suite.
- [ ] Build sạch, visual QA tất cả trang, metadata/bibliography/cross-reference check.
- [ ] Cập nhật manifests đúng scope sau build, không trước.
- [ ] Tạo `Reports/BSPC_FINAL_READINESS_REPORT.md` theo format mục 9.
- [ ] Tác giả duyệt PDF cuối cùng và bộ file upload; model không tự submit.

Output tối thiểu của model thực thi:

1. Tất cả source và PDF ENG/VI/report đã đồng bộ hoặc chỉ rõ file nào cố ý giữ lịch sử.
2. `BSPC_REVISION_LOG.md` và `BSPC_CLAIM_EVIDENCE_LEDGER.md`.
3. `BSPC_SUBMISSION_REQUIREMENTS_CHECK.md` có nguồn hiện hành.
4. Artifact index và provenance clarification; analyses mới chỉ khi được chọn/có dữ liệu hợp lệ.
5. Manuscript/supplement source bundle portable và manifest tương ứng; draft cover letter/highlights đúng định dạng được yêu cầu.
6. `BSPC_FINAL_READINESS_REPORT.md`: PASS/BLOCKED theo từng gate, không chỉ câu “đã hoàn thành”.

## 7. Lệnh kiểm tra và tiêu chí kỹ thuật cho model thực thi

Các lệnh sau là mẫu thực thi tại root `D:\SleepTCN`, PowerShell; nếu chuyển máy Linux, đổi shell/path tương ứng nhưng giữ semantics. Đọc `--help` của scripts trước truyền tham số; không suy đoán CLI hoặc dùng script tạo artifact mới như công cụ “sửa” artifact cũ.

### 7.1. Read-only baseline và tests

```powershell
git status --short
git rev-parse HEAD
Get-FileHash -Algorithm SHA256 configs/shhs_zero_shot_v1.json
Get-FileHash -Algorithm SHA256 configs/shhs_v1_protocol.json
Get-FileHash -Algorithm SHA256 Reports/SHHS_E0_E3_E6_N3_AUDIT.json
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
```

Mốc trước sửa: 150 tests pass. Sau sửa có thể nhiều hơn nếu thêm tests; không xóa/bỏ qua test lỗi chỉ để giữ số xanh. Nếu dữ liệu nằm ngoài repository không có, integration checks phải trả `not run / missing input`, không giả PASS.

### 7.2. Kiểm tra manifest `.sha256` hiện hành, không ghi file

Chạy ở root repository. Đường dẫn trong từng manifest tương đối với thư mục chứa manifest:

```powershell
$auditManifestFiles = @(
    'Reports/REPORT_MANIFEST.sha256',
    'Reports/paper_en/SUBMISSION_MANIFEST.sha256'
)
foreach ($auditManifestFile in $auditManifestFiles) {
    $auditManifestFull = (Resolve-Path -LiteralPath $auditManifestFile).Path
    $auditManifestDir = Split-Path -Parent $auditManifestFull
    $auditEntryCount = 0
    foreach ($auditLine in Get-Content -LiteralPath $auditManifestFull -Encoding UTF8) {
        if ([string]::IsNullOrWhiteSpace($auditLine) -or $auditLine.StartsWith('#')) { continue }
        if ($auditLine -notmatch '^([0-9a-fA-F]{64})\s+\*?(.+)$') {
            throw "Malformed manifest line in $auditManifestFile"
        }
        $auditExpected = $Matches[1].ToLowerInvariant()
        $auditRelative = $Matches[2].Trim()
        $auditTarget = Join-Path $auditManifestDir $auditRelative
        if (-not (Test-Path -LiteralPath $auditTarget -PathType Leaf)) {
            throw "Missing manifest file: $auditRelative"
        }
        $auditActual = (Get-FileHash -LiteralPath $auditTarget -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($auditActual -ne $auditExpected) {
            throw "Hash mismatch: $auditRelative"
        }
        $auditEntryCount += 1
    }
    [pscustomobject]@{ Manifest = $auditManifestFile; Entries = $auditEntryCount; Status = 'PASS' }
}
```

Mốc trước sửa: 39 và 13 entries. Sau sửa số entry có thể tăng để bao phủ dependency; không lấy số lượng bằng mốc làm điều kiện duy nhất. Việc hash thay đổi ở manuscript đã sửa là dự kiến; update current manifest **sau** review diff/build/QA. Hash mismatch ở locked scientific inputs phải điều tra, không cập nhật giá trị kỳ vọng cho qua.

### 7.3. Build và kiểm tra PDF

1. Tạo thư mục build mới dưới workspace bằng `New-Item -ItemType Directory`; kiểm tra tên không trùng thư mục chứa dữ liệu cần giữ. Không xóa thư mục source để “clean build”.
2. Từ working directory `Reports/paper_en`, mẫu command cho main:

```powershell
New-Item -ItemType Directory -Path .bspc-build-review -Force
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=.bspc-build-review main.tex
bibtex .bspc-build-review/main
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=.bspc-build-review main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=.bspc-build-review main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=.bspc-build-review supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=.bspc-build-review supplement.tex
```

3. Nếu supplement mới có bibliography riêng, thêm BibTeX và hai passes theo cấu hình thực tế. Nếu cross-document refs cần thêm package, kiểm tra đường dẫn aux đúng output directory.
4. Build tương tự từ `Reports/paper` cho VI và `Reports` cho báo cáo đầy đủ. Đọc BUILD tại mỗi nơi; tránh accidentally dùng `main.aux` của tài liệu khác.
5. Kiểm tra exit code của từng lệnh, không chỉ lệnh cuối. Nếu MiKTeX bị chặn ghi cache/appdata ngoài workspace, xin đúng quyền cần thiết hoặc dùng TeX installation phù hợp; không vô hiệu hóa sandbox.
6. Tìm lỗi:

```powershell
rg -n -i 'undefined|LaTeX Error|Package .* Error|Citation .* Warning|Reference .* Warning|Overfull|Underfull' Reports -g '*.log' -g '*.blg'
```

7. Đọc cảnh báo theo ngữ cảnh; xử lý overfull thật, font thiếu, cross-reference sai; một từ “undefined” trong nội dung debug không tự là lỗi manuscript.
8. Render bằng `pdftoppm -png -r 120 <pdf> <workspace-output-prefix>` hoặc công cụ PDF tương đương; xem **mọi trang**, không chỉ trang 1. Kiểm tra bảng/hình/font/caption/margin/dấu tiếng Việt, trang trống và float xa đoạn viện dẫn.
9. Trích PDF bằng `pdftotext -layout <pdf> <workspace-text-file>`; kiểm tra title/abstract/keywords, search claim cũ và số liệu ở bản đọc thật.
10. Chỉ sau QA mới cập nhật bốn delivery PDFs. So SHA của mỗi delivery với build mới tương ứng, rồi cập nhật current manifests. Không dùng PDF cũ có timestamp mới thay cho build mới.

### 7.4. Numeric acceptance tests nên bổ sung

- Từ mỗi confusion matrix, tổng rows/columns/support đúng expected counts; classes cùng thứ tự.
- Pooled F1 tính từ sum confusion không bằng mean subject F1 nói chung; test bảo vệ hai output names khác nhau.
- Không có subject overlap giữa train/validation/test outer folds; mọi recording của một subject cùng fold; cache nguồn đúng fold.
- Prediction arrays giữa models align record, epoch, subject, true-label masks trước paired bootstrap.
- Subject-cluster bootstrap không resample mỗi recording độc lập nếu một subject có hai nights.
- E3−E2 EDF CI lower bound >0 trong khi Holm p >0.05; report text được review theo đúng hai facts này.
- Boundary masks không vượt recording/gap; near/stable/intermediate disjoint và union đủ valid epochs; persistent masks dùng đúng minimum run length.
- Oracle chỉ chuyển cell đúng hướng, không thay true-label support; pooled metric tính lại từ counts chưa rounding.
- Mọi quoted score/CI/p trong bảng generated có precision policy; latex số từ đúng JSON/key và seed.
- Source bundle có đủ `\input`, `\includegraphics`, bibliography dependencies; clean build không tìm nhầm asset ngoài bundle.

Nếu chưa có tests parse LaTeX đáng tin cậy, không viết regex phức tạp rồi coi nó thay thế review. Có thể generate numeric table fragments từ aggregate schema đã kiểm tra và review phần prose riêng.

## 8. Các câu hỏi chỉ tác giả mới trả lời được

Model thực thi nên gom một lần, không hỏi đi hỏi lại những gì repository đã cho biết:

1. Quân xác nhận dạng tên xuất bản, corresponding author và email hiện tại; thứ tự đồng tác giả và CRediT thực tế của từng người?
2. Có quyết định IRB/exemption hoặc xác nhận institutional requirements cho secondary analysis này không? Ai là người có quyền NSRR/DUA, acknowledgement chính thức đang áp dụng là gì?
3. Funding và competing interests hiện viết có đúng không? Có thesis/preprint/conference version hoặc submission song song cần khai báo không?
4. Có thể cung cấp original SHHS run manifest, locked predictions/probabilities, effective run configs và environment lock từ ổ E: hoặc một archive có quyền đọc không? Chỉ cần chia sẻ theo DUA, không tải công khai dữ liệu hạn chế.
5. Chronology thực của seed123 và seed2025 là gì: planned trước run, chạy sau khi xem seed42 hay có lý do thay đổi nào? Có protocol/tag/registration độc lập không?
6. Repository/reviewer archive dự định cung cấp ở đâu, version nào, license nào? Tác giả có cho phép public release không? Không tự thực hiện chỉ vì cần một URL.
7. Những công cụ AI nào thực sự đã dùng cho text, code, analysis hoặc figures, và ở mức nào để declaration trung thực?
8. Có bản BSPC Guide for Authors hiện hành hoặc screenshot submission requirements để xác nhận các mục đang 403 không?
9. Tác giả chọn nhánh nào: chỉ sửa + reuse existing evidence; thêm descriptive analyses từ predictions; hay mở một thí nghiệm mới có protocol riêng? Không mặc định hướng adaptation thay cho định vị empirical hiện tại.

Không trả lời được một câu không làm mọi việc khác dừng lại. Đánh dấu đúng phần phụ thuộc; tiếp tục các sửa lỗi đã đủ bằng chứng. Nhưng không xuất bản những statement cần xác nhận như thể đã được xác nhận.

## 9. Gate ra quyết định “có thể nộp”

| Gate | Điều kiện PASS | Khi chưa PASS |
|---|---|---|
| G1 — Scientific accuracy | E1/E2/control matrix đúng effective design; CI/estimand/family không mâu thuẫn; E6/causal claims đã giới hạn | Chưa nên nộp |
| G2 — Evidence completeness | Methods tái dựng được; claim chính có bảng/artifact; missing evidence được xử lý bằng bổ sung hoặc bỏ claim | Chưa nên nộp nếu claim vẫn thiếu nền |
| G3 — Provenance | Snapshot identity, chronology và replay scope rõ; không stale inline hash; availability có thật | Không ghi fully reproducible; giải quyết mâu thuẫn trước nộp |
| G4 — Editorial fit | Novelty paragraph đúng empirical framing; primary literature đối chiếu; không đòi SOTA nhưng đóng góp riêng rõ | Rủi ro desk reject khoa học; tác giả cân nhắc T20 |
| G5 — Declarations | Authors/roles/email/ethics/funding/AI/prior dissemination được xác nhận | Dừng submission, không tự điền |
| G6 — Submission package | Guide hiện hành đã đối chiếu; main/supplement/highlights/source/metadata đúng; all-page QA | Chưa upload file cuối |
| G7 — Synchronization | ENG/VI/report/docs hiện hành cùng facts; 4 PDF và manifests khớp; tests pass | Chưa bàn giao “hoàn tất” |

Mẫu final readiness report:

```text
Revision baseline commit:
Final source revision / local diff identity:
Date and reviewer:
G1 ... G7: PASS / BLOCKED / WAIVED_WITH_REASON
Verified directly from raw predictions:
Verified only from stored aggregate artifacts:
Missing external inputs:
Author confirmations received / outstanding:
Journal requirements verified, URL and access date:
Tests/build/visual QA outputs:
New exploratory analyses and their provenance:
Unchanged locked artifact hashes:
Final files to upload and files explicitly excluded:
Residual scientific limitations:
Recommendation: ready for author submission review / not yet ready, with exact reason.
```

**“Ready for author submission review” không đồng nghĩa đảm bảo accept.** Có thể đạt scientific/reporting correctness mà contribution vẫn bị reviewer đánh giá chưa đủ mới. Không gán tỷ lệ nhận 75–80% hoặc một xác suất giả chính xác.

## 10. Prompt bàn giao cho model thực thi

Sao chép đoạn dưới cùng repository và tài liệu này. Nếu chỉ cung cấp PDF mà không có code/artifacts, model phải điều chỉnh mức xác minh thay vì giả đã đọc chúng.

```text
Bạn được giao THỰC THI revision cho paper SleepTCN chuẩn bị nộp BSPC.
Hãy đọc đầy đủ Reports/BSPC_PRE_SUBMISSION_AUDIT_AND_EXECUTION_PLAN_VI.md,
mọi AGENTS.md áp dụng, main ENG, supplement, các báo cáo VI và code/config
được dẫn trong các task trước khi sửa claim tương ứng.

Mục tiêu là một empirical pipeline-evaluation và cross-cohort error-analysis
paper chính xác, minh bạch, nhất quán. KHÔNG tự biến nó thành một paper
kiến trúc SOTA hoặc adaptation method mới. Tác giả chính là Ngô Nhật Quân;
không tự sửa thứ tự tác giả hoặc gán CRediT khi chưa được xác nhận.

1. Kiểm tra working tree; giữ nguyên thay đổi của người dùng. Lập revision
   log và claim-evidence ledger như T00. Số dòng trong audit chỉ là mốc;
   tìm lại theo nội dung và báo nếu code/config hiện tại đã khác baseline.
2. Ưu tiên toàn bộ P0. Sửa đúng E1 training-recipe confounding, E2 bundle,
   CI/estimands, E6/transductive và inference limits trước khi polish prose.
3. Mỗi claim phải truy được tới đúng cohort, seed, aggregation, family,
   artifact và evidence class. Không copy số từ đoạn review này thay cho
   đọc JSON gốc; không tính ratio từ số đã rounding.
4. Đồng bộ main ENG, supplement, highlights, note, BUILD, paper VI, báo cáo
   đầy đủ và các docs hiện hành. Không blanket replace làm hỏng câu phủ
   định hoặc rewrite snapshot lịch sử. Lịch sử sai interpretation thì
   giữ artifact và tạo errata có liên kết.
5. Không chỉnh locked predictions, labels, protocols hoặc hashes kỳ vọng
   để làm validator pass. Phân biệt file integrity, provenance và rerun.
   Missing external input phải ghi BLOCKED_EXTERNAL, không giả đã xác minh.
6. Ưu tiên kiểm chứng phân tích seed123 đã có và khôi phục bảng Methods/
   results còn thiếu. Chỉ mở training/adaptation experiment mới sau khi
   tác giả chọn nhánh T20 và chốt protocol/budget. Không tune trên 180
   SHHS test subjects; phân tích mới trên cohort đã xem là hậu nghiệm.
7. Xác minh Guide BSPC hiện hành từ nguồn chính thức; audit đã gặp 403,
   nên không giả định giới hạn 5.000 từ, review anonymity hoặc template.
   Xác nhận authorship/ethics/funding/AI disclosure bằng thông tin thật.
8. Dựng và xem toàn bộ trang của cả bốn PDF. Chạy full test suite, numeric
   checks, citation/reference checks và portable-source-bundle build.
   Chỉ cập nhật current publication manifests sau build/QA; giữ lịch sử.
9. Bàn giao revision log, ledger, requirements check, artifact index,
   PDF/source package và final readiness report theo G1–G7. Nêu phần đã
   kiểm chứng từ predictions so với chỉ aggregate, và blockers còn lại.

Không tự submit, publish, ký thay đồng tác giả, public dữ liệu/subject IDs,
hoặc cam kết 100% acceptance. Không đánh dấu hoàn thành khi P0 chưa đạt
chỉ vì hết context hoặc đã chỉnh xong tiếng Anh. Tiếp tục các việc độc lập
khi một input ngoài repository còn thiếu; chỉ dừng phần thực sự phụ thuộc.
```

## 11. Bằng chứng baseline để so với bản sửa

Các hash dưới đây xác định **bản đã review**, không phải giá trị bắt bản sửa phải giữ:

| File | SHA-256 ngày audit |
|---|---|
| `Reports/paper_en/main.tex` | `c49ce26cb7b040489acacaf033b5fe18d649abd9ef745d488209427156f1c778` |
| `Reports/paper_en/supplement.tex` | `7ec4b210c56a7f795a90ffcd24c5e67099429effaf012f71b16fc2bc7b1b4847` |
| `Reports/output/pdf/SleepTCN_Scientific_Article_EN.pdf` | `9bc2f21db0c569b883cb5fa708dab02a5b9288fd22e215161dcf6d9f57bb0adb` |
| `Reports/output/pdf/SleepTCN_Supplement_EN.pdf` | `3deb984c9b428c7b441bc8686b63c0daef43ba2bb2328a58ad71314cf6ecfef8` |
| `Reports/output/pdf/SleepTCN_Scientific_Article_VI.pdf` | `1e0eadac341c3b1af5a707332b9737c40096f8501801dd0389d5b0b323ae06a3` |
| `Reports/output/pdf/SleepTCN_Gate1_8_SHHS_Report.pdf` | `bad4994ce4bc6c01dede36f5037df547e53d313f0d05053851f2b91df67920bd` |

Nếu baseline hiện tại khác những hash này, xác định diff trước áp dụng review; có thể một lỗi đã được sửa sau ngày audit. Không ghi đè bản mới về baseline cũ.
