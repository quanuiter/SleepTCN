# Phụ lục A2 cho giao thức SHHS1 holdout: phạm vi chuẩn hóa, năm seed, ensemble, EM và chi phí huấn luyện

Tệp cấu hình khóa: `configs/shhs_holdout_amendment_a2.json`. Runbook: `docs/AMENDMENT_A2_RUNBOOK_VI.md`.
Giao thức gốc: `docs/SHHS_HOLDOUT_CONFIRMATORY_PROTOCOL_V1.md` và
`configs/shhs_holdout_confirmatory_v1.json` (SHA-256 `b6653ada…`), phân tích gốc ở commit `2019b51`.
Phụ lục trước: `docs/SHHS_HOLDOUT_AMENDMENT_A1.md`.

## 1. Mục đích và phạm vi

A2 trả lời các điểm yếu được nêu khi rà soát bản thảo. Mỗi phân tích tương ứng với một điểm:

| Phân tích | Điểm yếu | Cách trả lời |
|---|---|---|
| **A** Phạm vi chuẩn hóa | E6 tính z-score trên cả bản ghi (Sleep-EDF khoảng 23 giờ, phần lớn ban ngày; SHHS1 chỉ đêm), nên thang đầu vào lệch khoảng 1,3 lần | Huấn luyện lại E6 với z-score trượt 90 phút (E6R), phạm vi thống kê giống nhau ở hai đoàn hệ, không dùng nhãn |
| **B** Số seed | Chỉ hai seed, một bậc tự do | Thêm ba seed cho E4 (dạng được khuyến nghị), tổng năm seed |
| **C** Bất đối xứng ensemble | Sleep-EDF dùng một mô hình mỗi người, SHHS1 dùng trung bình mười mô hình | Tính lại điểm SHHS1 của từng mô hình fold riêng lẻ |
| **D** EM với n = 5 | Tiên nghiệm ước lượng từ năm bản ghi | Lặp lại đúng quy trình cũ, chỉ thay dữ liệu ước lượng tiên nghiệm |
| **E** Chi phí huấn luyện | Tỷ số 10,2 lần lẫn kiến trúc với công thức huấn luyện | Tách thành số bước tối ưu, thời gian mỗi bước và phần còn lại |

Hai điểm chỉ xử lý bằng văn bản, không thêm thí nghiệm: ADAST dùng kiến trúc nền riêng (đối chứng
chỉ học nguồn của nó thấp hơn E3 0,0651), và hai đoàn hệ không có chung chuyển đạo EEG.

A2 **không** thay đổi, kiểm định lại hay diễn giải lại H1–H3, báo cáo phân tích gốc hay báo cáo A1.

## 2. Tình trạng của tập holdout

Nhãn holdout đã được dùng trong phân tích gốc và trong A1. A2 là phân tích **mô tả, sau xác nhận**,
không có kiểm định giả thuyết. Mọi arm, seed, độ dài cửa sổ, cỡ tập con, số lần bốc mẫu và quy tắc
diễn giải được khóa trước bất kỳ bước tiền xử lý, huấn luyện, suy luận hay phân tích nào của A2. Không
lựa chọn nào phụ thuộc vào kết quả A2.

## 3. Huấn luyện trên Sleep-EDF (khóa)

- **Dữ liệu và split:** Sleep-EDF SC, split mười fold theo đối tượng đã khóa
  (`data/splits/sleepedf_sc_10fold_seed42_v2.json`), fold xác thực là fold kế tiếp.
- **Runner:** `src/sleeptcn/experiment.run_experiment` với `configs/experiments_v2.json` không đổi.
  Công thức huấn luyện của mỗi arm đúng bằng công thức của experiment tương ứng (E6 hoặc E4).
- **Tương thích với split khóa:** file split không bị sửa (các lần chạy ghi lại hash của nó).
  `src/sleeptcn/dataset.POST_SPLIT_VARIANTS` cho phép biến thể A2; bước kiểm tra tiền xử lý A2 yêu cầu mỗi
  bản ghi Sleep-EDF mới có cùng khóa bản ghi, nhãn và chỉ số epoch gốc với bản `filtered_v2` tương ứng.
- **Arm:**

| Arm | Experiment | Biến thể dữ liệu | Seed | Số lần train |
|---|---|---|---|---|
| E6R | E6 | `filtered_zscore_rolling90_v3` | 42, 123 | 20 |
| E4 | E4 | `bandpass_v2` | 2025, 2027, 2029 | 30 |

- **E6R:** lọc thông dải 0,5–30 Hz, cắt ±800 µV, rồi z-score trượt có tâm với cửa sổ 90 phút (cắt ngắn
  ở hai đầu bản ghi), sàn độ lệch chuẩn 1 µV. Thống kê chỉ lấy từ tín hiệu đã lọc và cắt, không dùng
  nhãn. Cửa sổ 90 phút xấp xỉ một chu kỳ ngủ, nên mỗi cửa sổ thường chứa cả ngủ nông lẫn ngủ sâu. E6R
  dùng đúng hai seed của E6 lưu trữ để so từng seed.
- **Seed E4:** 2025 đã được liệt kê trong `configs/experiments_v2.json` nhưng chưa từng huấn luyện; 2027
  và 2029 được cố định ở đây. Cùng seed 42 và 123 lưu trữ, E4 có năm seed.
- **Thứ tự chạy:** E6R s42, E6R s123, E4 s2025, E4 s2027, E4 s2029; mỗi cặp đủ mười fold. Có thể chạy
  nhiều tiến trình song song trên một GPU; thiết lập tất định không đổi.
- **Kiểm tra tái lập:** trước mọi lần train A2, huấn luyện lại E4 fold 0 seed 123 bằng runner A2 và so
  với lần chạy lưu trữ ở commit `8af1d12` (số epoch, epoch tốt nhất, macro-F1 xác thực, dự đoán xác
  thực). Kết quả được báo cáo dù giống hệt hay khác; A2 vẫn tiếp tục vì GPU và phiên bản thư viện khác
  chiến dịch gốc.
- **Tập test Sleep-EDF:** chỉ mở **một lần**, sau khi mọi lần train A2 đã xong và qua kiểm tra lần chạy,
  với checkpoint tốt nhất đã khóa (dự đoán ngoài fold).
- **Lần chạy lỗi:** chạy lại hoặc tiếp tục với **cùng seed**; không thay, bỏ hay thêm seed.

## 4. Tiền xử lý và suy luận trên SHHS1 holdout

- **Đoàn hệ:** 500 người holdout của phân tích gốc.
- **Biến thể mới:** `filtered_zscore_rolling90_v3`, dùng cùng tệp EDF/XML, kênh, lấy mẫu lại, ánh xạ
  nhãn và cửa sổ đánh giá như manifest gốc (`f0be16e6…`); ghi vào manifest A2 riêng ở
  `data/processed/shhs_holdout_a2`.
- **Biến thể có sẵn:** các arm E4 dùng lại đầu vào `bandpass_v2` đã kiểm tra hash của manifest gốc.
- **Ensemble:** trung bình số học xác suất softmax của mười fold (cộng dồn float64, lưu float32), rồi
  argmax; giữ mọi tệp fold.
- **Thiết bị:** CPU, hoặc CUDA nếu trước đó code suy luận A2 tái tạo xác suất fold CPU đã lưu của
  E4_s42 cho hai bản ghi ở fold 0 và 9 trong sai số 1e-5 với argmax giống hệt.

## 5. Phân tích (khóa)

Bootstrap: 10 000 lần lấy mẫu lại theo người, bắt cặp, seed 2043. Không kiểm định giả thuyết.

### A. Phạm vi chuẩn hóa

- Đối chiếu ở từng seed (42, 123): E4 − E6R, E3 − E6R, E6R − E6.
- Chỉ số: macro-F1 trung bình theo người, macro-F1 gộp, độ thu hồi N3, tỷ lệ N3→N2.
- Thang đầu vào: trung vị và IQR độ lệch chuẩn trong cửa sổ đánh giá của đầu vào E6 và E6R, trên
  Sleep-EDF và trên holdout.
- Sleep-EDF: macro-F1 gộp ngoài fold của E6R so với E3, E4 và E6 cùng seed.
- **Quy tắc diễn giải:** bản thảo chỉ giữ câu "giữ biên độ đã hiệu chuẩn chuyển quần thể tốt hơn
  z-score" nếu E4 − E6R có khoảng bootstrap nằm trên 0 **ở cả hai seed**. Nếu không, câu đó bị giới hạn
  lại thành "z-score trên toàn bản ghi như đã triển khai", và kết quả E6R được báo cáo như bằng chứng
  rằng thiếu hụt của E6 phụ thuộc vào phạm vi chuẩn hóa.

### B. Năm seed huấn luyện của E4

- Seed 42, 123, 2025, 2027, 2029.
- Mỗi seed: macro-F1 trung bình theo người trên holdout và macro-F1 gộp ngoài fold trên Sleep-EDF. Trung
  bình, độ lệch chuẩn mẫu và khoảng giá trị qua năm seed. Cả mười chênh lệch từng cặp seed trên holdout,
  kèm khoảng bootstrap bắt cặp.
- Đối chiếu cấu hình: E4 ở từng seed trừ E0 ở seed 42 và 123 (mười cặp), và E4 ở từng seed trừ E2 ở seed
  42 và 123 (mười cặp), kèm khoảng bootstrap; đặt cạnh khoảng của mười chênh lệch giữa seed.
- **Quy tắc diễn giải:** bản thảo thay khoảng chênh lệch giữa hai seed bằng tóm tắt năm seed. Chỉ nói đối
  chiếu cấu hình "vượt biến thiên huấn luyện" nếu **mọi** cặp E4 − E0 và E4 − E2 lớn hơn chênh lệch tuyệt
  đối lớn nhất giữa các seed của E4.

### C. Ensemble so với mô hình đơn

- Mọi arm zero-shot của phân tích gốc (E0, E2, E3, E4, E6 ở seed 42 và 123) và mọi arm A2.
- Từng mô hình fold riêng lẻ trên holdout: macro-F1 trung bình theo người, macro-F1 gộp, độ thu hồi N3, tỷ
  lệ N3→N2; trung bình và khoảng qua mười mô hình; ensemble trừ trung bình mô hình đơn (macro-F1 trung bình
  theo người) kèm khoảng bootstrap bắt cặp.
- Mức sụt giữa hai đoàn hệ: với E3 và E4 ở seed 42 và 123, macro-F1 gộp ngoài fold trên Sleep-EDF trừ
  trung bình macro-F1 gộp của mô hình đơn trên holdout, đặt cạnh mức sụt tính bằng ensemble.

### D. Cỡ mẫu của EM

- **Mô hình và quy trình:** đúng mười mô hình E3 seed 123 và đúng quy trình của thí nghiệm hiệu chuẩn
  gốc. Với mỗi fold: hiệu chuẩn xác suất holdout bằng nhiệt độ đã lưu của fold đó
  (`runs/teacher_revision_cpu_20261003/calibration_10fold_seed123/fits`), ước lượng tiên nghiệm đích bằng
  `estimate_target_prior` (khởi tạo và so với tiên nghiệm nguồn đã lưu của fold; dung sai 1e-8, tối đa
  1000 vòng, sàn 1e-12), chỉnh bằng `adjust_prior`, rồi lấy trung bình mười fold trước argmax. **Chỉ dữ
  liệu dùng để ước lượng tiên nghiệm thay đổi.**
- **Xác suất holdout:** xác suất từng fold đã lưu của arm E3_s123, các epoch trong cửa sổ đánh giá.
- **Dữ liệu ước lượng tiên nghiệm:**
  1. năm bản ghi thích nghi gốc: dùng nguyên tiên nghiệm đích đã lưu;
  2. tập con ngẫu nhiên n ∈ {5, 20, 100} bản ghi holdout, mỗi mức 200 lần bốc, `default_rng(2044)`;
  3. đúng 20 000 epoch của mẫu AdaBN trong A1 (40 epoch mỗi bản ghi holdout), ghép theo khóa bản ghi và
     chỉ số epoch;
  4. cả 500 bản ghi holdout;
  5. từng bản ghi tự ước lượng tiên nghiệm và chỉ áp cho chính nó.
- **Kết quả:** tiên nghiệm N3 ước lượng (trung bình qua fold) so với tỷ lệ N3 quan sát trên holdout; sau
  khi chỉnh toàn bộ 500 bản ghi: tỷ lệ epoch dự đoán là N3, độ thu hồi N3 gộp, macro-F1 gộp và macro-F1
  trung bình theo người. Với tập con ngẫu nhiên: trung vị và bách phân vị 2,5–97,5 qua các lần bốc.
- **Quy tắc diễn giải:** nếu tiên nghiệm ước lượng từ mẫu AdaBN và từ cả 500 bản ghi vẫn thấp hơn một nửa
  tỷ lệ N3 quan sát, và làm độ thu hồi N3 thấp hơn ensemble chỉ hiệu chuẩn nhiệt độ, thì sự sụp đổ được quy
  cho việc giả định dịch chuyển nhãn không đúng khi có dịch chuyển đặc trưng, không phải do mẫu năm bản ghi.
  Nếu tiên nghiệm tiến gần tỷ lệ quan sát và không làm giảm độ thu hồi N3, sự sụp đổ được quy cho mẫu thích
  nghi quá nhỏ.

### E. Chi phí huấn luyện

- **Số bước:** số epoch hoàn thành và số bước tối ưu của mọi giai đoạn huấn luyện ở mọi fold trong chiến
  dịch seed 123 lưu trữ (E0, E2, E3, E4, E6), đọc từ bản ghi tiến trình trong checkpoint ở commit
  `8af1d12`; số bước mỗi epoch tính từ kích thước tập huấn luyện và batch size của giai đoạn đó.
- **Thời gian mỗi bước:** trung vị thời gian forward + backward + optimizer cho CNN 15 nhánh (batch 64),
  ResNet-1D (batch 64), BiLSTM (batch 4 chuỗi) và TCN (batch 8 chuỗi), đầu vào tổng hợp đúng kích thước
  huấn luyện, 20 lượt khởi động và 200 lượt đo, chạy một mình trên GPU thuê.
- **Kết quả:** với mỗi cấu hình: tổng số bước, số bước × thời gian mỗi bước, và phần còn lại của thời gian
  thực đo (xác thực, nạp dữ liệu, lưu checkpoint). Vì GPU thuê có thể khác V100 gốc, các cấu hình được so
  bằng tỷ số.
- **Quy tắc diễn giải:** bản thảo nói rõ bao nhiêu phần của tỷ số thời gian E0/E3 đến từ số bước tối ưu và
  bao nhiêu từ thời gian mỗi bước, và không còn trình bày tỷ số đó như một thuộc tính của kiến trúc.

## 6. Giới hạn đã biết

- E6R loại bỏ cả những biến đổi biên độ chậm hơn khoảng 90 phút trong đêm, nên không chỉ khớp phạm vi mà
  còn thay đổi một phần thông tin biên độ tương đối giữa các giai đoạn. Kết quả A được diễn giải đúng
  trong phạm vi này.
- Năm seed vẫn là ít để ước lượng đuôi của phân phối; B mô tả độ phân tán, không ước lượng phân phối.
- D chỉ dùng E3 seed 123, mô hình của thí nghiệm gốc.

## 7. Điều cấm

- Không thêm, bỏ hay thay arm, seed, fold, cỡ tập con hay số lần bốc sau khi khóa.
- Không mở tập test Sleep-EDF trước khi mọi lần train A2 xong và qua kiểm tra; không mở lần thứ hai.
- Không chọn checkpoint, seed hay thiết lập dựa trên kết quả holdout.
- Mọi lần chạy, kể cả lần cho kết quả bất lợi, đều được báo cáo.
- Sai lệch so với phụ lục này được ghi lại cùng lý do và ảnh hưởng.

## 8. Trình tự

1. Khóa `configs/shhs_holdout_amendment_a2.json` (chỉ đổi trường `status` thành
   `locked_before_a2_training`), commit và push.
2. Tiền xử lý Sleep-EDF và SHHS1 holdout cho biến thể `filtered_zscore_rolling90_v3`.
3. Phân tích C (phần các arm gốc), D và đọc số bước cho E; không cần GPU.
4. Trên GPU thuê: kiểm tra tái lập, 50 lần train, mở tập test Sleep-EDF một lần, đo thời gian mỗi bước.
5. Suy luận holdout cho năm cặp arm–seed mới.
6. Phân tích A, B, phần còn lại của C và E; cập nhật bản thảo theo các quy tắc diễn giải ở mục 5.
