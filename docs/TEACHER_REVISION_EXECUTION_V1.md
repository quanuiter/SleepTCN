# Giao thức thực hiện bổ sung sau góp ý thầy Trí - v1

> Cập nhật thực thi 03-10-2026: đã khôi phục đủ E3/E4 hai seed và mười fold;
> E4 seed 42 cùng calibration/EM đủ mười fold đã hoàn tất và kiểm chứng. Weighted
> và ADAST hiện vẫn là so sánh một fold. Xem
> `Reports/TEACHER_REVISION_STATUS_20261003_VI.md` và
> `Reports/TEACHER_RESPONSE_20261003_VI.md` cho trạng thái hiện hành.
> Phần mô tả CLI một fold và checkpoint thiếu bên dưới là hồ sơ phiên bản v1;
> runner mười fold hiện là `scripts/run_revision_calibration_10fold.py` với cấu hình
> riêng `configs/teacher_revision_calibration_10fold_v1.json`. Không sửa cấu hình
> pilot hoặc hash chạy cũ để ghi đè lịch sử.

Ngày 01-10-2026. Đây là giao thức **thí điểm sau quan sát**, không phải sửa hồi tố giao thức v2 hay đăng ký trước kết quả cũ. Báo cáo tiến độ thực tế: `Reports/TEACHER_REVISION_IMPLEMENTATION_20261001_VI.md`.

Cập nhật thực thi ngày 02-10-2026: `Reports/CPU_INTERVENTION_PILOTS_20261001_VI.md` thay thế trạng thái tiến độ của lượt CPU đầu trong `Reports/CPU_EXECUTION_RESULTS_20261001_VI.md`. Đã khôi phục và xác minh checkpoint đầy đủ E0/E3/E6 fold 0 seed 123 ở demo assets; hoàn tất calibration/EM và cặp TCN unweighted/weighted trên 180 người SHHS, có kiểm chứng lại. Vẫn thiếu checkpoint E4 và các fold còn lại. Các pilot là phần bổ sung sau quan sát, không sửa hồi tố giao thức/ensemble lịch sử.

## Phạm vi và trình tự

1. Đối chiếu hiện vật E3/E4: cùng đối tượng, epoch, nhãn, đơn vị, variant, seed và mã băm checkpoint. Kiểm tra `100*x_E3 ~= x_E4` trên các cửa sổ đã lưu; kiểm tra này không thay thế audit toàn bộ tín hiệu liên tục trước cắt.
2. Khôi phục checkpoint E4 seed 42 và chạy inference trên SHHS bằng giao thức cũ. Không cần huấn luyện lại nếu đủ checkpoint đúng nguồn gốc. Kết quả là phần mở rộng, không phải kết quả chính mới được khóa trước. Không thay E3 bằng E4 trong giao thức lịch sử.
3. Thí điểm ba nhánh đầu ra: nguyên gốc, temperature calibration, temperature calibration + EM prior adjustment. Đã chạy và kiểm chứng fold 0 trên dữ liệu thật: EM không cải thiện N3 và làm giảm macro-F1. Chưa mở rộng đủ mười fold.
4. Nếu thử weighted TCN: giữ encoder/cache cố định, tính trọng số chỉ từ tập train, huấn luyện lại cả đối chứng unweighted và weighted với cùng revision/seed. Không trộn checkpoint cũ và mới để gọi là đối chiếu được kiểm soát hoàn toàn.
5. Thêm đối chứng ngoài gồm source-only và UDA của cùng backbone. ADAST là ứng viên để tích hợp, không được gọi là phương pháp tốt nhất hoặc triển khai hoàn tất. Chốt giao thức trước chạy đầy đủ; không dùng điểm test để điều chỉnh.

## Calibration / EM: lựa chọn đã triển khai cho pilot

- Mỗi fold có calibrator riêng, fit NLL trên validation Sleep-EDF tương ứng; không dùng outer test. Validation từng dùng cho early stopping, phải công khai việc dùng lại. Không fit một calibrator ensemble từ dự đoán mười mô hình trên toàn bộ 78 người nguồn.
- Temperature scaling đơn tham số, bounds [0.05,20], xác suất floor 1e-12. Đây là baseline đơn giản, **không phải** bản tái lập bias-corrected calibration của [Alexandari và cộng sự, ICML 2020](https://proceedings.mlr.press/v119/alexandari20a.html).
- Prior nguồn là trung bình xác suất đã calibration trên validation nguồn của fold, có trọng số theo epoch. Không dùng tỷ lệ nhãn SHHS; không chia đầu ra TCN cho trọng số lớp của encoder.
- EM fit trên xác suất adaptation đích không nhãn, khởi tạo prior nguồn, tối đa 1.000 vòng, dừng khi max thay đổi prior < 1e-8. Không hội tụ thì CLI dừng, không xuất kết quả. Đơn vị fit là epoch, không phải mỗi người một trọng số bằng nhau; đây là lựa chọn pilot cần ghi rõ.
- Áp dụng từng calibrator/prior lên xác suất inference của cùng fold, rồi lấy trung bình xác suất mười fold bằng float64 và argmax. Dùng đủ mười fold, không chọn fold theo kết quả đích. CLI hiện **chỉ** xử lý một fold; công cụ ensemble/evaluation chiến dịch thực còn cần nối với hiện vật nguồn gốc đã kiểm tra.
- Có thể dùng tập adaptation riêng, hoặc toàn bộ bản ghi inference không nhãn để ước lượng prior theo chế độ transductive; phải chốt chế độ trước chạy. Không gọi tập 180 người đã xem kết quả là holdout mới.
- Cửa sổ adaptation phải được chọn không cần nhãn đích. Các NPZ SHHS hiện tại được cắt theo first/last true sleep: **không đủ điều kiện này**, dù có thể tách bỏ trường `y`. Nếu inference toàn bản ghi thay đổi ngữ cảnh TCN, chạy lại baseline tương ứng. Đánh giá trên benchmark window cũ chỉ được làm sau inference và phải mô tả tách biệt.
- Hội tụ EM không chứng minh giả định label shift đúng; montage, scoring và tuổi vẫn có thể làm thay đổi phân bố tín hiệu theo lớp. Không cam kết N3 sẽ tăng hay macro-F1 sẽ được giữ nguyên.

## Hợp đồng đầu vào CLI

Chạy trên CPU:

```powershell
.venv/Scripts/python.exe scripts/run_calibration_pilot.py --manifest <manifest.json> --output <pilot_fold_00.npz>
```

Manifest JSON schema 1, mẫu cấu trúc (thay các giá trị mô tả bằng metadata thật trước chạy):

```json
{
  "schema_version": 1,
  "class_order": ["W", "N1", "N2", "N3", "REM"],
  "outer_fold": 0,
  "checkpoint_sha256": "<64 lowercase hex characters>",
  "source_split_sha256": "<64 lowercase hex characters>",
  "source_role": "validation",
  "adaptation_window_policy": "label_independent",
  "source_validation": "source_validation.npz",
  "target_adaptation": "target_adaptation.npz",
  "target_inference": "target_inference.npz"
}
```

Đường dẫn tương đối tính từ thư mục manifest. Source NPZ phải có đúng `probabilities (N,5)`, `labels (N,)` và `epoch_id (N,)` là chuỗi duy nhất. Lọc epoch unknown/padding ở nguồn trước đóng gói, có audit riêng. Target NPZ chỉ có `probabilities` và `epoch_id`; CLI từ chối file chứa nhãn hoặc khai báo cửa sổ phụ thuộc nhãn. ID epoch phải xác định duy nhất bản ghi + vị trí thời gian và phải khớp giữa các fold trước ensemble.

CLI xác minh cấu trúc, không tự chứng minh lời khai trong manifest về role hay cách chọn cửa sổ. Cần đối chiếu split/checkpoint/window provenance độc lập. Không chuyển NPZ benchmark cũ thành đầu vào đủ chuẩn chỉ bằng đổi tên trường. Đầu ra gồm ba mảng xác suất, epoch IDs và metadata/mã băm input/code; không có metric test. File đã tồn tại không bị ghi đè. Các tệp cấp người tham gia giữ ở kho hạn chế truy cập, không thêm vào ZIP nộp báo.

## Can thiệp loss và đánh giá N3

Utility `source_class_weights` đã chuẩn bị trọng số N/(5*n_c), bỏ nhãn -1 và từ chối lớp train vắng mặt. Historical runner vẫn dùng loss cũ. Runner pilot riêng `scripts/run_weighted_tcn_cpu_pilot.py` đã hoàn tất cặp fold 0 và lưu loss weights/config/code hash. Công cụ recovery chỉ hoàn tất suy luận bị gián đoạn, không thay checkpoint. Weighted cải thiện N3 nhưng có đánh đổi về N2 và các metric tổng thể; không tự động thay mô hình cuối.

Primary contrast của pilot: calibrated+EM so với calibrated-only. Nhánh raw cho biết ảnh hưởng của calibration. Báo cáo subject-mean macro-F1 và N3 F1/recall/precision, N2 recall, true-N2→predicted-N3, cùng ma trận nhầm lẫn đầy đủ. Dùng bootstrap bắt cặp theo đối tượng, 10.000 lần, seed 2031; xem CI pilot là mô tả sau quan sát. Không hứa “duy trì hiệu năng” hay non-inferiority khi chưa chốt margin có cơ sở. Nếu mở kiểm định nhiều metric/nhánh, khóa family Holm trước đọc kết quả thay vì chọn p có lợi.

Subject không có nhãn N3: recall N3 không xác định; ghi số người có N3 và phân tích có điều kiện, đồng thời giữ mọi người trong macro-F1 5 lớp (zero_division=0) và đánh giá nhầm N2→N3. Không gán recall=0 cho người không có N3 hoặc loại họ khỏi toàn bộ đánh giá. Pooled metrics báo cáo riêng. Hai seed không tạo thành 360 người độc lập; fold không phải 10 cohort độc lập.

## Đối chứng UDA ngoài

[Kho ADAST của tác giả](https://github.com/emadeldeen24/ADAST) đã được pin tại `e0fb503544ddd38f71027c09e3401b900f3dabc3`, license Apache-2.0. Đã audit dataloader/self-training, chạy kiểm tra model/loss CPU và triển khai adapter/source-only theo `configs/teacher_revision_adast_fold0_pilot_v1.json`. Giữ model class, similarity penalty, lịch loss và quy tắc logits upstream; bỏ đọc/log nhãn đích trong huấn luyện. Hai nhánh cùng khởi tạo, mẫu nguồn và 1.140 updates, chọn checkpoint cuối theo ngân sách cố định. Preprocessing được đồng nhất theo dự án nên đây là pilot ADAST trong giao thức chung, không phải tái lập nguyên bản paper. Trạng thái và kết quả thực tế xem báo cáo pilot; không suy từ việc có adapter rằng baseline đã được xác nhận.

## Tài nguyên và chặn thực nghiệm

Full EDF processed và SHHS processed/archived predictions hiện truy cập được. Checkpoint đầy đủ tại đường dẫn inventory cũ không còn ở đó, nhưng đã tìm được bản sao E0/E3/E6 fold 0 seed 123 khớp hash ở demo assets và tạo xác suất validation từ chúng. PyTorch hiện tại là CPU-only. Calibration/EM và weighted TCN fold 0 đã hoàn tất trên CPU; không coi GPU là điều kiện bắt buộc cho các pilot này. Mở rộng nhiều fold/seed, huấn luyện lại encoder hoặc chạy campaign UDA lớn cần benchmark ngân sách riêng; thiếu checkpoint E4 là thiếu hiện vật, không phải chỉ thiếu GPU.

Không bypass kiểm tra clean worktree/code hash của các runner cũ. Seed fix thay đổi runner hash: chạy chiến dịch mới trong revision/workspace riêng sau khi chủ dự án quyết định lưu revision; không sửa manifest/hash checkpoint cũ cho khớp code mới.
