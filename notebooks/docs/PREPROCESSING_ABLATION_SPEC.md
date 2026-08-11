# Đặc tả tiền xử lý loại bỏ thành phần v2

Mọi biến thể dùng cùng kênh EEG Fpz-Cz, 100 Hz, epoch 30 giây, ánh xạ nhãn và cửa sổ ngủ của v1.
Nhãn và `original_epoch_index` phải giống tuyệt đối giữa năm biến thể.

| Biến thể | Lọc dải | Cắt biên độ | Đổi thang |
|---|---|---|---|
| `paper_raw_v1` | Không | Không | Không |
| `bandpass_v2` | Butterworth 0,5–30 Hz | Không | Không |
| `bandpass_clip_v2` | Như trên | ±800 µV | Không |
| `filtered_v2` | Như trên | ±800 µV | Chia hằng số 100 |
| `filtered_zscore_v2` | Như trên | ±800 µV | Z-score riêng từng bản ghi |

Ghi chú thực thi 2026-08-11: kiểm tra bitwise toàn bộ 153 cặp cho thấy `bandpass_v2` và
`bandpass_clip_v2` có dữ liệu khoa học giống hệt nhau; `clip_fraction` bằng 0 cho mọi bản ghi.
Do đó không chạy `bandpass_clip_v2`/E5 ở các fold 01--09. Xem
`data/manifests/bandpass_clip_identity_v2.json` để có chứng cứ tái lập.

Z-score dùng trung bình và độ lệch chuẩn của toàn bộ tín hiệu bản ghi sau lọc/cắt, không dùng nhãn
và không dùng thống kê của đối tượng khác. Cách này phù hợp phân tích toàn đêm offline; không được
tự động suy rộng kết luận sang suy luận thời gian thực.

Mỗi NPZ z-score phải lưu `normalization_scope`, `normalization_mean` và `normalization_std`.
Mọi đầu ra được đối chiếu mã băm EDF nguồn và được ghi nguyên tử.
