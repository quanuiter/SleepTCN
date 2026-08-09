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

Z-score dùng trung bình và độ lệch chuẩn của toàn bộ tín hiệu bản ghi sau lọc/cắt, không dùng nhãn
và không dùng thống kê của đối tượng khác. Cách này phù hợp phân tích toàn đêm offline; không được
tự động suy rộng kết luận sang suy luận thời gian thực.

Mỗi NPZ z-score phải lưu `normalization_scope`, `normalization_mean` và `normalization_std`.
Mọi đầu ra được đối chiếu mã băm EDF nguồn và được ghi nguyên tử.
