# Dữ liệu

Dữ liệu EDF gốc không được sao chép vào Git.

- Windows: `E:/research/Dataset/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette`
- Máy GPU: gắn hoặc sao chép tới `/workspace/datasets/sleep-edfx/1.0.0/sleep-cassette`

Các thư mục được tạo khi chạy:

```text
data/manifests/
data/processed/paper_raw_v1/
data/processed/filtered_v2/
data/processed/filtered_legacy_v1/
```

Không chỉnh sửa dữ liệu nguồn. Mỗi dữ liệu đã xử lý phải có manifest và mã băm riêng.

