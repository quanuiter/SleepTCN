# Phạm vi sử dụng notebook

```text
00_kiem_tra_moi_truong.ipynb       # kiểm tra workspace và môi trường
10_kiem_thu_hop_dong_cpu.ipynb    # smoke CPU, không phải đánh giá chất lượng
20_chay_thu_gpu.ipynb              # smoke GPU có kiểm soát
```

Toàn bộ smoke GPU, 10 fold, hai chiến dịch seed 42/123, Gate 1--8 và test đã hoàn tất. Ba notebook
trên hiện chỉ phục vụ kiểm tra môi trường và minh họa hợp đồng CPU/GPU; chúng không phải nguồn kết
quả cuối và không cần được chạy lại để tái tạo các bảng hiện hành.

`10_kiem_thu_hop_dong_cpu.ipynb` chỉ dành cho kiểm tra CPU, không cần chạy lại trên GPU.

Notebook chỉ đảm nhiệm vai trò điều phối và trình bày. Mã dùng chung nằm trong `src/sleeptcn`; kết quả
hiện hành được truy nguyên từ artifact trong `runs/v2`, còn trạng thái tổng hợp được ghi tại
`docs/STATUS_V2.md`.
