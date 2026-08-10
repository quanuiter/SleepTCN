# Các notebook hiện có

```text
00_kiem_tra_moi_truong.ipynb       # kiểm tra workspace và môi trường
10_kiem_thu_hop_dong_cpu.ipynb    # smoke CPU, không phải đánh giá chất lượng
20_chay_thu_gpu.ipynb              # smoke GPU có kiểm soát
```

Trên Docker, chỉ cần chạy `00` nếu muốn xem/ghi báo cáo môi trường bằng notebook và chạy `20`
cho smoke GPU. Hai notebook này không thay thế việc chạy đủ E0--E6 và không mở tập test.
Có thể bỏ qua notebook hoàn toàn và chạy CLI trong `docs/GPU_DEPLOYMENT.md`; đây là cách dễ
ghi log và tiếp tục bằng `--resume` hơn.

`10_kiem_thu_hop_dong_cpu.ipynb` chỉ dành cho kiểm tra CPU, không cần chạy lại trên GPU.

Notebook chỉ điều phối và trình bày. Mã dùng chung nằm trong `src/sleeptcn`; runner chính thức là
`scripts/run_experiment.py`, còn `scripts/validate_run_artifacts.py` phải chạy sau mỗi run.
