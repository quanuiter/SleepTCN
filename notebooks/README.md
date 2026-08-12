# Các notebook hiện có

```text
00_kiem_tra_moi_truong.ipynb       # kiểm tra workspace và môi trường
10_kiem_thu_hop_dong_cpu.ipynb    # smoke CPU, không phải đánh giá chất lượng
20_chay_thu_gpu.ipynb              # smoke GPU có kiểm soát
```

Smoke GPU đã hoàn tất. Trên Docker mới, chỉ cần chạy `00` nếu muốn xem/ghi báo cáo môi trường;
không cần chạy lại `20` cho mỗi fold. Notebook không thay thế sáu full run đang hoạt động
E0/E1/E2/E3/E4/E6 và không được dùng để mở test. Có thể bỏ qua notebook hoàn toàn và chạy CLI
trong `docs/DOCKER_GPU_RUNBOOK.md`; đây là cách dễ ghi log và tiếp tục bằng `--resume` hơn.

`10_kiem_thu_hop_dong_cpu.ipynb` chỉ dành cho kiểm tra CPU, không cần chạy lại trên GPU.

Notebook chỉ điều phối và trình bày. Mã dùng chung nằm trong `src/sleeptcn`; runner chính thức là
`scripts/run_experiment.py`, còn `scripts/validate_run_artifacts.py` phải chạy sau mỗi run.
