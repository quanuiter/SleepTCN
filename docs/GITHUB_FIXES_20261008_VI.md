# Các sửa chữa sau review GitHub — 08-10-2026

## Đã xử lý

| Vấn đề | Thay đổi và kiểm tra |
|---|---|
| Test import payload riêng khi collection | Ba test module ADAST lấy adapter/metrics/training core từ mã nguồn cùng bytes trong repo. 409 test được thu thập không lỗi trên checkout chỉ chứa nguồn và bằng chứng tổng hợp công khai. |
| Unit test lẫn dữ liệu thật/training | Giữ nguyên các test chuyên biệt, đánh dấu và yêu cầu bật riêng. Suite mặc định chặn optimizer `.step()`; 377 passed, 32 skipped theo chính sách, không phải 409 passed. |
| Lỗi progress Windows | Thêm `sleeptcn.runtime_io.ReportingBudget`: tệp tạm riêng, retry hữu hạn cho sharing lock, giữ snapshot pending và cảnh báo nếu vẫn khóa. Lỗi I/O khác vẫn được báo; checkpoint/kết quả không dùng cơ chế bỏ qua lỗi này. |
| Cache hash tính lại dù có cache | `ImmutableHashCache` chỉ đọc lại khi cần, kiểm tra expected digest mọi lần, phát hiện thay đổi stat và thay đổi trong lúc đọc. Verifier cuối vẫn đọc/hash lại độc lập. Chưa gán số phần trăm tăng tốc. |
| Không được thay bytes code lịch sử | Có runner `run_adast_fullsource_local_evaluation_v2.py` riêng, output riêng và bind helper mới vào proof. Runner cũ, training core và helper đã dùng trong thí nghiệm không bị sửa. |
| Build thiếu bản dịch Việt | `build_public_delivery.py` và wrapper PowerShell build đủ Anh, Việt BSPC và phụ lục từ staging sạch; Việt dùng XeLaTeX/Times New Roman. Source ZIP Anh được biên dịch lại riêng. |
| Manifest phụ thuộc tmp và tự cập nhật bằng chứng | Public evidence manifest riêng được kiểm tra trước/sau build, không tự rebaseline. Manifest bàn giao không còn đường dẫn tmp/private. Các manifest cũ được giữ trong backup local của lượt build đầu. |
| EOL phá hash sau Git | Attributes giữ bytes cho nguồn/bằng chứng có hash. Hai tệp lock môi trường được phục hồi LF đúng checksum lịch sử vốn có, không thay phiên bản dependency. Kiểm tra checkout thực với cả autocrlf=true và false đạt. |
| Hướng dẫn cũ | Cập nhật README, hướng dẫn test/build, phân biệt môi trường gốc PyTorch 2.5.1 và ADAST Colab 2.11.0+cu130; thêm workflow CI source-only. CI đã có cấu hình, chưa chạy trên GitHub vì chưa push. |
| Git theo dõi build rác | Gỡ 306 tệp khỏi index: 303 tệp trong danh sách review và ba bản build trùng `main.bbl`, `main.pdf`, `supplement.pdf`. Tất cả tệp local còn nguyên, hash trước/sau khớp. Không rewrite lịch sử. |

## Bằng chứng kiểm tra

- Bản sao nguồn 435 tệp, không mang theo payload EEG/checkpoint riêng: Git add/checkout chỉ thực hiện trong repo tạm độc lập để thử bộ lọc EOL; không stage mã nguồn vào repo chính.
- 409 tests collected; 377 passed, 32 skipped, 29 warnings. Warnings gồm các cảnh báo thư viện/collection hiện có; không coi test skipped là passed.
- Suite cuối chặn optimizer update; không khởi chạy chiến dịch train, inference nghiên cứu hoặc bootstrap lại.
- Public manifest và source ZIP kiểm tra đạt trên cả hai kiểu checkout. Hai lock môi trường khớp checksum đã khai báo trước đó.
- Bằng chứng `final` và `best` giữ đủ 15 hash mã thực thi mỗi bộ, đối chiếu cả snapshot. `final` có bản sửa đường dẫn SHA từ trước đợt bảo trì này; bản sửa đó đã được proof ghi nhận và không bị thay đổi tiếp.
- PDF hiện hành: Anh 17 trang, Việt 18 trang, phụ lục 15 trang. Tổng 50 trang giữ nội dung và bố cục; bản dựng mới được render/đối chiếu với bản bàn giao trước. Không thêm figure, sửa số liệu hay đổi kết luận khoa học.
- Các bài test mới kiểm tra hash cache, lỗi khóa tệp, không ghi đè ZIP khi đóng gói lỗi, chống manifest trùng/vượt thư mục, thành viên ZIP, checksum lịch sử và liên kết code của runner v2.

Hồ sơ chạy chi tiết nằm ở `tmp/github_source_check/`, `tmp/github_delivery_qa_20261008/`
và `tmp/github_review_20261007/untracking_verification.json`; chúng là hồ sơ local, không phải tệp cần push.

## Lệnh dùng lại

```powershell
python -m pytest -q
python scripts/build_public_delivery.py --verify-only
python scripts/check_source_checkout.py --test
```

Nếu cần dựng lại tài liệu sau sửa nội dung: `python scripts/build_public_delivery.py`.
Trên máy Windows bị giới hạn TEMP, trình kiểm tra checkout tự dùng thư mục scratch riêng trong workspace.
Không tự sửa quyền thư mục người dùng hoặc cài lại MiKTeX. Các lượt build tại máy này dùng MiKTeX đã cài.

## Trạng thái Git và việc còn cần chủ sở hữu quyết định

- **Chưa commit và chưa push.** Index hiện chỉ có các thao tác gỡ tracking tệp sinh tự động; code/docs mới chưa stage.
- Không xóa tệp trên ổ đĩa. Có thể phục hồi tracking từ lịch sử Git nếu sau này cần; không khôi phục bằng cách ghi đè file local.
- Danh sách nên push/giữ local của review vẫn áp dụng. Bổ sung runtime helper, runner v2, builder/checker, test, CI, public evidence manifest và hướng dẫn mới vào nhóm mã nguồn công khai.
- Không tự chọn LICENSE cho dự án hoặc thay giấy phép upstream ADAST. Người sở hữu quyết định quyền cấp phép; đây không phải lỗi kỹ thuật còn bỏ dở.
- Việc duyệt quyền công bố metadata đã tracked từ trước và các báo cáo vận hành nằm ngoài khẳng định “test/build đạt”. Không dùng `git add .`; trước push phải rà staged allowlist cuối và đích remote/branch.

Đợt sửa này không chứng nhận đã tái lập nghiên cứu từ dữ liệu mới, không phải benchmark tăng tốc,
và không thay thế sự đồng ý của tác giả cho nộp bài.
