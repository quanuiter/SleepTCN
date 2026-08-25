# Danh mục kiểm tra bản thảo và hồ sơ công bố

## Yêu cầu về nội dung khoa học

- [ ] Mọi số liệu trong Abstract khớp bảng sinh tự động.
- [ ] Chỉ gọi Macro-F1 là chỉ số chính; Accuracy/kappa là hỗ trợ.
- [ ] Nêu rõ 78 đối tượng, 153 bản ghi, 195.469 epoch hợp lệ.
- [ ] Nêu rõ một training seed 42.
- [ ] Không coi 10 fold là 10 mẫu độc lập.
- [ ] Holm chỉ gồm bốn so sánh chính.
- [ ] E4−E2 được ghi là phân tích thứ cấp.
- [ ] Không tạo p-value E5−E4.
- [ ] Báo cáo đồng thời tốc độ, tham số và VRAM.
- [ ] Không sử dụng tuyên bố 8,2×.
- [ ] Không tuyên bố domain shift, zero-shot, SHHS hoặc giá trị lâm sàng.

## Yêu cầu về trình bày

- [ ] Thêm trích dẫn paper gốc và các công trình liên quan bằng nguồn chính thức.
- [ ] Định nghĩa mọi chữ viết tắt khi xuất hiện lần đầu.
- [ ] Mọi bảng/hình có caption, đơn vị và phạm vi đo.
- [ ] Hình t-SNE được ghi rõ chỉ mang tính mô tả.
- [ ] Sơ đồ quy trình phân biệt preprocessing, extractor và sequence model.
- [ ] Phụ lục ghi commit, mã băm config/split và môi trường phần mềm.

## Tái lập và provenance

- [ ] Nhánh/commit công bố đã được gắn tag.
- [ ] Không đưa dataset, cache hoặc metadata mức epoch lên kho công khai.
- [ ] Lệnh tái tạo bảng/hình Gate 8 được ghi trong README/runbook.
- [ ] Tất cả kiểm thử và manifest xuất bản trả trạng thái đạt.


## Gate 8 — ablation nhóm đặc trưng C/P/N

- [ ] Ghi rõ Full CPN tái sử dụng E1, còn CP/CN/C được huấn luyện lại TCN.
- [ ] Ghi rõ vector thay thế chỉ được tính từ dữ liệu train hợp lệ trong từng fold.
- [ ] Tiêu chí chính là Macro-F1 vùng chuyển pha ±1; Holm gồm đúng ba so sánh Gate 8.
- [ ] Không dùng cụm “12% thông tin” hoặc bất kỳ phần trăm thông tin nào.
- [ ] Không diễn giải p lớn thành tương đương hoặc không thua kém.
- [ ] Ghi rõ Gate 8 là phân tích cơ chế bổ sung với một training seed.
- [ ] Không tuyên bố P/N vô dụng hoặc C đủ thay thế CPN.
