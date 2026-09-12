# Những việc Quân cần xác nhận trước khi nộp BSPC

Cập nhật kiểm tra 11-09-2026. Nội dung dưới đây lưu thông tin tác giả đã cung cấp trước đó,
không phải xác minh độc lập mọi khai báo. Các việc còn cần xác nhận được liệt kê đầy đủ trong
`BSPC_SUBMISSION_CHECKLIST_20260911.md`, gồm căn cứ ethics, vai trò CRediT thực tế, đồng ý của
người được cảm ơn, nguồn hình và yêu cầu portal. Đây không phải yêu cầu chạy thêm thí nghiệm.

## 1. Tên và đóng góp của tác giả

- Đã ghi Ngô Nhật Quân / Ngo Nhat Quan đứng đầu và là tác giả liên hệ, email `23521258@gm.uit.edu.vn`, theo xác nhận của bạn. Phạm Thái Sơn vẫn là đồng tác giả thứ hai trong bản hiện tại.
- Theo xác nhận mới nhất, cả hai tác giả đã đọc và duyệt bản cuối. CRediT đã được ghi theo phân công đã thống nhất: Ngô Nhật Quân chịu trách nhiệm về conceptualization, methodology, software, data curation, formal analysis, investigation, visualization, viết bản đầu và sửa bài; Phạm Thái Sơn tham gia validation, investigation, visualization và sửa bài.
- Bản thảo không ghi tỷ lệ phần trăm đóng góp; CRediT mô tả vai trò thực tế thay cho con số phần trăm.
- Không tự thêm “Supervision” cho Sơn hoặc thêm giảng viên làm đồng tác giả chỉ dựa vào chức danh. Hãy ghi đúng đóng góp thực tế và hỏi người liên quan.
- Theo quyết định của tác giả chính, Nguyễn Hồ Duy Trí không được liệt kê là đồng tác giả. Bản thảo ghi lời cảm ơn về hướng dẫn học thuật, định hướng đề tài và góp ý cho các phiên bản trước; cần có sự đồng ý của thầy trước khi giữ tên trong bản nộp cuối.
- Mỗi tác giả đã xác nhận đọc và duyệt bản cuối. ORCID chỉ điền nếu đã có và kiểm tra đúng tài khoản.

## 2. Ethics: yêu cầu đạo đức nghiên cứu

Hiểu đơn giản: trường/đơn vị có yêu cầu xét duyệt hoặc xác nhận miễn xét duyệt cho việc phân tích lại dữ liệu người tham gia này hay không?

Theo xác nhận của tác giả, phân tích thứ cấp dữ liệu đã khử định danh này không cần xét duyệt đạo đức tại đơn vị áp dụng; bản thảo không nêu mã phê duyệt. Hãy giữ lại căn cứ hoặc trao đổi xác nhận này trong hồ sơ nội bộ nếu tạp chí yêu cầu giải trình.

Bạn đã xác nhận đã xin được quyền truy cập SHHS/NSRR cho phân tích này. Không đưa mật khẩu, token hoặc subject IDs vào manuscript; giữ acknowledgement và các điều kiện sử dụng bắt buộc của NSRR khi nộp.

## 3. DUA/NSRR: quyền dùng dữ liệu SHHS

DUA là thỏa thuận sử dụng dữ liệu. Hãy kiểm tra tài khoản/email hoặc hỏi người đã tải SHHS cho dự án:

- Ai đứng tên yêu cầu và được cấp quyền truy cập?
- Quyền đó có bao phủ người tham gia dự án và mục đích nghiên cứu hiện tại không?
- Có acknowledgement, citation hoặc hạn chế chia sẻ nào phải giữ không?

Giữ thông tin xác nhận trong hồ sơ riêng; không gửi mật khẩu, token hoặc danh sách subject IDs vào manuscript. Không upload raw SHHS hoặc prediction cấp người tham gia lên GitHub chỉ để có một “reproducibility package”. Nếu chưa rõ quyền, hỏi đầu mối dữ liệu trước khi công bố. Xem [cổng dữ liệu NSRR](https://sleepdata.org/) và thông tin kèm quyền truy cập thực tế của nhóm.

## 4. Funding: nguồn tài trợ

Nghiên cứu nhận khoảng 6.000.000 VND hỗ trợ sinh viên từ Trường Đại học Công nghệ Thông tin, Đại học Quốc gia Thành phố Hồ Chí Minh. Bản thảo ghi khoản hỗ trợ này, không tự gán mã grant hoặc nguồn tài trợ khác khi chưa có thông tin chính thức.

## 5. Competing interests: xung đột lợi ích

Theo xác nhận của tác giả, hai tác giả không có xung đột lợi ích cần khai báo. Declaration trong bản thảo dùng công thức tiêu chuẩn “The authors declare no competing interests.”

## 6. Mã nguồn và artifact cho reviewer

Hiện đã có mã, protocol snapshots, aggregate kết quả và `Reports/ARTIFACT_INDEX.md`. URL repository đã được cung cấp là `https://github.com/quanuiter/SleepTCN`; khi nộp cần push các thay đổi hiện tại và ghi commit/tag tương ứng, vì remote hiện vẫn có thể đang ở revision cũ hơn bản manuscript local.

- GitHub có thể dùng làm repository nếu link truy cập được và nội dung không chứa dữ liệu SHHS hạn chế, subject IDs hoặc prediction cấp cá thể. Chọn repository hoặc archive người phản biện truy cập được; xác nhận với đồng tác giả những gì được phép đưa lên.
- Ghi version/commit/release thật và license do chủ sở hữu chấp thuận; model không tự cấp license hoặc tự public kho mã.
- Có thể tách mã/aggregate công bố được khỏi dữ liệu SHHS hạn chế. Việc thiếu quyền chia sẻ raw data phải được giải thích, không che bằng câu “fully reproducible”.
- BSPC dùng single-anonymized review theo Guide for Authors đã được tác giả gửi ngày 11-09-2026;
  giữ tên tác giả trong manuscript.

## 7. Khai báo AI

Trong lượt chỉnh này, OpenAI Codex hỗ trợ biên tập, tổ chức bài, đối chiếu code/số liệu đã lưu và mã vẽ lại hình từ aggregate hiện có. Không huấn luyện hay sinh prediction mới; bạn xác nhận không dùng công cụ AI nào khác cho việc chuẩn bị bản thảo.

Không có công cụ AI nào khác được dùng cho việc chuẩn bị bản thảo theo xác nhận hiện tại. Bản thảo ghi rõ OpenAI Codex đã hỗ trợ biên tập, tổ chức nội dung, kiểm tra nhất quán và dựng hình từ số liệu đã lưu; hai tác giả đã đọc và duyệt bản cuối và chịu trách nhiệm về nội dung. [Chính sách AI của Elsevier](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals).

## 8. Kiểm tra yêu cầu BSPC trong cổng nộp

Đã đọc đủ PDF 22 trang của Guide for Authors do tác giả gửi ngày 11-09-2026.
Việc trang web trả HTTP 403 trước đó không còn ngăn việc đối chiếu guide.

- Abstract dưới 250 từ; keywords 1–7; full paper thông thường khoảng 5.000 từ, không phải giới hạn cứng.
- Review single-anonymized; cung cấp tên tác giả và địa chỉ bưu chính đầy đủ; LaTeX là nguồn hợp lệ.
- Highlights bắt buộc, file editable riêng, 3–5 ý, mỗi ý tối đa 85 ký tự; graphical abstract khuyến khích.
- Hoàn thành declarations tool và upload file Word xuất từ công cụ, kể cả khi không có competing interests.
- Supplement nộp đồng thời, có dẫn chiếu và mô tả file; nhà xuất bản không dàn trang lại supplement.
- Research data theo Option C: cung cấp repository/citation/link hoặc lý do không thể chia sẻ;
  quyền truy cập NSRR không tự cho phép phân phối lại dữ liệu.
- Vẫn cần kiểm tra các trường và loại file thực tế trên portal khi upload.

## 9. Có cần thêm số liệu không?

Không cần huấn luyện mới chỉ để sửa các lỗi diễn giải đã xác định. Lượt này chỉ đối chiếu/tái tính kết quả đã lưu và đưa thêm bằng chứng seed 123 đã tồn tại vào supplement.

Các bổ sung **chưa thực hiện**, cần bạn chọn nếu muốn:

1. Bảng demographics đúng 78 EDF/180 SHHS: lấy từ metadata hợp lệ, không dùng trung bình toàn cohort để giả làm trung bình của mẫu.
2. Phân bố/CI bổ sung theo subject, phân tích calibration hoặc độ nhạy ensemble: cần một kế hoạch tái phân tích, không bịa từ confusion matrix gộp.
3. Baseline mới, holdout mới hoặc adaptation: là công việc thực nghiệm riêng. Chỉ bắt buộc nếu bạn muốn thêm claim mà thiết kế hiện tại chưa hỗ trợ; không phải điều kiện mặc định do bài chưa có kiến trúc mới.

Phần seed 2025 xuất hiện trong config lịch sử nhưng chưa có kết quả báo cáo: cần bạn xác nhận diễn tiến kế hoạch, không sửa config hoặc chế lý do. Seed 123 được trình bày thận trọng là sensitivity, không nâng thành bằng chứng xác nhận độc lập.

## Mẫu trả lời ngắn khi đã có thông tin

```text
Tên xuất bản / email liên hệ:
Đóng góp của Quân:
Đóng góp của Sơn:
Xác nhận của trường về ethics:
Người đứng tên/quyền NSRR-SHHS:
Funding và acknowledgement của nhóm:
Competing interests của từng tác giả:
Repository/archive URL + version + nội dung được phép chia sẻ:
Các công cụ AI thực dùng:
Preprint/thesis/conference version hoặc submission khác: Không có theo xác nhận hiện tại.
Các tác giả đã duyệt bản cuối: Có / Chưa
```

Không cần điền nội dung nhạy cảm trong câu trả lời; chỉ cung cấp thông tin đủ để soạn declaration đúng sự thật.
