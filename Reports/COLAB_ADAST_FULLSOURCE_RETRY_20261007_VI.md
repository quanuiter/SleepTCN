# Tiếp tục đối chiếu full-source source-only/ADAST trên GPU – 07/10/2026

## Trạng thái mới nhất: đánh giá epoch 30 và so sánh ngân sách đã đạt kiểm chứng

Lỗi đường dẫn đã được sửa và kiểm chứng hoàn tất sau 65,173 giây: đủ 180 tệp dự đoán SHHS, OOF mười fold, 20 checkpoint, 40 phát lại mô hình/bản ghi đích và 20 batch nguồn. Confusion, toàn bộ chỉ số và bootstrap được tính lại khớp hoàn toàn; aggregate gốc giữ nguyên hash. So sánh bốn hệ thống trên cùng 180 người, 169.012 epoch chấm điểm và 10.000 bootstrap dùng chung lượt lấy mẫu cũng đạt kiểm chứng bằng tiến trình riêng. Lượt best theo validation nguồn đã bắt đầu; không thay kết quả chính epoch 30 bằng điểm best sau khi xem SHHS.

Kết quả chính full-source: macro-F1 trung bình theo người của source-only/ADAST là 0,5010/0,5455; ADAST trừ đối chứng là +0,0445 (95% CI [0,0352; 0,0539]). Ở ngân sách nhỏ, hiệu ứng là -0,0452 [-0,0558; -0,0347]. Chênh lệch hai hiệu ứng là +0,0897 [0,0767; 0,1023]. Recall N3 full-source tăng 0,2432 lên 0,6111, nhưng recall N1 giảm 0,1568 xuống 0,1057; accuracy là 0,6898/0,6873. Recall REM tăng 0,6976 lên 0,8360 nhưng F1 REM giảm 0,0488 do precision thấp hơn. Vì vậy không mô tả mọi lớp đều cải thiện. Thay ngân sách đồng thời đổi số update, số lần xem target và vị trí scheduler theo update, không cô lập riêng độ phủ nguồn. Paper/PDF chưa được cập nhật trong bước sửa mã này.

### Hồ sơ sửa lỗi đường dẫn

Lượt epoch 30 đã suy luận/chấm đủ 180 SHHS và OOF Sleep-EDF mười fold. Bước kiểm chứng độc lập dừng lúc 19:30 vì hàm SHA256 nhận đường dẫn dạng chuỗi từ JSON thay vì `Path`; chưa có sai khác số liệu được báo. Sau yêu cầu sửa của người dùng, chỉ thay một dòng chuyển kiểu tại biên hash trong evaluator, và sửa cùng lỗi ở bước phân tích ngân sách. 17 kiểm thử thuần đều đạt, gồm từ chối mọi sửa mã ngoài phép chuyển kiểu và từ chối thay đổi snapshot gốc. Không train hoặc tạo lại dự đoán epoch 30.

Verifier sửa lỗi bắt đầu lúc 20:04, giới hạn 1.800 giây. Nó kiểm tra nguyên snapshot đã chạy, cho phép đúng một dòng chuyển `Path`, đối chiếu mọi hash đầu vào/checkpoint/dự đoán, tính lại toàn bộ confusion/metric/bootstrap, phát lại 40 cặp mô hình/bản ghi đích và 20 batch nguồn. Aggregate, specification, snapshot và các dự đoán gốc không bị thay. Khi proof đạt mới tiếp tục so sánh ngân sách và chấm best riêng. Tổng ngân sách hoạt động vẫn 18.000 giây, tính cả 2.124 giây của lượt đã chấm. Hồ sơ sửa ở `runs/adast_fullsource_retry_20261007/verification_path_repair_20261007`.

### Ghi nhận đánh giá trước lỗi kiểm chứng đường dẫn

Người dùng đã kết nối lại ổ E. Lúc 18:54 ngày 07/10, manifest, đủ 180 bản ghi tín hiệu và đủ 180 tệp tham chiếu chấm điểm đều đọc được. Đã tiếp tục đánh giá local, giữ riêng hai tệp hồ sơ của lần dừng khi thiếu ổ; lần trước chưa có dự đoán nào. Tiến trình mới đã qua kiểm tra dữ liệu/checkpoint và thực sự suy luận SHHS. Chuỗi chạy lần lượt epoch 30, so sánh tác dụng ADAST ở hai ngân sách với 10.000 bootstrap bắt cặp, rồi checkpoint best chọn trên validation nguồn. Các giai đoạn sau chỉ chạy khi giai đoạn trước hoàn tất và proof khớp hash. Giới hạn chung 18.000 giây; không train hoặc tự chạy lại khi lỗi. Hồ sơ tiến trình mới nằm ở `runs/adast_fullsource_retry_20261007/evaluation_after_drive_reconnect_20261007`.

### Kiểm chứng đạt và điểm dừng do thiếu ổ trước đó

Verifier v2 hoàn tất lúc 18:04 ngày 07/10, sau 336,985 giây. Đủ 16 mô hình mới, 496 diagnostic và 32 checkpoint best/final đã đạt kiểm tra độc lập trên toàn validation nguồn. Hai proof đều `passed`, cùng gắn với aggregate có SHA256 `52af729f0fe3b7814215e7950b524867c9eb930ac45868ee6b9940faf62b5432`; hash thực tế của aggregate và independent proof khớp con trỏ kết quả. Đây là kiểm chứng mô hình/validation nguồn, không phải điểm SHHS. Wrapper PowerShell ghi exit code null và suy ra failed; trạng thái wrapper này không thay thế hai proof đã hoàn tất và đối chiếu hash.

Đã khởi chạy bước đánh giá epoch 30 lúc 18:07. Bước kiểm tra dữ liệu dừng sau 8,473 giây vì ổ E không có trong danh sách ổ filesystem hiện tại (chỉ C và D), nên không đọc được manifest SHHS tại vị trí lịch sử. Chưa suy luận hoặc chấm người SHHS nào; chưa có số liệu full-source SHHS/OOF hay so sánh hai ngân sách để đưa vào paper. Chuỗi phân tích và best chưa chạy. ZIP, checkpoint, lịch sử và hồ sơ lỗi đều giữ nguyên; không tự khởi chạy lại. Cần kết nối lại ổ chứa SHHS hoặc xác định vị trí mới của đúng bộ dữ liệu trước khi tiếp tục. Hồ sơ: `runs/adast_fullsource_retry_20261007/evaluation_blocker_20261007.json`.

### Ghi nhận v2 trước khi hoàn tất

Sau yêu cầu tiếp tục đánh giá, đã tạo verifier v2 riêng; giữ nguyên verifier, log và điểm dừng của lượt 30 phút. V2 bắt đầu thực thi lúc 17:58:26 +07, không train hoặc sửa checkpoint. Kiểm tra vẫn bao gồm 496 diagnostic, 32 checkpoint best/final trên toàn validation nguồn, hai đường attention, sampling/loss/LR/optimizer/RNG và hash. Kết quả kiểm tra từng mô hình được lưu riêng để không mất hồ sơ tiến độ nếu dừng. Giới hạn an toàn của lượt mới là 18.000 giây, không tự chạy lại sau lỗi/giới hạn.

Đo nguyên nhân chậm trên 256 đoạn validation nguồn của ADAST fold 2 tìm thấy 74.149 giá trị float32 subnormal trong trạng thái mô hình. Với hai đường attention chấm riêng, chế độ cũ hai thread mất 22,276 giây; bốn thread cùng chế độ xử lý denormal phù hợp mất 0,223 giây. Logits và argmax trên phép thử này khớp hoàn toàn; hash trạng thái mô hình không thay đổi. V2 dùng bốn thread, tái sử dụng cùng feature encoder cho hai đường attention và bật flush-denormal chỉ khi suy luận, rồi trả về chế độ cũ trước kiểm tra trạng thái. Cả hai đường vẫn chấm đủ mọi đoạn validation và đối chiếu đúng chỉ số đã lưu từ CUDA, không bỏ đường hoặc giảm mẫu để đạt thời gian. Mã forward tối ưu đã qua kiểm thử exact-logit, batch cuối, eval mode, trạng thái/gradient và stop guard. 75 kiểm thử verifier/helpers/đánh giá đều đạt.

Đây là xử lý kỹ thuật của inference local, không thay đổi recipe huấn luyện, tiền xử lý, tiêu chí chọn checkpoint hoặc benchmark vận hành trong bài. Gói đầy đủ vẫn giữ nguyên SHA đã quan sát từ Colab. Chỉ sau khi proof v2 hoàn tất mới đánh giá SHHS và OOF nguồn. Mã đánh giá ghi rõ chế độ CPU float32/flush-denormal; kết quả chính vẫn epoch 30, best theo validation nguồn vẫn tách riêng.

### Điểm dừng của lượt kiểm tra trước

Verifier kết thúc lúc 17:46:52 +07 sau 1.804,906 giây; lỗi duy nhất trong traceback là `TimeoutError: Verification limit reached`. Đã kiểm tra trọn vẹn năm mô hình: hai nhánh ở fold 2, hai nhánh ở fold 3, và source-only fold 4. Mô hình ADAST fold 4 đã qua phần best, đang phát lại final thì chạm giới hạn; không xác định phần final đã hoàn tất qua cả hai attention. Không có lỗi hash hoặc sai khác confusion/metric được báo trước điểm dừng.

Đây là giới hạn của kiểm tra/suy luận local, không phải thất bại huấn luyện GPU. ZIP đầy đủ và tất cả checkpoint vẫn giữ nguyên; kiểm tra lại hash ZIP và verifier sau khi dừng vẫn khớp hồ sơ lúc chạy. Chưa tạo proof hoàn tất hoặc con trỏ verified-results; đánh giá full-source SHHS chưa được khởi chạy. Không tự chạy lại verifier, tăng giới hạn, bỏ bớt kiểm tra hay train lại. Cần xử lý tốc độ kiểm tra toàn validation trước khi tiếp tục bước đánh giá và cập nhật kết luận paper.

Các bản ghi bên dưới mô tả tiến trình trước điểm dừng, không đại diện trạng thái đang chạy.

### Đã nhận gói đầy đủ và khởi chạy kiểm tra

Gói đầy đủ đã tải về lúc 17:15 ngày 07/10, kích thước 700.561.132 byte. SHA256 của tệp local khớp đúng giá trị `DOWNLOAD_SHA256` quan sát trong Colab. Đã khởi chạy verifier độc lập lúc 17:16; quá trình này chỉ kiểm tra và suy luận trên CPU, không huấn luyện lại. Verifier kiểm tra 496 diagnostic và phát lại 32 checkpoint best/final trên toàn validation nguồn, giới hạn 1.800 giây. Giữ nguyên mã huấn luyện và các checkpoint lịch sử.

Đã chuẩn bị mã chấm tổ hợp full-source trên 180 người SHHS, OOF Sleep-EDF, và so sánh hiệu ứng ADAST giữa hai ngân sách bằng 10.000 lượt bootstrap bắt cặp ở cấp người. Mô hình epoch 30 dùng cho đối chiếu chính; best theo validation nguồn chấm riêng. Mã phân tích kiểm tra lại kết quả bằng tiến trình riêng. Chưa khởi chạy đánh giá full-source SHHS trong khi verifier checkpoint còn hoạt động; chưa thay số liệu/kết luận của bản thảo bằng diagnostic nguồn.

39 kiểm thử thuần của verifier và mã đánh giá/phân tích đều đạt. Kiểm thử bao gồm công thức năm lớp, giữ đủ mười fold, từ chối proof thiếu hoặc input thay đổi, và đối chiếu khoảng tin cậy difference-of-differences với bootstrap tính thủ công. Không dùng dữ liệu thực để thử chọn cấu hình và không train trong fixture. Launcher đánh giá đã chuẩn bị nhưng chưa chạy; yêu cầu proof checkpoint đạt trước khi bắt đầu, giữ kết quả epoch 30 và best trong hai thư mục riêng.

### Ghi nhận trước khi tải gói hoàn tất

Ngày 07/10, ô 18 kết thúc lúc 16:43 +07 và in `EIGHT MATCHED FULL-SOURCE PAIRS COMPLETE. DOWNLOAD RESULTS FOR LOCAL VERIFICATION.` Đã quan sát đủ 16 mô hình mới (source-only và ADAST cho folds 2–9), mỗi mô hình hoàn thành 30 epoch trên T4. Không chạm giới hạn năm giờ và không train CPU. Thời lượng chính xác của runner sẽ lấy từ aggregate sau khi nhận gói kết quả; không dùng thời gian ô notebook thay cho thời gian huấn luyện.

Gói đầy đủ `SleepTCN_ADAST_Fullsource_Completion_Results_20261006.zip` có kích thước hiển thị 668,11 MiB; SHA256 thực sự in ở `DOWNLOAD_SHA256` là `59cd81bca2a1bb67149b8d3348415d63c8524642240a32b608723ce56c0dd295`. Ngày trong tên tệp là ngày thiết kế chiến dịch, không phải ngày lượt retry hoàn tất. Đã yêu cầu tải qua ngăn Tệp một lần; tại lần kiểm tra này chưa thấy gói đầy đủ ở máy. Không gửi yêu cầu tải trùng hoặc chạy lại ô. Ba bản backup folds 2–4 đã có ở máy và khớp size/hash; các backup khác chưa xác nhận được lưu local.

Bước tiếp theo: đối chiếu hash ZIP, kiểm tra độc lập diagnostics và checkpoint của 16 mô hình mới, kiểm tra lại bốn mô hình reuse folds 0–1, rồi mới chấm SHHS và OOF Sleep-EDF. Các điểm trong output huấn luyện là validation nguồn; chưa có điểm SHHS của tổ hợp full-source và chưa cập nhật kết luận khoa học của paper từ log này. Bằng chứng terminal: `runs/adast_fullsource_retry_20261007/training_complete_20261007.jpg`.

### Chuẩn bị kiểm tra độc lập lúc 17:12 +07

Đã thêm `scripts/verify_adast_fullsource_completion_results.py` và các kiểm thử chỉ dành cho verification, không train CPU kể cả trong fixture. 35 kiểm thử của verifier mới và các helper diagnostics/sampling hiện có đều đạt; hai cảnh báo chỉ liên quan quyền ghi pytest cache, không ảnh hưởng kết quả test. Verifier yêu cầu gói đầy đủ, kiểm tra cả 496 diagnostic nguồn qua hai đường attention và phát lại 32 checkpoint best/final trên toàn validation nguồn; kiểm tra sampling, loss, LR, optimizer update count, khởi tạo, chọn best first-tie và hash. Không nhập runner training để thực thi.

Kiểm tra đọc-only bổ sung trên backup fold 2 đã đối chiếu hash gói với log, xác nhận cả hai nhánh đủ 30 epoch và optimizer final khớp ngân sách/LR; pseudo-label có dtype int64 đúng schema. Đây chỉ là kiểm tra chuẩn bị trên backup, không phải bằng chứng toàn bộ chiến dịch đã đạt verifier. Gói cuối vẫn chưa xuất hiện ở hai vị trí tải local tại lần kiểm tra 17:12; không gửi yêu cầu download trùng và chưa chạy verifier production hoặc đánh giá SHHS.

## Ghi nhận huấn luyện đã bắt đầu

Sau upload, output xác nhận ALL INPUT HASHES VERIFIED. Đã quan sát epoch 1/30 của source-only fold 2: 1.189 update, 17,07 giây; source-validation macro-F1 0,69975, N1 recall 0,2549, N3 recall 0,8440. Đây là diagnostic của validation nguồn trong quá trình học, không phải điểm SHHS hoặc kết quả toàn chiến dịch. Ảnh training_started_20261007.jpg ghi nhận output mới và T4 đang thực thi.

## Ghi nhận giai đoạn upload lúc 13:07 +07

Thẻ notebook người dùng vừa gửi đã kết nối được Tesla T4. Chẩn đoán mới xác nhận CUDA, PyTorch 2.11.0+cu130 và 14,56 GiB VRAM. Kiểm tra tiến trình không có runner huấn luyện trước khi bắt đầu; ngăn tệp ban đầu chỉ có .config và sample_data.

Đã khởi chạy ô 18 (cell-TREUIm2Jd7ZW, execution 3 của phiên mới) khoảng 13:05 và gửi đúng chín phần dữ liệu đã cho phép bằng filechooser một lần. Các phần đang tăng kích thước; chưa tải xong. Output hiện là WAITING FOR INPUT PARTS. NO TRAINING STARTED. Đây là upload, chưa phải train. Launcher sẽ tự kiểm tra kích thước/hash, ghép ZIP và kiểm chứng payload trước khi train CUDA. Không gửi lại phần đang tải và không chạy lại ô.

## Phạm vi giữ nguyên sau rà soát

- 16 mô hình còn thiếu: source-only và ADAST cho folds 2–9, 30 epoch toàn bộ nguồn mỗi epoch, seed 123; dùng lại bốn mô hình local folds 0–1 đã kiểm chứng.
- Gói khoa học/lõi training/cấu hình không thay đổi. 40 kiểm thử trước chạy đạt; toàn bộ payload, chín phần và 48 tệp inventory reuse đã được kiểm tra lại hash.
- Không train CPU. Giới hạn training chung 18.000 giây, lưu checkpoint optimizer/RNG mỗi epoch; backup từng cặp fold sau hoàn tất. Yêu cầu download không đồng nghĩa đã lưu local, cần kiểm tra tệp và hash.
- Không đưa dữ liệu/nhãn 180 người SHHS test lên cloud, không mount Drive. Mỗi mô hình lấy đúng train/validation của fold dù bundle vật lý chứa toàn nguồn đã được cho phép.
- Mục tiêu: đo tác dụng ADAST với đối chứng cùng ngân sách full-source, sau đó so sánh tác dụng thích nghi giữa ngân sách nhỏ và lớn. Không thêm nhánh chỉ để có nhiều kết quả.
- Chỉ sau kiểm chứng đầy đủ checkpoint/diagnostics mới đánh giá SHHS local và cập nhật paper. Đối chiếu chính final epoch 30; phân tích best theo validation nguồn riêng.

Đã cập nhật và bật lại lịch theo dõi cũ cho đúng ô/phiên retry, giữ im lặng khi tiến trình bình thường và chỉ báo thay đổi cần biết. Hồ sơ live nằm trong runs/adast_fullsource_retry_20261007/colab_launch_record_current.json; các snapshot chuẩn bị cũ không đại diện trạng thái hiện tại.
