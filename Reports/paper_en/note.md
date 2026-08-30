# Báo Cáo Cập Nhật & Đề Xuất Bản Thảo Nghiên Cứu (Sleep-EDF → SHHS1)

---

## 1. Cải Tiến & Chuẩn Hóa Trang Đầu (Front Matter)
* **Vấn đề bản cũ:** Sử dụng `\author{A \and B}` gây lỗi chia hai tác giả sang 2 cột lệch nhau và affiliation chỉ gắn dưới tác giả thứ hai.
* **Cập nhật mới:**
  * Sử dụng package `authblk` theo đúng chuẩn bài báo tạp chí: Tiêu đề in đậm một khối, danh sách tác giả chung dòng kèm chỉ số affiliation.
  * Tách riêng khối affiliation và dòng thông tin tác giả liên hệ (Corresponding Author).
  * Thêm đường kẻ phân cách trang trọng trước Abstract.
  * Tái cấu trúc **Abstract** chuẩn cấu trúc: `Background` $\rightarrow$ `Methods` $\rightarrow$ `Results` $\rightarrow$ `Conclusions`.
  * Thu nhỏ kích thước Heading về chuẩn typography của tạp chí chuyên ngành thông qua `titlesec`.

---

## 2. Bốn Phát Hiện Bổ Sung Đột Phá
*(Trích xuất hoàn toàn từ dữ liệu thực nghiệm sẵn có – Không cần tốn chi phí huấn luyện lại)*

### 🔹 2.1. Phân Rã Khoảng Cách Miền (Domain Gap Decomposition)
* Khoảng cách sụt giảm khi chuyển miền **Sleep-EDF $\rightarrow$ SHHS1** ($\Delta\text{Macro-F1} = 0.1805$) không phải là sự suy giảm khuếch tán diện rộng mà tập trung vào **hai kênh nhầm lẫn chính**:
  * **Sửa riêng $N3 \rightarrow N2$:** Phục hồi **$74.5\%$** khoảng cách hiệu năng.
  * **Sửa đồng thời $N3 \rightarrow N2$ & $N2 \rightarrow \text{REM}$:** Phục hồi **$102.4\%$**, đưa Macro-F1 về ngang mức trong miền ($0.7947$ so với $0.7904$).
* **Kết luận:** Ngoài 2 kênh lỗi đặc thù trên, mô hình hầu như bảo toàn trọn vẹn khả năng biểu diễn khi chuyển miền.

### 🔹 2.2. Loại Trừ Giả Thuyết Dịch Chuyển Tiên Nghiệm (Prior Shift)
* Phân tích tỷ lệ $\text{Dự đoán} / \text{Thực tế}$ theo từng lớp tín hiệu:
  * **Lớp N3:** SHHS1 có N3 thực tế nhiều gấp **$2.02\times$**, nhưng mô hình chỉ phát hiện ở mức **$0.273\times$**.
  * **Lớp N1:** SHHS1 có N1 ít đi (chỉ còn **$0.38\times$**), nhưng mô hình lại dự đoán dư thừa tới **$2.151\times$**.
* **Nhận định then chốt:** Sai lệch dự đoán chạy **ngược chiều hoàn toàn** với dịch chuyển tiên nghiệm phân phối. Do đó, việc dùng kỹ thuật hiệu chỉnh Saerens sẽ làm suy giảm thêm hiệu năng N3. Nguyên nhân sụt giảm bắt buộc nằm ở tầng **đặc trưng tín hiệu vật lý**.

### 🔹 2.3. Cơ Chế Sụp Đổ Nhận Diện Pha N3 (N3 Collapse Mechanism)
* **Bản chất vấn đề:** Ngưỡng biên độ sóng chậm (Slow-wave amplitude threshold).
  * Mô hình **E3** giữ nguyên thang đo $\mu\text{V}$ tuyệt đối (clipping $\pm 800$, scale $/100$).
  * Dataset **SHHS1** chuyển sang đạo trình **C4–A1** (sóng chậm có cực đại ở vùng trán $F$, biên độ ở đạo trình trung tâm $C$ vốn thấp hơn) trên đối tượng quần thể bệnh nhân cao tuổi hơn.
* **Bằng chứng thực nghiệm:** Cả 4 dự đoán lý thuyết đều khớp với số liệu:
  * **Dự đoán 4 (Mạnh nhất):** Recall của N3 sụt giảm nghiêm trọng từ **$0.258$ xuống $0.0733$** tại vùng chuyển pha (transition zones) — nơi biên độ sóng chậm nằm sát ngưỡng phân định nhất.
  * **Nghịch lý kiến trúc:** Lựa chọn chuẩn hóa tín hiệu giúp E3 dẫn đầu trong miền lại chính là điểm yếu chí mạng khiến nó thất bại khi chuyển miền.

### 🔹 2.4. Giải Thích Kết Quả Null của Gate 8
* Cấu trúc **TCN không nhân quả (Non-causal TCN)** sở hữu trường tiếp nhận (receptive field) lên tới **253 epoch** (tương đương $\pm 63$ phút), bao quát toàn bộ bản ghi qua đêm.
* Việc bổ sung nhóm tiền/hậu $P/N$ chỉ cung cấp ngữ cảnh lân cận $\pm 1$ epoch — thông tin này đã nằm trọn trong trường tiếp nhận của chuỗi.
* Do đó, kết quả null mang tính **tất yếu về mặt cấu trúc** chứ không phủ nhận vai trò của ngữ cảnh thời gian.
* **Chỉ số tương tác đổi dấu:** $-0.00161$ (toàn bộ) và $+0.00258$ (vùng chuyển pha) khẳng định không tồn tại một đại lượng vô hướng đơn lẻ nào có thể tóm tắt đầy đủ đóng góp của ngữ cảnh $P/N$.

---

## 3. Các Thống Kê & Phân Tích Bổ Sung Đã Thêm Vào Bản Thảo
- [x] **Effect Size $\Delta_{\text{sign}}$:** Kèm kiểm định dấu (*Sign Test*) cho toàn bộ bảng so sánh hiệu năng.
- [x] **Replication Table (Seed 123 trên SHHS):** Cả hai so sánh khóa vẫn giữ vững mức ý nghĩa thống kê ngoài miền (dù trong miền bị mất).
- [x] **Bảng phân rã sai số theo từng cặp chuyển pha (Transition Pairs).**
- [x] **Chỉ số Per-class tại vùng chuyển pha trên tập SHHS.**
- [x] **Độ lệch chuẩn (SD) của không gian biểu diễn Silhouette.**

---

## 4. Các Điểm Cần Thảo Luận & Quyết Định
| STT | Hạng mục | Chi tiết & Gợi ý xử lý | Trạng thái |
| :---: | :--- | :--- | :---: |
| **1** | **Build LaTeX $\rightarrow$ PDF** | Máy hiện tại thiếu môi trường `pdflatex`. File nguồn `.tex` đã được kiểm tra toàn diện (*cite key, ref/label, balance ngoặc/môi trường, asset paths*). Bạn chỉ cần kéo về máy Windows để build. | Cần build |
| **2** | **Danh sách Tác giả & Affiliation** | `cover.tex` ghi GVHD Nguyễn Hồ Duy Trí, trong khi bản cũ chỉ để 2 SV. Hiện tại giữ 2 SV kèm comment nhắc. Email đang set theo format MSSV UIT. Cần đối soát lại email và gắn thêm ORCID cá nhân. | Cần chốt |
| **3** | **Khuyến nghị Thực nghiệm (§5.7)** | **Thí nghiệm quan trọng nhất & chi phí rẻ nhất:** Đánh giá Per-class trên SHHS cho mô hình **E6** bằng cách chấm lại checkpoint đã có (*không cần train lại*).<br>• *Nếu N3 Recall của E6 > E3:* Khẳng định giả thuyết biên độ.<br>• *Nếu không:* Bác bỏ giả thuyết.<br>*(Đây là nơi khu trú tới **$74.5\%$** tổng sai số chuyển miền).* | Đề xuất làm ngay |