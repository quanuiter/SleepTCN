# Rà soát tài liệu tham khảo paper BSPC - 01/10/2026

## Kết luận

Đã kiểm tra toàn bộ 29 mục trong `paper_en/references.bib`, 26 mục được trích dẫn và
xuất ra PDF sau chỉnh sửa. Không có khóa trích dẫn thiếu, mục trùng DOI, hoặc sai thứ tự
lần xuất hiện đầu tiên. Bản dịch Việt dùng cùng danh mục với bản Anh. Supplement độc lập
không có lệnh trích dẫn thư mục hay danh mục tham khảo riêng; các liên kết S1-S17 của bài
chính vẫn được giải quyết.

Các nguồn đã xác minh là công trình có thật, phù hợp với mục đích dẫn trong paper.
Không phát hiện nguồn cần loại vì giả mạo hoặc vì nơi đăng có dấu hiệu tạp chí săn mồi.
Đây là đánh giá xuất xứ và mức phù hợp, không phải bảo đảm mọi kết luận của nguồn đều đúng.
Không tự gán Q1/Q2 hay impact factor khi chưa đối chiếu theo năm và ngành cụ thể.

## Các sửa đã thực hiện

- Bài NSRR: sửa Michael Kim thành **Matthew Kim**, Dennis Mobley thành **Daniel Mobley**.
  Đối chiếu bài gốc tại [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6188513/) và metadata Crossref.
- Bài độ tin cậy chấm giấc ngủ: sửa Steve Van Hout thành **Steven Van Hout** theo
  [bài gốc](https://pmc.ncbi.nlm.nih.gov/articles/PMC3525994/).
- Dataset Sleep-EDF: ghi **Bob Kemp** là tác giả tài nguyên, PhysioNet là kho lưu trữ;
  cập nhật ngày truy cập thực tế thành 1 October 2026. Trang tài nguyên ghi Published
  Oct. 24, 2013 và Version 1.0.0; lịch sử bổ sung dữ liệu năm 2018 không tự thay năm xuất
  bản của tài nguyên này. Nguồn: [trang dataset](https://physionet.org/content/sleep-edfx/1.0.0/).
- Bổ sung trích dẫn nền tảng PhysioNet do trang dataset hiện yêu cầu: Pollard và cộng sự,
  *Nature Health* 1(8), 792-795 (2026), [bài chính thức](https://www.nature.com/articles/s44360-026-00096-z).
  Đây là bài Comment về nền tảng dữ liệu, không phải bằng chứng hiệu năng phân loại.
- Rechtschaffen và Kales (1968): ghi vai trò **editors**, bổ sung cơ quan xuất bản đầy đủ
  và link [NLM Catalogue](https://www.ncbi.nlm.nih.gov/nlmcatalog/173471).
- AASM Manual v2.4: sửa tên đệm Troester thành **Matthew M. Troester** theo trang đầu
  bản manual. Không sao chép M. T. từ bài editorial về cập nhật manual: hai nguồn có
  trường tên không nhất quán. Trang đầu bản manual được đối chiếu ở bản số hóa trên
  một host bên thứ ba; không đưa host đó vào bibliography. Link trong paper là
  [thông báo phiên bản chính thức của AASM](https://aasm.org/resources/pdf/scoring-manual-update-april-2017.pdf),
  được ghi rõ là **Publisher's version announcement**, không giả là link toàn bộ sách.
- TCN: bổ sung DOI arXiv do DataCite đăng ký, giữ nhãn **arXiv preprint**.
- ICML calibration: bổ sung PMLR volume 70, series, publisher và
  [link proceedings chính thức](https://proceedings.mlr.press/v70/guo17a.html).
- Holm (1979): bổ sung [JSTOR stable link](https://www.jstor.org/stable/4615733).
  Không tự điền DOI chỉ dựa trên việc ID của JSTOR trông giống hậu tố DOI.
- Sách bootstrap: giữ năm 1994 theo edition gắn DOI đang dẫn và trang nhà xuất bản;
  chuẩn hóa publisher thành Chapman & Hall. Không đổi sang 1993 chỉ vì một số thư mục
  khác ghi năm đó cho bản/ấn bản khác. [Trang nhà xuất bản](https://www.routledge.com/link/link/p/book/9780412042317).
- Bảo vệ EEG và tên riêng trong BibTeX để style không biến chúng thành chữ thường.
  Giữ nguyên các tên model như ZleepAnlystNet, DeepSleepNet, SleepTransformer và ADAST.
- Phân biệt **article number** với **page range** ở sáu bài: 9859, 04TR01, e40211,
  104501, 104429 và zsaf063. Sửa dấu câu thừa sau tiêu đề kết thúc bằng dấu hỏi.
- Đưa trích dẫn Efron-Tibshirani về ngay phần bootstrap; Wilcoxon vẫn dẫn bài Wilcoxon.
  Việc renumber được thực hiện bởi BibTeX, không sửa tay số trong bài.

## Bảng đối chiếu từng reference đang xuất bản

Số dưới đây là số **sau** chỉnh sửa. Metadata DOI được đối chiếu với Crossref; riêng arXiv
là DataCite. Những nguồn sách/dataset hoặc metadata đăng ký không đủ được đối chiếu thêm
với trang nguồn chính thức, bài gốc lưu tại PMC/PubMed hoặc catalogue NLM.

| Số | Nguồn | Nơi đăng / xuất bản | Kết quả và link kiểm tra |
| --- | --- | --- | --- |
| 1 | Rosenberg & Van Hout, 2013 | Journal of Clinical Sleep Medicine 9(1), 81-87 | Đúng DOI/trang/năm; sửa tên và chữ hoa. [DOI](https://doi.org/10.5664/jcsm.2350) |
| 2 | ZleepAnlystNet, 2024 | Scientific Reports 14(1), 9859 | Đúng ba tác giả; 9859 là mã bài. Khóa nội bộ `jadhav2024zleepanlystnet` không phải tên tác giả và không xuất ra PDF. [DOI](https://doi.org/10.1038/s41598-024-60796-y) |
| 3 | Bai, Kolter & Koltun, 2018 | arXiv:1803.01271 | Preprint, không ghi như bài journal/conference đã phản biện. [arXiv](https://arxiv.org/abs/1803.01271) |
| 4 | ResNet, 2016 | IEEE CVPR, 770-778 | Đúng tác giả/trang/năm; là nguồn nền tảng ResNet, không tự chứng minh ResNet-1D tốt hơn cho EEG. [Bài chính thức](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) |
| 5 | DeepSleepNet, 2017 | IEEE TNSRE 25(11), 1998-2008 | Đúng tác giả, DOI, volume/issue/trang. [DOI](https://doi.org/10.1109/TNSRE.2017.2721116) |
| 6 | AttnSleep, 2021 | IEEE TNSRE 29, 809-818 | Đúng metadata; tên AttnSleep xuất hiện trong nội dung nhưng không thuộc tiêu đề chính thức. [DOI](https://doi.org/10.1109/TNSRE.2021.3076234) |
| 7 | SleepTransformer, 2022 | IEEE TBME 69(8), 2456-2467 | Đúng metadata; giữ dấu trong tên Chén. [DOI](https://doi.org/10.1109/TBME.2022.3147187) |
| 8 | Phan & Mikkelsen review, 2022 | Physiological Measurement 43(4), 04TR01 | Đúng DOI/năm; 04TR01 là mã bài. [DOI](https://doi.org/10.1088/1361-6579/ac6049) |
| 9 | SleepInceptionNet study, 2023 | Journal of Medical Internet Research 25, e40211 | Đúng năm/tác giả; dùng tiêu đề đầy đủ chính thức. [Bài chính thức](https://www.jmir.org/2023/1/e40211/) |
| 10 | He et al. cross-scenario, 2023 | BSPC 81, 104501 | Năm DOI chứa 2022 nhưng năm xuất bản volume là 2023, không phải lỗi. [Bài chính thức](https://www.sciencedirect.com/science/article/pii/S1746809422009557) |
| 11 | ADAST, 2023 | IEEE TETCI 7(1), 210-221 | Đúng metadata; không đổi năm journal sang năm đăng ký DOI 2022. [DOI](https://doi.org/10.1109/TETCI.2022.3189695) |
| 12 | Van Der Donckt et al., 2023 | BSPC 81, 104429 | Giữ tiêu đề đầy đủ gồm phụ đề; Crossref trả title ngắn nhưng publisher xác nhận bản đầy đủ. [Bài chính thức](https://www.sciencedirect.com/science/article/abs/pii/S1746809422008837) |
| 13 | Kemp et al., 2000 | IEEE TBME 47(9), 1185-1194 | Đúng metadata; giữ EEG và tên Oberyé. [DOI](https://doi.org/10.1109/10.867928) |
| 14 | Sleep-EDF Expanded | PhysioNet, v1.0.0 | Nguồn dataset có thật; sửa tác giả, kho lưu trữ và ngày truy cập. [Dataset](https://physionet.org/content/sleep-edfx/1.0.0/) |
| 15 | Pollard et al., 2026 | Nature Health 1(8), 792-795 | Nguồn nền tảng theo yêu cầu trích dẫn của PhysioNet; loại bài Comment. [Bài chính thức](https://www.nature.com/articles/s44360-026-00096-z) |
| 16 | Quan et al., 1997 | Sleep 20(12), 1077-1085 | Metadata Crossref thiếu nhiều trường; đối chiếu đủ tác giả/trang ở PubMed. [PubMed](https://pubmed.ncbi.nlm.nih.gov/9493915/) |
| 17 | Zhang et al. NSRR, 2018 | JAMIA 25(10), 1351-1358 | Sửa tên Kim/Mobley theo bài gốc. [Bài gốc](https://pmc.ncbi.nlm.nih.gov/articles/PMC6188513/) |
| 18 | Massimini et al., 2004 | Journal of Neuroscience 24(31), 6862-6870 | Đúng metadata; nguồn về sóng chậm, không phải chứng minh nguyên nhân lỗi N3 của nghiên cứu hiện tại. [DOI](https://doi.org/10.1523/JNEUROSCI.1318-04.2004) |
| 19 | Rechtschaffen & Kales, 1968 | NINDB / Neurological Information Network | Sách lịch sử về quy tắc scoring; sửa editors và publisher. [NLM](https://www.ncbi.nlm.nih.gov/nlmcatalog/173471) |
| 20 | AASM Manual v2.4, 2017 | American Academy of Sleep Medicine | Đúng phiên bản lịch sử; không gọi là phiên bản mới nhất. Link được ghi rõ là thông báo phiên bản. [AASM](https://aasm.org/resources/pdf/scoring-manual-update-april-2017.pdf) |
| 21 | Davidson et al. N3, 2025 | Sleep 48(10), zsaf063 | Đúng bốn tác giả/DOI/issue; không nhầm ngày online March với issue October. [DOI](https://doi.org/10.1093/sleep/zsaf063) |
| 22 | Saerens et al., 2002 | Neural Computation 14(1), 21-41 | Đúng metadata; nguồn label-prior adjustment. [DOI](https://doi.org/10.1162/089976602753284446) |
| 23 | Guo et al., 2017 | ICML / PMLR 70, 1321-1330 | Bổ sung series/volume/link; không tự tạo DOI khi proceedings không cung cấp. [PMLR](https://proceedings.mlr.press/v70/guo17a.html) |
| 24 | Efron & Tibshirani, 1994 | Chapman & Hall | Đúng edition gắn DOI đang dùng; citation được đặt lại đúng phần bootstrap. [DOI](https://doi.org/10.1201/9780429246593) |
| 25 | Wilcoxon, 1945 | Biometrics Bulletin 1(6), 80-83 | Crossref chỉ ghi trang đầu 80; giữ dải trang đầy đủ theo bài/issue gốc. [JSTOR](https://www.jstor.org/stable/3001968) |
| 26 | Holm, 1979 | Scandinavian Journal of Statistics 6(2), 65-70 | Đúng theo mục lục journal; bổ sung stable link. [Mục lục gốc](https://www.jstor.org/stable/i412579) |

## Ba mục có trong `.bib` nhưng không xuất ra PDF

- Rousseeuw (1987), silhouette: DOI `10.1016/0377-0427(87)90125-7`, metadata đúng.
- Ohayon et al. (2004), meta-analysis: DOI `10.1093/sleep/27.7.1255`, metadata đúng.
- Cliff (1993), dominance statistics: DOI `10.1037/0033-2909.114.3.494`, metadata đúng.

Giữ các mục này trong kho bibliography không làm danh mục PDF bị dư. Không ép thêm
trích dẫn Cliff: độ trội về dấu của các chênh lệch bắt cặp đang dùng không tự động đồng
nhất với mọi định nghĩa Cliff's delta cho hai mẫu độc lập.

## Chất lượng nơi đăng và cách sử dụng

- IEEE Transactions là nhóm tạp chí lưu trữ có phản biện; các tên journal trong paper
  khớp bản gốc, không dùng website bắt chước publisher. [IEEE](https://technav.ieee.org/topic/ieee-transactions/).
- Scientific Reports là journal của Nature Portfolio, không phải journal Nature.
  Đánh giá uy tín không cho phép coi mọi paper trên đó là bằng chứng quyết định.
  [Hướng dẫn tác giả chính thức](https://www.nature.com/srep/author-instructions).
- Sleep là journal có phản biện của Sleep Research Society và Australasian Sleep Association.
  [About SLEEP](https://academic.oup.com/sleep/pages/about).
- JMIR là journal có phản biện về y học số; Physiological Measurement là journal chuyên
  ngành của IOP/IPEM; BSPC là journal Elsevier phù hợp với signal processing y sinh.
  [JMIR](https://www.jmir.org/about-journal/focus-and-scope),
  [IOP](https://publishingsupport.iopscience.iop.org/journals/physiological-measurement/about-physiological-measurement/),
  [Elsevier](https://shop.elsevier.com/journals/biomedical-signal-processing-and-control/1746-8094).
- CVPR/ICML là nguồn proceedings đúng venue, khác với preprint. arXiv không phải
  journal; bài TCN vẫn có thể dùng cho xuất xứ kiến trúc nhưng phải ghi đúng loại tài liệu.
- AASM Manual là nguồn của tổ chức ban hành quy tắc; NLM Catalogue chỉ xác minh thông tin
  thư mục của sách 1968, không thay thế nội dung sách.

## Thứ tự, định dạng và giới hạn kiểm tra link

Danh mục theo **thứ tự trích dẫn đầu tiên**, không theo alphabet, năm hay thứ hạng venue.
Đã kiểm tra lệnh `cite`, thứ tự `bibitem` và số `bibcite` khớp cả 26 mục. Không cần xếp
paper nổi tiếng lên trước các nguồn được dẫn trước trong nội dung.

Giữ style numeric hiện có: tên journal đầy đủ nhất quán, tên tác giả viết tắt, DOI/URL
clickable, số bài được phân biệt với số trang. Đây là định dạng nhất quán cho bản thảo
hiện tại, **không khẳng định đã khớp từng dấu câu với style xuất bản cuối của BSPC**.
Guide BSPC live trả 403 trong lượt kiểm tra này; hướng dẫn Elsevier nêu phải đối chiếu
Guide của journal và phân biệt yêu cầu bản nộp ban đầu với bản accepted.
[Hướng dẫn Elsevier](https://www.elsevier.support/publishing/answer/how-should-i-prepare-the-references-in-my-manuscript).

Đã kiểm tra DOI/URL của mọi mục; không có link trả 404 trong phép resolve nguồn được dẫn.
Có trang trả 403, IEEE trả 202, và IOP trả trang kiểm tra trình duyệt dù status 200.
Các trạng thái này không chứng minh link hỏng hoặc nội dung toàn văn đã đọc được.
DOI đã resolve đúng publisher và metadata được đối chiếu độc lập. Không bảo đảm toàn văn
miễn phí: paywall và quyền truy cập tổ chức là vấn đề khác.

Chưa thực hiện tra cứu retraction toàn diện trên một registry chuyên biệt. Không thấy
thông báo bất thường ở nguồn đã mở; điều này không tương đương bảo đảm mọi reference
không từng có correction/retraction. Đặc biệt, Crossref `update-to` rỗng không đủ cho kết luận đó.

## Kiểm tra kỹ thuật sau sửa

- BibTeX không có warning/error; 26 mục được sử dụng.
- Hai PDF vẫn 14 trang, không undefined citation/reference hoặc overfull box.
- Đã render và xem toàn bộ 28 trang của hai bản, kiểm tra riêng trang References.
- Cả hai PDF chứa đủ 26 đích DOI/URL trong annotation clickable, không chỉ chữ URL.
- ZIP nộp bài gồm 12 entries đã được dựng độc lập; bản main 14 trang và supplement
  7 trang đều build thành công. Văn bản theo từng trang và bibliography main khớp
  bản bàn giao; khác biệt mã băm PDF do metadata build không được coi là khác nội dung.
- Các chuỗi số trong tám bảng, mã thí nghiệm và kết quả thống kê không thay đổi.
- Audit thô: `../tmp/reference_audit_20261001/remote_metadata.json` và
  `../tmp/reference_audit_20261001/structure_checks.json`.
- Không thay đổi protocol, checkpoint, prediction hoặc manifest thí nghiệm.
