ỨNG DỤNG LLM VÀ GNN VÀO PHÁT HIỆN LỖ HỔNG BẢO MẬT TRONG MÃ NGUỒN

Đồ án Học máy của Nhóm 01, Viện Công nghệ số, Trường Đại học Thủ Dầu Một.
Thành viên: Nguyễn Tâm Bình, Nguyễn Hoàng Thiên, Vũ Đức Thắng.

Nhóm xây dựng quy trình phân loại hàm trên R2Vul cho C, C#, Java, JavaScript và Python. Quy trình dùng Joern để sinh đồ thị thuộc tính mã nguồn, UniXcoder và Qwen2.5-Coder để biểu diễn mã, sau đó đánh giá 13 luồng thực nghiệm. Nhóm cũng triển khai giao diện demo và trực quan hóa đồ thị tri thức trong Gephi.

1. Liên kết
- Báo cáo (Google Drive): https://drive.google.com/drive/folders/1ISh0kwqz3SeBzCN0M9EECsXkakkYDOQq?usp=sharing
- Mã nguồn (GitHub): https://github.com/nguyentambinh/hoc-may-llm-gnn

2. Kết quả kiểm thử
- Stacking (L13) trên 1.784 hàm có đồ thị: Accuracy 75,84%, Precision 70,14%, Recall 88,74%, F1 78,35%, AUROC 84,49%, AUPRC 82,45%.
- Trung bình macro theo 5 ngôn ngữ: F1 macro 79,66%, AUROC macro 86,02%.
- Trên toàn bộ 1.838 hàm kiểm thử ở ngưỡng 0,5: LoRA đạt F1 macro 75,43%, cao hơn mốc CodeBERT/CLS (71,51%) trong nghiên cứu R2Vul 3,92 điểm phần trăm.

3. Mã nguồn
- src/, scripts/, configs/: xử lý dữ liệu, sinh đồ thị, huấn luyện và đánh giá.
- colab/: notebook cho các bước chạy GPU.
- demo/: giao diện và máy chủ demo.
- tests/: kiểm thử. Danh sách thư viện Python nằm trong requirements.txt.

Kho chứa mã nguồn. Bộ dữ liệu và trọng số mô hình được lưu riêng.
