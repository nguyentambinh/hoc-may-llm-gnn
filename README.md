# Ứng dụng LLM và GNN vào phát hiện lỗ hổng bảo mật trong mã nguồn

Đồ án Học máy của Nhóm 01, Viện Công nghệ số, Trường Đại học Thủ Dầu Một.

Nhóm xây dựng quy trình phân loại hàm trên R2Vul cho C, C#, Java, JavaScript và Python. Quy trình dùng Joern để sinh đồ thị thuộc tính mã nguồn, UniXcoder và Qwen2.5-Coder để biểu diễn mã, sau đó đánh giá 13 luồng thực nghiệm. Nhóm cũng triển khai giao diện demo và trực quan hóa đồ thị tri thức trong Gephi.

## Kết quả kiểm thử

| Cấu hình | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stacking (L13) | 75,84% | 70,14% | 88,74% | 78,35% | 84,49% | 82,45% |

Các chỉ số trong bảng tính trên 1.784 hàm kiểm thử có đồ thị. Trung bình theo năm ngôn ngữ, stacking đạt F1 macro 79,66% và AUROC macro 86,02%. Trên toàn bộ 1.838 hàm và cùng ngưỡng 0,5, LoRA đạt F1 macro 75,43%, cao hơn mốc CodeBERT/CLS 71,51% trong nghiên cứu R2Vul 3,92 điểm phần trăm.

## Mã nguồn

- `src/`, `scripts/`, `configs/` chứa phần xử lý, huấn luyện và đánh giá.
- `colab/` chứa notebook cho một số bước chạy GPU.
- `demo/` chứa giao diện và máy chủ demo.
- `tests/` chứa kiểm thử. Danh sách thư viện Python nằm trong `requirements.txt`.

Kho chứa mã nguồn. Bộ dữ liệu và trọng số mô hình được lưu riêng.
