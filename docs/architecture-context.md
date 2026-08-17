# 🏛️ Architecture Context: Mental Health VLM Cloud Platform
**Tags:** `#project`, `#VLM`, `#DeepLearning`, `#Cloud`, `#NLP`, `#ComputerVision`
**Trạng thái:** 🟢 Thiết kế kiến trúc lõi
**Mục tiêu:** Xây dựng hệ thống Vision-Language Model (VLM) trên nền tảng Cloud để phân tích trạng thái tâm lý (Emotion/Depression Analysis) thông qua việc dung hợp đa thể thức (Text + Image/Video Frames).

---

## 1. Tổng Quan Kiến Trúc Hệ Thống (System Overview)
Hệ thống chuyển dịch hoàn toàn lên môi trường Điện toán đám mây (Cloud Computing), áp dụng kiến trúc Microservices để đảm bảo khả năng mở rộng (scalability) và hiệu năng suy luận (inference performance).

* **Client-side (Frontend):** Web/Mobile App gửi luồng dữ liệu video và văn bản lên server qua WebSocket hoặc REST API.
* **Gateway & Load Balancer:** Nginx hoặc API Gateway để điều phối traffic.
* **VLM Inference Microservice:** Chạy trong các Docker container trên Cloud Instances có GPU (AWS EC2 G4dn hoặc GCP Compute Engine), phơi bày các API (FastAPI) để nhận dữ liệu và trả về vector trạng thái tâm lý.
* **Knowledge Graph Database:** Neo4j để lưu trữ Personal Knowledge Graph, hỗ trợ Context Retrieval cho RAG.

---

## 2. Kiến Trúc Học Sâu: Vision-Language Model (VLM)
Đây là "trái tim" của hệ thống. Thay vì phân tích riêng lẻ, ta sử dụng kiến trúc Two-Tower (Hai tháp) kết hợp với cơ chế Cross-Attention để VLM hiểu được sự tương quan giữa nét mặt và lời nói.

* **Vision Encoder (Bộ mã hóa hình ảnh):** Sử dụng **Vision Transformer (ViT)** hoặc **ResNet-50**. Ảnh tĩnh cắt từ video sẽ được chia thành các patch và nhúng (embed) thành các vector đặc trưng.
* **Language Encoder (Bộ mã hóa ngôn ngữ):** Sử dụng **RoBERTa** (tốt cho phân tích cảm xúc) hoặc các mô hình mã nguồn mở nhẹ như **Llama-3-8B**. Text input sẽ được token hóa và mã hóa.
* **Cross-Modal Fusion Layer (Lớp dung hợp đa thể thức):** Đây là nơi phép màu xảy ra. Ta sử dụng cơ chế Cross-Attention để "dạy" mô hình biết nên tập trung vào từ ngữ nào khi nét mặt thay đổi. 

Công thức toán học cốt lõi của cơ chế Cross-Attention được biểu diễn bằng LaTeX như sau:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
Trong đó, $Q$ (Query) là dữ liệu từ tháp ngôn ngữ, còn $K$ (Key) và $V$ (Value) là dữ liệu từ tháp hình ảnh (hoặc ngược lại). Quá trình này giúp mô hình tính toán ma trận trọng số tương quan, kết hợp hai không gian vector riêng biệt thành một biểu diễn duy nhất cho trạng thái tâm lý.

---

## 3. Thuật Toán Tối Ưu Hóa (Optimization Algorithms)
Việc huấn luyện (Fine-tuning) một VLM đòi hỏi tài nguyên tính toán khổng lồ. Để một sinh viên có thể chạy trên Cloud mà không bị "cháy túi", em bắt buộc phải dùng các kỹ thuật tối ưu sau:

* **AdamW Optimizer:** Đây là thuật toán tiêu chuẩn cho các mô hình Transformer, kết hợp giữa Adam và Weight Decay để tránh over-fitting, giúp mô hình hội tụ nhanh hơn trên tập dữ liệu y khoa.
* **LoRA (Low-Rank Adaptation) / QLoRA:** Kỹ thuật Fine-tuning hiệu quả về mặt tham số (PEFT). Thay vì cập nhật toàn bộ hàng tỷ tham số của VLM, LoRA chỉ đóng băng mô hình gốc và chèn thêm các ma trận phân hạng thấp (low-rank matrices). Em có thể giảm lượng VRAM cần thiết từ 32GB xuống chỉ còn khoảng 8GB-12GB.
* **Mixed Precision Training (Huấn luyện độ chính xác hỗn hợp):** Sử dụng FP16 (Float16) thay vì FP32 (Float32) cho các phép tính lan truyền ngược (backpropagation), giúp tăng tốc độ huấn luyện lên gấp đôi và giảm một nửa bộ nhớ.

---

## 4. Ngăn Xếp Công Nghệ (Tech Stack) Cần Học
* **Core Deep Learning:** PyTorch (Bắt buộc) và thư viện Hugging Face (Transformers, Diffusers, PEFT).
* **Backend & MLOps:** Python (FastAPI để làm API siêu tốc), Docker (đóng gói mô hình), Triton Inference Server (tối ưu hóa việc gọi model trên GPU).
* **Cloud Infrastructure:** Làm quen với AWS (SageMaker, EC2, S3) hoặc Google Cloud Platform (Vertex AI, Cloud Run).
* **Version Control cho Dữ liệu:** DVC (Data Version Control) kết hợp với Git để quản lý các tập dataset tâm lý học.
