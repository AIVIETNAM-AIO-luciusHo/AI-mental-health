task-id: t01-dreaddit-floor | vòng: 1 | vai: Leader | trạng thái: CHỜ NGƯỜI DÙNG DUYỆT

# PLAN — t01-dreaddit-floor

Nhánh: `task/t01-dreaddit-floor` (tách từ `feat/new-version` @ `fc9a1fa`). Chỉ Leader merge.

## 1. Mục tiêu và phạm vi KHÔNG làm

**Mục tiêu.** Dựng môi trường chạy được từ bản clone sạch, rồi đo và ghi vào `results/ablation.json`
hai mốc sàn của Dreaddit trên split val đã đóng băng: **majority** và **TF-IDF+LogReg**. Đây là mốc INV-3
mà mọi kết quả Transformer sau này phải đứng cạnh. Hiện repo chưa có `.venv`, `data/` hay `results/`, nên
chưa có con số nào được đo ở bản này.

**KHÔNG làm** (YAGNI/INV-5):
- Không đụng `model.py`, `train.py`, `dataset.py`, tháp thị giác, LoRA, SDCNL, MOSEI, FER-2013.
- Không chạy bất kỳ train transformer nào (việc của Colab, task sau).
- Không chấm trên split **test** (chỉ dùng một lần, ở cuối).
- Không sửa logic `baseline_tfidf.py` / `metrics.py`. Không thêm test framework, DVC, CI.
- Không sửa tay `results/ablation.json`.

## 2. File được phép sửa

| File | Quyền |
|---|---|
| `requirements.txt` | TẠO MỚI: pandas, pyarrow, scikit-learn, transformers, (torch ghi chú cài bằng CPU index) |
| `src/data.py` | CHỈ thêm assert âm tính vào khối `__main__` (AC-5). Không đổi hàm. |
| `results/ablation.json` | CHỈ qua `metrics.save()` do `baseline_tfidf.py` gọi |
| `work/t01-dreaddit-floor/*` | báo cáo, ghi chú |

Sửa file khác = vi phạm → REWORK. `.venv/`, `data/` không commit (đã ignore).

## 3. Giao diện và điểm có thể sai âm thầm

Không có hàm mới. Giao diện hiện có, dùng nguyên:
`load_dreaddit(split) -> DataFrame[text,label,post_id,subreddit]`;
`split_by_group(df) -> (train, val)`; `assert_disjoint(a, b)` raise `AssertionError`;
`evaluate(y_true, y_pred, variant, split, notes) -> dict`; `save(result)` upsert theo `(variant, split)`.

Điểm dễ sai âm thầm:
1. **Majority floor có hai nghĩa.** Bộ phân loại luôn đoán một lớp có balanced accuracy = **0.5 đúng** (INV-2),
   còn con số "52.8 %" trong WBS là tỉ lệ lớp dương (accuracy). Báo cáo phải ghi rõ cột nào. Đừng gọi 0.528 là
   balanced accuracy.
2. **Rò rỉ split (INV-1).** Chỉ có `post_id`; không có tác giả. Không được bịa khóa nhóm. Rò rỉ mức tác giả là
   hạn chế đã biết, phải nêu trong báo cáo.
3. **Nhãn trong majority** lấy từ mode của *train*, áp lên val. Không lấy từ val.
4. **Upsert** theo `(variant, split)`: chạy lại ghi đè dòng cũ, đúng ý đồ; kiểm tra không sinh dòng trùng.
5. Cả hai dòng phải cùng split `val` (INV-3).

## 4. BẢNG TIÊU CHÍ NGHIỆM THU

Chạy từ gốc repo, trong `.venv` đã cài từ `requirements.txt`. Mốc theo `[đặc tả]` lấy từ WBS/roadmap, chưa phải số đo.

| ID | Lệnh | Kết quả mong đợi | Loại | Mức |
|---|---|---|---|---|
| AC-1 | `pip install -r requirements.txt` trong `.venv` mới, rồi `python -c "import sklearn,pandas,pyarrow,transformers,torch"` | exit 0 | chức năng | BẮT BUỘC |
| AC-2 | `python src/data.py` | exit 0, in `all checks passed`; số dòng train+val = train đầy đủ; in dòng "grouped split leaks 0 posts" | bất biến (INV-1) | BẮT BUỘC |
| AC-3 | `python src/baseline_tfidf.py val` | exit 0, in `all checks passed` | chức năng | BẮT BUỘC |
| AC-4 | `python -c "import json;r=json.load(open('results/ablation.json'));d={(x['variant'],x['split']):x for x in r};m=d[('majority','val')];t=d[('tfidf+logreg','val')];assert m['balanced_accuracy']==0.5;assert t['balanced_accuracy']>0.6;assert m['n']==t['n']==445;assert len(r)==len(d)"` | exit 0: majority = 0.5 đúng; TF-IDF > 0.6; cùng n=445; không dòng trùng | bất biến (INV-2, INV-3) | BẮT BUỘC |
| AC-5 | `python src/data.py` sau khi thêm assert âm tính: dựng 2 DataFrame cùng một `post_id`, gọi `assert_disjoint`, kỳ vọng `AssertionError` | khối `__main__` raise nếu `assert_disjoint` KHÔNG ném lỗi; chạy thật vẫn xanh | âm tính (INV-1) | BẮT BUỘC |
| AC-6 | `python src/baseline_tfidf.py val` chạy lần 2 rồi lặp lệnh AC-4 | vẫn exit 0, vẫn đúng 2 dòng val (upsert, không nhân đôi) | hồi quy | BẮT BUỘC |
| AC-7 | `python src/dataset.py && python src/model.py && python src/metrics.py && python src/train.py --smoke` | cả 4 exit 0, in `all checks passed` | hồi quy | BẮT BUỘC |
| AC-8 | `git diff --stat feat/new-version...HEAD` | chỉ gồm các file ở mục 2 | bất biến (phạm vi) | BẮT BUỘC |
| AC-9 | So `tfidf+logreg` val balanced_accuracy với `[đặc tả]` ≈ 0.74 (bản tham chiếu khác) | trong ±0.03; lệch hơn thì ghi chú nguyên nhân, không tự nới tiêu chí | chức năng | NÊN |
| AC-10 | `grep -nE "= *[0-9]" src/data.py src/baseline_tfidf.py` đối chiếu với hằng số có tên đầu module | không có hằng số ảnh hưởng kết quả mới viết thẳng trong code | bất biến (INV-6) | NÊN |

Nhãn trong báo cáo: mọi con số chỉ được gắn **[đo đạc thực tế]** nếu trích được từ `results/ablation.json` hoặc
một lệnh trong bảng trên; 0.7431 và 52.8 % từ tài liệu cũ là **[đặc tả]**.

## 5. Rủi ro và giả định

- Chưa có `.venv`, `data/`, `results/`; sklearn chưa cài ở Python hệ thống. [chưa hiện thực]
- Lần chạy đầu tải parquet Dreaddit từ HuggingFace (~2.2 MB), cần mạng. Nếu HF đổi URL, task sẽ FAIL, báo ESCALATE. [đặc tả]
- Cài torch+transformers có thể nặng; dùng wheel CPU (`--index-url https://download.pytorch.org/whl/cpu`), máy có 8 GB RAM. [đặc tả]
- Con số 0.74 và split 2,393/445/715 đến từ tài liệu cũ, chưa kiểm ở bản này. [chưa hiện thực]
- `.gitignore` có dòng `/my-docs ` (dấu cách cuối) có thể không khớp; ngoài phạm vi, chỉ ghi nhận.

## 6. Bước kế tiếp cho Tester

Tester chạy lần lượt AC-1…AC-10, ghi `TEST_REPORT.md` (task-id, vòng, lệnh, stdout rút gọn, PASS/FAIL từng AC).
**Tester được thêm test ngoài bảng** nếu nó kiểm chứng được bằng máy và nằm trong file ở mục 2; phải ghi rõ
là test bổ sung. Tester không được nới lỏng hoặc xóa tiêu chí trong bảng.
Nếu round ≥ 3 vẫn FAIL, Leader chọn: tách task / sửa tiêu chí (ghi lý do) / đổi cách tiếp cận / ESCALATE.
