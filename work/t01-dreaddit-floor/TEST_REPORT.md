task-id: t01-dreaddit-floor | vòng: 1

# TEST_REPORT — Tester (độc lập, tự chạy lại; dùng `.venv/bin/python`, từ gốc repo)

| AC | Lệnh | Exit | Kết quả thực | Kết luận |
|---|---|---|---|---|
| AC-1 | `.venv/bin/python -c "import sklearn,pandas,pyarrow,transformers,torch"` | 0 | import đủ. KHÔNG cài lại (.venv đã có sẵn; không chạy pip). | PASS |
| AC-2 | `python src/data.py` | 0 | train 2393 / val 445 / test 715 [đo đạc thực tế]; "grouped split leaks 0 posts; naive row split would leak 115"; "all checks passed" | PASS |
| AC-3 | `python src/baseline_tfidf.py val` | 0 | "all checks passed"; tfidf+logreg bal_acc 0.7431, majority 0.5000 | PASS |
| AC-4 | lệnh assert JSON nguyên văn trong PLAN | 0 | majority bal_acc=0.5 (acc 0.5056), tfidf 0.7431, n=445 cả hai, 2 dòng không trùng | PASS |
| AC-5 | `python src/data.py` (có khối âm tính) | 0 | xanh. Xem TEST BỔ SUNG: đảo guard thì FAIL (exit 1) | PASS |
| AC-6 | chạy lại `baseline_tfidf.py val` (exit 0) rồi lặp AC-4 | 0 | AC-4 vẫn exit 0; ablation.json vẫn đúng 2 dòng | PASS |
| AC-7 | `dataset.py`, `model.py`, `metrics.py`, `train.py --smoke` | 0,0,0,0 | đều in "all checks passed" (train smoke: loss 0.7067 -> 0.6457) | PASS |
| AC-8 | `git diff --stat feat/new-version...HEAD` + `git diff --stat feat/new-version` + `git status --short` | 0 | `...HEAD` RỖNG (chưa commit gì). Working tree: `M src/data.py` (+10), `M .gitignore` (+2), `?? requirements.txt`, `?? results/`, `?? work/`. **`.gitignore` KHÔNG nằm trong danh sách mục 2.** Diff: thêm 2 dòng `/my-docs ` (dấu cách cuối) và dòng trống. | **FAIL** (đúng chữ tiêu chí) |
| AC-9 | so 0.7431 với [đặc tả] 0.74 | - | lệch +0.0031, trong ±0.03 | PASS |
| AC-10 | `grep -nE "= *[0-9]" src/data.py src/baseline_tfidf.py` + đọc diff | - | Xem dưới | PASS |

## Ghi chú AC-8
Vi phạm phạm vi: `.gitignore` bị sửa ngoài mục 2. Mục 5 PLAN có nhắc dòng `/my-docs ` là "ngoài phạm vi, chỉ ghi nhận" nhưng diff cho thấy nó đã được THÊM (không phải đã tồn tại). Cần Leader quyết định: Coder hoàn tác, hoặc Leader nới mục 2 (tôi không tự nới). Ngoài ra `...HEAD` rỗng vì chưa commit; kiểm bằng working tree là phép thay thế.
Lưu ý: `src/data.py` diff đúng 10 dòng, chỉ trong khối `__main__`, không đổi hàm (đúng mục 2).

## AC-10
Hằng số có tên đầu module: `SEED=42`, `VAL_FRAC=0.15` (data.py); `SEED=42`, `MIN_DF=2`, `MAX_ITER=1000` (baseline_tfidf.py). `data.py:62 random_state=seed` là tham số, không phải literal. Diff của Coder (khối `__main__`) không có hằng số ảnh hưởng kết quả viết thẳng (chỉ dữ liệu giả `[1,2]`/`[2,3]`). Ghi nhận: `ngram_range`, `C` của LogReg không dùng literal khớp regex trong grep (chưa kiểm bằng đọc ngoài grep của diff; không thuộc diff Coder).

## TEST BỔ SUNG (mutation check)
Bản sao `.../scratchpad/mut/data.py` với `if shared:` -> `if False:` trong `assert_disjoint` (repo không bị sửa): chạy `python mut/data.py` -> in "assert_disjoint did NOT raise on a shared post_id", **exit 1**. Xác nhận AC-5 thật sự bắt được guard bị vô hiệu. `git status` sau đó không đổi.

## Bất thường khác
- Lần đầu tôi gõ nhầm `"src/train.py --smoke"` trong dấu ngoặc (lỗi của tôi, exit 2); chạy lại đúng thì exit 0.
- Cả tfidf và majority đều có `recorded` 2026-10-24; số liệu 0.7431 trùng khít [đặc tả] — đã chạy thật hai lần, kết quả ổn định.

## KẾT LUẬN: FAIL (vòng 1) — mọi AC kỹ thuật PASS, riêng AC-8 BẮT BUỘC FAIL do `.gitignore` ngoài phạm vi.
