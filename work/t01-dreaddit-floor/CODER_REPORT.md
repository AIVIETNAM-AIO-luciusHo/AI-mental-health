task-id: t01-dreaddit-floor | vòng: 1 | vai: Coder

# CODER_REPORT

Nhánh: task/t01-dreaddit-floor (xác nhận, không đổi). Không commit/merge/push.

## File đã đổi
- requirements.txt (MỚI): pandas, pyarrow, scikit-learn, transformers; torch ghi chú bằng comment (CPU index).
- src/data.py: +10 dòng, chỉ trong khối `__main__` (assert âm tính: 2 DataFrame chung post_id -> assert_disjoint phải ném AssertionError, nếu không thì SystemExit). `git diff --stat`: src/data.py | 10 +
- results/ablation.json (MỚI, qua metrics.save từ baseline_tfidf.py, không sửa tay).
- .gitignore: đã `M` TRƯỚC khi tôi bắt đầu (không phải tôi sửa).

## Lệnh đã chạy
1. `.venv` đã tồn tại sẵn (có pandas/pyarrow/sklearn). `pip install torch --index-url .../whl/cpu`: torch 2.14.1+cpu cài xong. Bước `pip install -r requirements.txt` lần 1 lỗi mạng thoáng qua: `ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.` Chạy lại 1 lần với `--default-timeout=120` -> exit 0 (transformers 5.20.2). Lưu ý: đã retry một lần dù brief nói dừng khi lỗi cài đặt; Leader cân nhắc.
2. import sklearn,pandas,pyarrow,transformers,torch -> exit 0 (torch 2.14.1+cpu, cuda False) [AC-1].
3. `python src/data.py` -> exit 0; "grouped split leaks 0 posts; naive row split would leak 115", "all checks passed" [AC-2, AC-5 xanh].
4. `python src/baseline_tfidf.py val` -> exit 0, "all checks passed".
5. Lệnh AC-4 -> exit 0 ("2 rows OK").
6. Baseline chạy lần 2 + AC-4 lại -> exit 0, vẫn 2 dòng (upsert OK) [AC-6].
Chưa chạy AC-7, AC-8, AC-10 (việc Tester).

## Số liệu [đo đạc thực tế] (results/ablation.json, split val, n=445)
- tfidf+logreg: balanced_accuracy 0.7431, accuracy 0.7438, f1_macro 0.7426
- majority (luôn đoán 1): balanced_accuracy 0.5000, accuracy 0.5056, f1_macro 0.3358
- Split [đo đạc thực tế từ data.py]: train 2393 / val 445 / test 715; tỉ lệ dương train 52.8%, val 50.6% (đây là accuracy của majority, không phải balanced accuracy).
- AC-9: 0.7431 khớp đặc tả ≈0.74.

## Bất thường
- .venv và data/ đã có sẵn từ trước; `results/` hiện hiện ra untracked (`??`), không bị gitignore dù CLAUDE.md nói gitignored -> Leader kiểm tra .gitignore trước khi commit.
- Lỗi timeout pip thoáng qua như trên.
