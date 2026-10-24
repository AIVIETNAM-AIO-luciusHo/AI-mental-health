task-id: t01-dreaddit-floor | vòng: 1 | vai: Leader

# ACCEPTANCE — quyết định: MERGE (chờ người dùng cho phép commit + merge)

Không tin TEST_REPORT. Leader tự chạy lại: data.py, baseline_tfidf.py val, lệnh AC-4, dataset/model/metrics .py, train.py --smoke. Tất cả exit 0, "all checks passed"; AC-4 ok.

## Số đo [đo đạc thực tế] (results/ablation.json, split val, n=445)
- majority: balanced_accuracy 0.5000 (accuracy 0.5056)
- tfidf+logreg: balanced_accuracy 0.7431
- Split 2393/445/715; grouped split leak 0 post, naive leak 115 (in từ `python src/data.py`).
- "52.8 %" là tỉ lệ lớp dương của train (accuracy), KHÔNG phải balanced accuracy.
Giới hạn INV-1: Dreaddit không có trường tác giả, chỉ đảm bảo tách theo post_id.

## Xử lý AC-8 (Tester báo FAIL)
`.gitignore` đã ở trạng thái `M` từ trước khi Coder chạy (có trong git status đầu phiên, thay đổi của người dùng: thêm `/my-docs`). Không phải Coder sửa. Lỗi nằm ở tiêu chí: lệnh `git diff feat/new-version...HEAD` rỗng vì chưa commit, và mục 2 không tính thay đổi có sẵn của người dùng.
Sửa tiêu chí (b), lý do trên: AC-8 mới = `git status --short` chỉ gồm requirements.txt, src/data.py, results/, work/ cộng `.gitignore` đã có sẵn (loại khỏi commit của task). Với tiêu chí đã sửa: PASS.

## Diff
src/data.py: +10 dòng, chỉ trong `__main__`, không hằng số mới. requirements.txt hợp lệ. Không assert nào bị xóa/nới. Mutation check của Tester xác nhận AC-5 bắt được guard bị vô hiệu.

## Ghi nhận
- Retry cài đặt do timeout mạng: chấp nhận.
- results/ không bị gitignore; ablation.json là sổ cái, nên commit.
- Chưa làm: commit, merge vào feat/new-version, Daily_Notes (3 dòng) — làm sau khi người dùng cho phép.
