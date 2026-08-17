"""Ablation bookkeeping: one JSON file, one row per (variant, split).

Roadmap 00_Master_Roadmap.md 2.5. `python src/metrics.py` runs the self-check.

    from src.metrics import evaluate, save, table
    save(evaluate(y_true, y_pred, "tfidf+logreg", "val"))
    print(table())

Every model in the ablation -- TF-IDF, text-only, vision-only, late fusion,
cross-attention -- writes through here, so the comparison table cannot drift from the
numbers that were actually measured.

Re-running a variant REPLACES its row rather than appending a duplicate, so the file
always reflects the current state of each variant instead of its history.
"""

import datetime
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

RESULTS = Path(__file__).resolve().parent.parent / "results" / "ablation.json"

# INV-2: balanced accuracy leads. Accuracy is reported beside it because Dreaddit is
# near-balanced (~52% positive), which makes the two nearly equal -- stating both keeps
# that honest instead of implying the metric is doing work it is not.
COLUMNS = ["variant", "split", "n", "balanced_accuracy", "accuracy", "f1_macro"]


def evaluate(y_true, y_pred, variant, split, notes=""):
    """-> a result dict. Does not write anything; pass it to `save()`."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true vs {len(y_pred)} pred")
    if len(y_true) == 0:
        raise ValueError("cannot evaluate an empty split")
    return {
        "variant": variant,
        "split": split,
        "n": len(y_true),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "recorded": datetime.date.today().isoformat(),
        "notes": notes,
    }


def load(path=RESULTS):
    return json.loads(path.read_text()) if Path(path).exists() else []


def save(result, path=RESULTS):
    """Upsert on (variant, split). Returns all rows."""
    path = Path(path)
    rows = [r for r in load(path)
            if (r["variant"], r["split"]) != (result["variant"], result["split"])]
    rows.append(result)
    rows.sort(key=lambda r: (r["split"], -r["balanced_accuracy"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def table(path=RESULTS, split=None):
    """Markdown table, ready to paste into the roadmap."""
    rows = [r for r in load(path) if split is None or r["split"] == split]
    if not rows:
        return "_no results recorded yet_"
    head = f"| {' | '.join(COLUMNS)} |\n|{'---|' * len(COLUMNS)}"
    body = "\n".join(
        "| " + " | ".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in COLUMNS
        ) + " |"
        for r in rows
    )
    return f"{head}\n{body}"


if __name__ == "__main__":
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "ablation.json"

    perfect = evaluate([0, 1, 0, 1], [0, 1, 0, 1], "oracle", "val")
    assert perfect["balanced_accuracy"] == 1.0 and perfect["f1_macro"] == 1.0

    # Balanced accuracy must punish always-predicting-the-majority; accuracy does not.
    skewed_true, skewed_pred = [0] * 9 + [1], [0] * 10
    lazy = evaluate(skewed_true, skewed_pred, "majority", "val")
    assert lazy["accuracy"] == 0.9, lazy["accuracy"]
    assert lazy["balanced_accuracy"] == 0.5, lazy["balanced_accuracy"]

    save(perfect, tmp)
    save(lazy, tmp)
    assert len(load(tmp)) == 2

    # Re-running a variant replaces its row instead of duplicating it.
    save(evaluate([0, 1, 0, 1], [0, 1, 1, 1], "oracle", "val"), tmp)
    rows = load(tmp)
    assert len(rows) == 2, f"upsert failed, {len(rows)} rows"
    assert next(r for r in rows if r["variant"] == "oracle")["balanced_accuracy"] == 0.75

    # Same variant on a different split is a distinct row, not a replacement.
    save(evaluate([0, 1], [0, 1], "oracle", "test"), tmp)
    assert len(load(tmp)) == 3

    for bad in ([[0, 1], [0]], [[], []]):
        try:
            evaluate(*bad, "x", "val")
            raise SystemExit(f"FAIL: accepted bad input {bad}")
        except ValueError:
            pass

    assert "balanced_accuracy" in table(tmp)
    assert table(tmp, split="test").count("\n") == 2, "split filter not applied"
    assert "no results" in table(Path(tempfile.mkdtemp()) / "empty.json")

    print(table(tmp))
    print("all checks passed")
