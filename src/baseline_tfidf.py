"""Classical baselines: the floor the two-tower VLM has to clear.

Roadmap 00_Master_Roadmap.md 2.1. `python src/baseline_tfidf.py` trains and records both
baselines in ~5 seconds on CPU.

INV-3: no transformer result is reportable without these numbers beside it, on the same
frozen split and the same metric. On ~2.4k rows of short Reddit text, TF-IDF + logistic
regression is a genuinely competitive baseline -- it is not a straw man, and if the VLM
cannot beat it that is the finding.
"""

import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from data import load_dreaddit, split_by_group
from metrics import evaluate, save, table

SEED = 42
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_ITER = 1000


def build_model():
    """TF-IDF over word 1-2 grams -> L2 logistic regression.

    `class_weight="balanced"` costs nothing on Dreaddit's 52/48 split but keeps the
    baseline honest if a more skewed dataset is swapped in later.
    """
    return make_pipeline(
        TfidfVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF, sublinear_tf=True),
        LogisticRegression(max_iter=MAX_ITER, class_weight="balanced", random_state=SEED),
    )


def main(split_name="val"):
    train, val = split_by_group(load_dreaddit("train"))
    holdout = val if split_name == "val" else load_dreaddit("test")

    if split_name == "test":
        # The test split is spent once, at the end, against the finished model.
        print("WARNING: scoring on test. Do this once, not while iterating.\n")

    # Majority class: the floor below which a model has learned nothing at all.
    majority = int(train["label"].mode()[0])
    save(evaluate(holdout["label"], [majority] * len(holdout), "majority", split_name,
                  notes=f"always predicts {majority}"))

    model = build_model()
    model.fit(train["text"], train["label"])
    pred = model.predict(holdout["text"])

    n_features = len(model.named_steps["tfidfvectorizer"].vocabulary_)
    result = evaluate(holdout["label"], pred, "tfidf+logreg", split_name,
                      notes=f"{NGRAM_RANGE[0]}-{NGRAM_RANGE[1]}grams, min_df={MIN_DF}, "
                            f"{n_features} features")
    save(result)

    print(table(split=split_name))
    print(f"\ntrained on {len(train)} rows, scored on {len(holdout)} ({split_name})")
    print(f"balanced accuracy {result['balanced_accuracy']:.4f} "
          f"vs majority floor {load_majority(split_name):.4f}")
    return result


def load_majority(split_name):
    from metrics import load
    return next(r["balanced_accuracy"] for r in load()
                if r["variant"] == "majority" and r["split"] == split_name)


if __name__ == "__main__":
    result = main(sys.argv[1] if len(sys.argv) > 1 else "val")

    # A baseline that cannot beat the floor is a broken baseline, not a finding.
    assert result["balanced_accuracy"] > 0.5, "TF-IDF failed to beat chance -- check the split"
    print("\nall checks passed")
