"""Dreaddit loader + leakage-safe splitter.

Task (MHV-WBS-001 §1.P0.4): binary screening from text alone.
`python src/data.py` runs the self-check.

LEAKAGE UNIT (INV-1). Dreaddit ships no author field -- Reddit usernames were not
released. So "subject-disjoint" is not literally achievable here: two posts by the
same person are indistinguishable from two posts by different people.

What IS achievable, and what this module enforces, is *post*-disjointness. Dreaddit
segments ~100-token windows out of longer posts, so one post yields up to 6 rows
(measured: 2838 rows from 2343 posts in train). Those rows share wording and are
near-duplicates. Splitting them at random is the leak that actually happens here.

The residual author-level risk is unfixable with this dataset and is reported as a
limitation rather than papered over.
"""

from pathlib import Path
import urllib.request

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "dreaddit"
URL = ("https://huggingface.co/datasets/andreagasparini/dreaddit/"
       "resolve/main/data/{split}-00000-of-00001.parquet")

# INV-6: result-changing knobs live here, named, not buried inline.
SEED = 42
VAL_FRAC = 0.15
GROUP = "post_id"
KEEP = ["text", "label", GROUP, "subreddit"]  # subreddit retained for X.1 subgroup metrics


def load_dreaddit(split="train"):
    """-> DataFrame[text, label, post_id, subreddit]. Downloads on first use.

    `split` is "train" or "test"; the shipped test split is verified post-disjoint
    from train (checked in the self-check below), so it is used as-is.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    path = DATA_DIR / f"{split}.parquet"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(split=split), path)

    df = pd.read_parquet(path, columns=KEEP)
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 0]
    return df.reset_index(drop=True)


def split_by_group(df, val_frac=VAL_FRAC, seed=SEED):
    """Split off a validation set with no `post_id` shared across the boundary.

    Stratifies on label as far as grouping allows -- GroupShuffleSplit balances
    group sizes, not classes, so the resulting balance is checked, not assumed.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, df["label"], groups=df[GROUP]))
    train, val = df.iloc[train_idx], df.iloc[val_idx]
    assert_disjoint(train, val, "train", "val")
    return train.reset_index(drop=True), val.reset_index(drop=True)


def assert_disjoint(a, b, name_a="a", name_b="b"):
    """INV-1 enforcement. Unconditional -- a hard rule, not a test-only check."""
    shared = set(a[GROUP]) & set(b[GROUP])
    if shared:
        raise AssertionError(
            f"{GROUP} leak: {len(shared)} group(s) in both {name_a} and {name_b}, "
            f"e.g. {sorted(shared)[:3]}"
        )


def describe(df, name):
    """P1.4: class balance is printed, never assumed."""
    n, pos = len(df), df["label"].mean()
    print(f"{name:>6}: {n:5d} rows | {df[GROUP].nunique():5d} posts | "
          f"positive {pos:.1%} | majority-class baseline {max(pos, 1 - pos):.1%}")


if __name__ == "__main__":
    train_full, test = load_dreaddit("train"), load_dreaddit("test")
    train, val = split_by_group(train_full)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        describe(part, name)

    # The shipped split is only usable as a test set if it is already post-disjoint.
    assert_disjoint(train_full, test, "train_full", "test")
    assert_disjoint(train, test, "train", "test")
    assert_disjoint(val, test, "val", "test")

    assert len(train) + len(val) == len(train_full), "rows lost in split"
    assert set(train_full["label"]) == {0, 1}, "labels are not binary 0/1"
    assert train_full["text"].notna().all(), "null text survived loading"
    assert 0.05 < val["label"].mean() < 0.95, "val split collapsed to one class"

    # The leak this module exists to prevent: random row splitting tears posts apart.
    from sklearn.model_selection import train_test_split
    naive_tr, naive_va = train_test_split(train_full, test_size=VAL_FRAC, random_state=SEED)
    leaked = len(set(naive_tr[GROUP]) & set(naive_va[GROUP]))
    print(f"\ngrouped split leaks 0 posts; naive row split would leak {leaked}")
    assert leaked > 0, "no leak to prevent -- re-check that posts really do span rows"

    print("all checks passed")
