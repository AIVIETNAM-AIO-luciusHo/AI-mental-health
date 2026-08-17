"""PyTorch Dataset + collate for the two-tower model.

Roadmap 00_Master_Roadmap.md Phase 2.1. `python src/dataset.py` runs the self-check
(offline, ~1s -- it builds a throwaway tokenizer rather than downloading one).

    from src.data import load_dreaddit, split_by_group
    from src.dataset import SessionDataset, make_collate

    train, val = split_by_group(load_dreaddit("train"))
    ds = SessionDataset(train, tokenizer)
    dl = DataLoader(ds, batch_size=8, collate_fn=make_collate(tokenizer))

DYNAMIC PADDING. `__getitem__` tokenizes without padding; the collate pads each batch to
its own longest sequence via `tokenizer.pad()`. Padding to a fixed 512 instead would waste
most of every batch -- Dreaddit segments are ~100 tokens, so a fixed width does ~5x the
attention work for identical results.

VISION IS A PLACEHOLDER. `pixel_values` is zeros until a facial-affect dataset lands
(roadmap 2.2, the open blocker). Shapes, masking and the collate path are all real, so
swapping zeros for decoded frames touches `_frames()` and nothing else.

Frames are marked VALID (mask 1) even though they are zeros. Marking them invalid would
leave rows with no unmasked frames, which makes cross-attention emit NaN -- see the guard
in model.forward(). Zero pixels give a constant vision embedding, which is harmless.
"""

from functools import partial

import torch
from torch.utils.data import Dataset

IMAGE_SIZE = 224
MAX_LENGTH = 512
NUM_FRAMES = 1  # dummy count; raise once real frames exist (1 fps, cap 32 -- roadmap 1.3)


class SessionDataset(Dataset):
    """Wraps the DataFrame from `src.data.load_dreaddit`."""

    def __init__(self, df, tokenizer, max_length=MAX_LENGTH, num_frames=NUM_FRAMES,
                 image_size=IMAGE_SIZE):
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1; see the NaN guard in model.forward()")
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def _frames(self, row):
        """Real frame decoding replaces this. Returns (N, 3, H, W)."""
        return torch.zeros(self.num_frames, 3, self.image_size, self.image_size)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        enc = self.tokenizer(row["text"], truncation=True, max_length=self.max_length)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "pixel_values": self._frames(row),
            "label": int(row["label"]),
        }


def collate(batch, tokenizer):
    """Dynamic padding on both modalities.

    Text -> padded to the batch's longest sequence.
    Frames -> padded to the batch's largest frame count, with `frame_mask` marking real
    frames. Clips are equal-length today; they will not be once video is real.
    """
    text = tokenizer.pad(
        [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch],
        return_tensors="pt",
    )

    frames = [b["pixel_values"] for b in batch]
    counts = [f.shape[0] for f in frames]
    if min(counts) < 1:
        raise ValueError("every sample needs >=1 frame; a fully-masked row NaNs cross-attention")

    n_max = max(counts)
    pixel_values = torch.zeros(len(batch), n_max, *frames[0].shape[1:])
    frame_mask = torch.zeros(len(batch), n_max)
    for i, f in enumerate(frames):
        pixel_values[i, : f.shape[0]] = f
        frame_mask[i, : f.shape[0]] = 1

    return {
        "input_ids": text["input_ids"],
        "attention_mask": text["attention_mask"],
        "pixel_values": pixel_values,
        "frame_mask": frame_mask,
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def make_collate(tokenizer):
    """DataLoader(collate_fn=make_collate(tok))."""
    return partial(collate, tokenizer=tokenizer)


def _offline_tokenizer():
    """A real HuggingFace tokenizer with a throwaway vocab -- keeps the self-check offline
    while still exercising the genuine `.pad()` path rather than a stub."""
    import os
    import tempfile

    from transformers import BertTokenizer

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"w{i}" for i in range(50)]
    path = os.path.join(tempfile.mkdtemp(), "vocab.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(vocab))
    return BertTokenizer(vocab_file=path)


if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader

    tok = _offline_tokenizer()
    df = pd.DataFrame({
        "text": ["w1 w2", "w1 w2 w3 w4 w5 w6 w7 w8", "w3", "w4 w5 w6"],
        "label": [0, 1, 1, 0],
    })
    ds = SessionDataset(df, tok, num_frames=2, image_size=32)
    assert len(ds) == 4

    batch = collate([ds[0], ds[1]], tok)
    assert batch["input_ids"].shape[0] == 2
    assert batch["pixel_values"].shape == (2, 2, 3, 32, 32), batch["pixel_values"].shape
    assert batch["frame_mask"].shape == (2, 2)
    assert (batch["frame_mask"] == 1).all(), "dummy frames must stay valid or attention NaNs"
    assert batch["labels"].tolist() == [0, 1]

    # Dynamic padding: a short batch must not be padded out to a long one's width.
    short = collate([ds[0], ds[2]], tok)
    long = collate([ds[1], ds[3]], tok)
    assert short["input_ids"].shape[1] < long["input_ids"].shape[1], \
        "batches padded to a fixed width -- dynamic padding is not working"
    assert short["input_ids"].shape[1] < MAX_LENGTH, "padded to max_length instead of batch max"

    # Padding is masked, and padded positions carry the pad id.
    pad_row = long["attention_mask"][1]
    assert pad_row.sum() < pad_row.numel(), "expected padding in the shorter row"
    assert (long["input_ids"][1][pad_row == 0] == tok.pad_token_id).all()

    # Variable frame counts: the shorter clip is padded and masked, not silently dropped.
    mixed = collate([{**ds[0], "pixel_values": torch.zeros(1, 3, 32, 32)}, ds[1]], tok)
    assert mixed["pixel_values"].shape[1] == 2
    assert mixed["frame_mask"].tolist() == [[1, 0], [1, 1]]

    try:
        collate([{**ds[0], "pixel_values": torch.zeros(0, 3, 32, 32)}], tok)
        raise SystemExit("FAIL: zero-frame sample was accepted")
    except ValueError:
        pass

    # End to end through a DataLoader and the real model, on tiny random towers.
    from model import TwoTowerVLM, _tiny

    dl = DataLoader(ds, batch_size=2, collate_fn=make_collate(tok), shuffle=False)
    vision, language = _tiny()
    model = TwoTowerVLM(vision, language, num_heads=2).eval()
    for b in dl:
        with torch.inference_mode():
            logits = model(b["input_ids"], b["attention_mask"], b["pixel_values"], b["frame_mask"])
        assert logits.shape == (b["labels"].shape[0], 2)
        assert not logits.isnan().any(), "NaN logits"

    print(f"batches ok | dynamic pad {short['input_ids'].shape[1]} vs "
          f"{long['input_ids'].shape[1]} tokens | frames padded + masked")
    print("all checks passed")
