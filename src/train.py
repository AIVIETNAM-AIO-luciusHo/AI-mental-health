"""Training loop. Same code path locally and on Colab.

Roadmap 00_Master_Roadmap.md 2.1 / 2.4.

    python src/train.py --smoke      # offline, ~10s, tiny random weights, CPU
    python src/train.py              # real roberta-base; slow on CPU, meant for Colab

Colab:
    !git clone <repo> && cd <repo> && pip install -q torch transformers pandas pyarrow scikit-learn
    !python src/train.py --epochs 3 --batch-size 16
Data is fetched from HuggingFace and split from SEED=42, so nothing needs uploading and the
notebook trains on byte-identical rows to the local smoke test.

WHY A PLAIN LOOP AND NOT `transformers.Trainer`. Trainer would cover the text-only baseline
in fewer lines, but `TwoTowerVLM` does not return a loss and does not match its expected
signature -- adopting Trainer now means writing this loop anyway in two weeks. One loop,
swapped by `forward_fn`, serves both.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from data import load_dreaddit, split_by_group
from dataset import SessionDataset, make_collate
from metrics import evaluate, save

SEED = 42
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.1
MAX_GRAD_NORM = 1.0
CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def device_of(prefer_gpu=True):
    return torch.device("cuda" if prefer_gpu and torch.cuda.is_available() else "cpu")


def text_forward(model, batch):
    """HuggingFace sequence-classification head. Ignores pixel_values entirely."""
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits


def vlm_forward(model, batch):
    """TwoTowerVLM. Wired now so the fusion model needs no loop changes later."""
    return model(batch["input_ids"], batch["attention_mask"],
                 batch["pixel_values"], batch["frame_mask"])


@torch.inference_mode()
def predict(model, loader, forward_fn, device):
    model.eval()
    preds, golds = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = forward_fn(model, batch)
        preds.append(logits.argmax(-1).cpu())
        golds.append(batch["labels"].cpu())
    return torch.cat(golds).numpy(), torch.cat(preds).numpy()


def train(model, train_loader, val_loader, forward_fn, epochs, device,
          lr=LR, accum_steps=1, variant="text-only", results_path=None, save_ckpt=True):
    """Returns the best result dict. Checkpoints the best epoch by balanced accuracy."""
    model.to(device)
    # Only optimise what is unfrozen -- passing frozen params to AdamW still allocates
    # optimizer state for them, which is pure VRAM waste on a 6 GB card.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)

    total_steps = max(1, (len(train_loader) // accum_steps) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_FRAC), total_steps)
    loss_fn = nn.CrossEntropyLoss()

    use_amp = device.type == "cuda"
    best, losses = None, []

    for epoch in range(1, epochs + 1):
        model.train()
        running, t0 = 0.0, time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(forward_fn(model, batch), batch["labels"])
            # Scale so accumulated gradients average rather than sum.
            (loss / accum_steps).backward()
            running += loss.item()

            if step % accum_steps == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        losses.append(running / len(train_loader))
        secs = time.perf_counter() - t0

        y_true, y_pred = predict(model, val_loader, forward_fn, device)
        result = evaluate(y_true, y_pred, variant, "val",
                          notes=f"epoch {epoch}/{epochs}, lr={lr}, accum={accum_steps}")

        peak = f" | peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if use_amp else ""
        print(f"epoch {epoch}/{epochs} | loss {losses[-1]:.4f} | "
              f"bal_acc {result['balanced_accuracy']:.4f} | f1 {result['f1_macro']:.4f} | "
              f"{secs:.1f}s{peak}")

        if best is None or result["balanced_accuracy"] > best["balanced_accuracy"]:
            best = result
            if save_ckpt:
                CKPT_DIR.mkdir(parents=True, exist_ok=True)
                # Colab wipes the filesystem -- download this, do not rely on it persisting.
                torch.save({"model": model.state_dict(), "epoch": epoch, "result": result},
                           CKPT_DIR / f"{variant}.pt")

    if results_path is not None:
        save(best, results_path)
    else:
        save(best)
    return best, losses


def build_text_only(smoke=False):
    """roberta-base + classification head, or a tiny random stand-in for the smoke test."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if smoke:
        from dataset import _offline_tokenizer
        from transformers import RobertaConfig, RobertaForSequenceClassification
        # 514, not 64: the smoke run tokenizes REAL Dreaddit text, and Roberta offsets
        # position ids by padding_idx+1, so usable length is max_position_embeddings - 2.
        cfg = RobertaConfig(hidden_size=48, num_hidden_layers=1, num_attention_heads=2,
                            intermediate_size=37, vocab_size=100,
                            max_position_embeddings=514, num_labels=2)
        return RobertaForSequenceClassification(cfg), _offline_tokenizer()

    name = "FacebookAI/roberta-base"
    return (AutoModelForSequenceClassification.from_pretrained(name, num_labels=2),
            AutoTokenizer.from_pretrained(name))


def loaders(tokenizer, batch_size, smoke=False, num_workers=0):
    train_df, val_df = split_by_group(load_dreaddit("train"))
    if smoke:
        train_df, val_df = train_df.head(32), val_df.head(16)
    # image_size=32 for the text path: frames are ignored by text_forward, and full 224px
    # zeros would move ~600 KB per sample through the collate for nothing.
    mk = lambda df: SessionDataset(df, tokenizer, num_frames=1, image_size=32)
    collate = make_collate(tokenizer)
    return (DataLoader(mk(train_df), batch_size=batch_size, shuffle=True,
                       collate_fn=collate, num_workers=num_workers),
            DataLoader(mk(val_df), batch_size=batch_size, collate_fn=collate,
                       num_workers=num_workers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="offline CPU sanity run")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is present")
    args = ap.parse_args()

    set_seed()
    device = device_of(prefer_gpu=not args.cpu and not args.smoke)
    model, tokenizer = build_text_only(smoke=args.smoke)
    train_loader, val_loader = loaders(tokenizer, args.batch_size, smoke=args.smoke)

    epochs = min(args.epochs, 2) if args.smoke else args.epochs
    print(f"device {device} | {len(train_loader.dataset)} train / "
          f"{len(val_loader.dataset)} val | epochs {epochs}")

    if args.smoke:
        smoke_test(model, train_loader, val_loader, device, epochs)
        return

    best, _ = train(model, train_loader, val_loader, text_forward, args.epochs, device,
                    lr=args.lr, accum_steps=args.accum_steps, variant="text-only")
    print(f"\nbest val balanced accuracy {best['balanced_accuracy']:.4f} "
          f"({best['notes']})\nrecorded in results/ablation.json")


def smoke_test(model, train_loader, val_loader, device, epochs=2):
    """Verifies the loop mechanically: gradients flow, weights move, loss falls, metrics
    land. Does NOT check that the model learns anything useful -- 32 random-init rows
    cannot show that, and pretending otherwise would be a fake green test."""
    import tempfile

    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    tmp = Path(tempfile.mkdtemp()) / "ablation.json"

    best, losses = train(model, train_loader, val_loader, text_forward, epochs=epochs,
                         device=device, lr=1e-3, variant="smoke",
                         results_path=tmp, save_ckpt=False)

    after = [p for p in model.parameters() if p.requires_grad]
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), \
        "no parameter changed -- optimizer.step() is not reaching the weights"
    assert all(np.isfinite(l) for l in losses), f"non-finite loss: {losses}"
    assert set(best) >= {"balanced_accuracy", "f1_macro", "accuracy"}
    assert tmp.exists(), "results were not written"
    assert not (CKPT_DIR / "smoke.pt").exists(), "smoke run wrote a checkpoint"

    # A real gradient signal: 8 fixed rows, high LR, loss must fall.
    set_seed()
    batch = next(iter(train_loader))
    batch = {k: v[:8].to(device) for k, v in batch.items()}
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
    fn = nn.CrossEntropyLoss()
    first = last = None
    for i in range(20):
        opt.zero_grad(set_to_none=True)
        loss = fn(text_forward(model, batch), batch["labels"])
        loss.backward()
        opt.step()
        first = loss.item() if i == 0 else first
        last = loss.item()
    assert last < first, f"loss did not decrease when overfitting 8 rows: {first:.4f} -> {last:.4f}"

    print(f"\noverfit check: loss {first:.4f} -> {last:.4f} over 20 steps")
    print("all checks passed")


if __name__ == "__main__":
    main()
