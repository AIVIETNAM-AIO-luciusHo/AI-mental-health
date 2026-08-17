"""Two-tower VLM: vision + language with cross-attention fusion.

Roadmap 00_Master_Roadmap.md Phase 2.3. `python src/model.py` runs the self-check.

    from src.model import build
    model = build()                      # roberta-base + vit-base, cross-attention
    logits = model(input_ids, attention_mask, pixel_values, frame_mask)

DESIGN NOTES (the non-obvious choices)

1. Frames -> K/V sequence. Each frame is encoded independently and reduced to its
   [CLS] token, giving (B, N_frames, d) as the key/value sequence. The alternative --
   keeping all patch tokens -- costs 197 tokens *per frame*; at 32 frames that is a
   6304-long K/V sequence and the attention matrix stops fitting anywhere sensible.
   Per-frame CLS is the cheap default. Revisit only if fusion underperforms.

2. Q from language, K/V from vision, per architecture-context.md §2. The model asks
   "given what was said, which moments in the face matter?" -- not the reverse.

3. `fusion="late"` is the control, not a fallback. Cross-attention has to beat it or
   it does not ship (roadmap 2.3). Same towers, same data, one flag apart, so the
   ablation is honest.

4. Towers are frozen by default. LoRA adapters attach to them later (roadmap 2.4);
   the fusion block and head always train fully.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

VISION_NAME = "google/vit-base-patch16-224"
LANGUAGE_NAME = "FacebookAI/roberta-base"

# INV-6: result-changing knobs named here, never inline.
NUM_HEADS = 12
DROPOUT = 0.1
NUM_LABELS = 2


def masked_mean(hidden, mask):
    """Mean over sequence positions, ignoring padding. (B, L, d), (B, L) -> (B, d)."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)


class TwoTowerVLM(nn.Module):
    """Encoders are injected so the self-check can build tiny random ones. Use `build()`
    for the real pretrained pair."""

    def __init__(self, vision, language, fusion="cross", num_labels=NUM_LABELS,
                 num_heads=NUM_HEADS, dropout=DROPOUT, freeze_encoders=True):
        super().__init__()
        if fusion not in ("cross", "late"):
            raise ValueError(f"fusion must be 'cross' or 'late', got {fusion!r}")

        self.vision, self.language, self.fusion = vision, language, fusion
        d_v = vision.config.hidden_size
        d_l = language.config.hidden_size

        # Towers need not share a width (and will not, if either is swapped out).
        self.vision_proj = nn.Linear(d_v, d_l)

        if fusion == "cross":
            if d_l % num_heads:
                raise ValueError(f"language hidden {d_l} not divisible by num_heads {num_heads}")
            self.attn = nn.MultiheadAttention(d_l, num_heads, dropout=dropout, batch_first=True)
            self.norm_q = nn.LayerNorm(d_l)
            self.norm_kv = nn.LayerNorm(d_l)
            self.norm_ff = nn.LayerNorm(d_l)
            self.ff = nn.Sequential(
                nn.Linear(d_l, d_l * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_l * 4, d_l)
            )
            head_in = d_l
        else:
            head_in = d_l * 2  # concat of the two pooled towers

        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, num_labels))

        if freeze_encoders:
            self.freeze_encoders()

    def freeze_encoders(self):
        """Roadmap 2.4: towers frozen, fusion + head train. LoRA re-enables a ~3% slice."""
        for p in self.vision.parameters():
            p.requires_grad_(False)
        for p in self.language.parameters():
            p.requires_grad_(False)

    def trainable_parameters(self):
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return n_train, sum(p.numel() for p in self.parameters())

    def encode_frames(self, pixel_values):
        """(B, N, 3, H, W) -> (B, N, d_l). Frames are flattened into the batch so the
        vision tower sees one big batch, never a Python loop."""
        b, n = pixel_values.shape[:2]
        flat = pixel_values.flatten(0, 1)
        cls = self.vision(pixel_values=flat).last_hidden_state[:, 0]  # [CLS] per frame
        return self.vision_proj(cls.view(b, n, -1))

    def forward(self, input_ids, attention_mask, pixel_values, frame_mask=None,
                return_attn=False):
        """frame_mask: (B, N), 1 = real frame, 0 = padding. Defaults to all-real."""
        text = self.language(input_ids=input_ids,
                             attention_mask=attention_mask).last_hidden_state
        frames = self.encode_frames(pixel_values)

        if frame_mask is None:
            frame_mask = torch.ones(frames.shape[:2], device=frames.device)

        # A row with every frame masked makes softmax average over all -inf -> NaN, and
        # the NaN then spreads through the loss with no error raised. Verified behaviour,
        # not a precaution. Checked here because every caller routes through forward().
        if (frame_mask.sum(1) == 0).any():
            bad = (frame_mask.sum(1) == 0).nonzero().flatten().tolist()
            raise ValueError(f"samples {bad} have no unmasked frames; cross-attention "
                             f"would return NaN. Give each sample >=1 valid frame.")

        if self.fusion == "late":
            pooled = torch.cat([masked_mean(text, attention_mask),
                                masked_mean(frames, frame_mask)], dim=-1)
            return self.head(pooled)

        # nn.MultiheadAttention's key_padding_mask is True == IGNORE -- inverted from
        # the HuggingFace convention. Getting this backwards silently attends to padding.
        fused, weights = self.attn(
            self.norm_q(text), self.norm_kv(frames), self.norm_kv(frames),
            key_padding_mask=(frame_mask == 0),
            need_weights=return_attn,
        )
        text = text + fused
        text = text + self.ff(self.norm_ff(text))
        logits = self.head(masked_mean(text, attention_mask))
        return (logits, weights) if return_attn else logits


def build(fusion="cross", vision_name=VISION_NAME, language_name=LANGUAGE_NAME, **kw):
    """Real pretrained pair. Downloads ~1 GB on first call."""
    return TwoTowerVLM(AutoModel.from_pretrained(vision_name),
                       AutoModel.from_pretrained(language_name), fusion=fusion, **kw)


def _tiny():
    """Randomly-initialised miniature towers -- lets the self-check run offline in a
    second. Deliberately mismatched widths (32 vs 48) to exercise vision_proj."""
    from transformers import RobertaConfig, RobertaModel, ViTConfig, ViTModel
    vision = ViTModel(ViTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                intermediate_size=37, image_size=32, patch_size=16))
    language = RobertaModel(RobertaConfig(hidden_size=48, num_hidden_layers=1,
                                          num_attention_heads=2, intermediate_size=37,
                                          vocab_size=100, max_position_embeddings=64))
    return vision, language


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, N = 2, 16, 4

    ids = torch.randint(5, 99, (B, L))
    attn_mask = torch.ones(B, L, dtype=torch.long)
    attn_mask[1, 12:] = 0                      # sample 1 has padded text
    px = torch.randn(B, N, 3, 32, 32)
    frame_mask = torch.ones(B, N)
    frame_mask[1, 2:] = 0                      # sample 1 has 2 padded frames

    for fusion in ("cross", "late"):
        vision, language = _tiny()
        model = TwoTowerVLM(vision, language, fusion=fusion, num_heads=2).eval()

        with torch.inference_mode():
            out = model(ids, attn_mask, px, frame_mask)
        assert out.shape == (B, NUM_LABELS), f"{fusion}: got {out.shape}"

        # Padded frames must not influence the result. This is the bug that hides:
        # an inverted key_padding_mask still produces plausible logits.
        px2 = px.clone()
        px2[1, 2:] = torch.randn(N - 2, 3, 32, 32) * 100
        with torch.inference_mode():
            out2 = model(ids, attn_mask, px2, frame_mask)
        assert torch.allclose(out, out2, atol=1e-5), f"{fusion}: padded frames leaked into output"

        n_train, n_total = model.trainable_parameters()
        assert n_train < n_total, f"{fusion}: encoders not frozen"
        for p in model.language.parameters():
            assert not p.requires_grad, f"{fusion}: language tower is trainable"

        # Gradients must actually reach the fusion/head, else training is a no-op.
        model.train()
        loss = model(ids, attn_mask, px, frame_mask).sum()
        loss.backward()
        head_w = model.head[1].weight
        assert head_w.grad is not None and head_w.grad.abs().sum() > 0, f"{fusion}: no head grad"
        if fusion == "cross":
            assert model.attn.out_proj.weight.grad.abs().sum() > 0, "no gradient into cross-attention"

        print(f"{fusion:>5}: logits {tuple(out.shape)} | "
              f"trainable {n_train:,}/{n_total:,} ({n_train / n_total:.1%})")

    # Attention weights are the demo artifact (roadmap 2.3): which frames each token used.
    vision, language = _tiny()
    model = TwoTowerVLM(vision, language, num_heads=2).eval()
    with torch.inference_mode():
        _, w = model(ids, attn_mask, px, frame_mask, return_attn=True)
    assert w.shape == (B, L, N), f"attention weights {w.shape}, expected {(B, L, N)}"
    assert torch.allclose(w[1, :, 2:].sum(), torch.tensor(0.0), atol=1e-6), \
        "masked frames received attention mass"
    print(f"attn : weights {tuple(w.shape)} | masked frames get 0 attention")

    print("all checks passed")
