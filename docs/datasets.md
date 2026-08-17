# Dataset Access — record

Companion to `AI_Engineer/Mental_Health_VLM_WBS.md` (`MHV-WBS-001` v2.0).

> [!important] Pivoted 2026-08-17 — §1 and §3 are cancelled, kept for the record
> This is now a **text-only portfolio project**. **Dreaddit and SDCNL (§2) are the primary
> datasets.** The DAIC-WOZ application (§1) and the IRB questions (§3) were cancelled to avoid
> EULA/ethics latency — see WBS `§0.0`. Sections 1 and 3 are retained only so the decision, and what
> it cost, stays legible.

Nothing in `data/` is committed (`.gitignore`).

---

## 1. ~~Multimodal~~ — CANCELLED 2026-08-17

*Kept for the record. Revisit only if the project ever goes multimodal (`P5`).*

### DAIC-WOZ — the standard set for this exact task

- Portal: <https://dcapswoz.ict.usc.edu/>
- Application: <https://dcapswoz.ict.usc.edu/daic-woz-database-download/> ("Apply Now DAIC-WOZ")
- Contents: **189 sessions**, 7–33 min each (avg 16). Transcripts + participant audio + **extracted
  facial features** per session. Labels: **PHQ-8**.
- Eligibility: academics and non-profit researchers only. **You must apply from your academic email
  address** — a personal Gmail is rejected. Complete → sign → submit the form.

### E-DAIC (Extended DAIC)

- Application: <https://dcapswoz.ict.usc.edu/extended-daic-database-download/>
- Used for the **AVEC 2019** challenge; depression *and* PTSD. USC notes its documentation is still
  being written, so expect thinner docs than DAIC-WOZ.
- Blank EULA to read before applying: <https://www.ihp-lab.org/downloads/Extended-DAIC-BLANK_EULA.pdf>

> [!important] Read before you build the loader
> DAIC-WOZ ships **pre-extracted facial features**, not raw video. That changes `P1.5`: the
> "sampling rate / face crop" decision may already be made for you by USC's feature pipeline. Confirm
> what the archive actually contains *before* writing frame-extraction code you don't need.

### CMU-MOSEI — open, no application

- Sentiment and emotion labels, **no clinical labels**. Useful for pretraining the fusion layer or
  sanity-checking the pipeline, **not** for a depression-screening claim.

---

## 2. PRIMARY DATASETS — Dreaddit & SDCNL

These are the four sets used by Mental-LLM (`docs/3643540.md`). Using the paper's own datasets makes
its reported numbers a **direct comparison**, not a loose analogy.

**Dreaddit is in use now** — fetched automatically by `src/data.py` from the HF parquet mirror
(`andreagasparini/dreaddit`, 2.2 MB, no auth). SDCNL is the planned second dataset (`P5`).

> [!note] Measured structure of Dreaddit (2026-08-17)
> 2,838 train / 715 test rows · **near-balanced, ~52 % positive** · no author field · `post_id`
> repeats up to 6× because ~100-token segments are windowed out of longer posts. The shipped
> train/test split is already post-disjoint (verified, 0 overlap). Full detail in WBS `§2.1`.

| Dataset | Task | Where |
|---|---|---|
| **Dreaddit** | binary stress | Kaggle `ruchi798/stress-analysis-in-social-media` (`dreaddit-train.csv`, 2.72 MB) · HF `andreagasparini/dreaddit` |
| **DepSeverity** | binary + **4-level** depression (DSM-5) | Same posts as Dreaddit, re-annotated — request from the authors |
| **SDCNL** | binary suicide ideation | <https://github.com/ayaanzhaque/SDCNL> — **CSVs committed in `data/`**, 1,895 posts (1,517 train / 379 test) |
| **CSSRS-Suicide** | binary + **5-level** suicide risk | 15 mental-health subreddits — request from the authors |

Reference implementation: <https://github.com/neuhai/Mental-LLM> (no data included, code only).

> [!warning] SDCNL labels are **subreddit-derived, not clinical**
> A post is labelled "suicidal" because it came from r/SuicideWatch. That is a proxy, and the paper
> itself builds a label-correction method precisely because the labels are noisy. Do not report an
> SDCNL number as a clinical result.

**Immediately available today: SDCNL + Dreaddit.** That is enough to build and validate the whole
Phase 1 loader and the Phase 2 text-only baseline while the DAIC application is pending.

---

## 3. ~~Ethics / IRB~~ — CANCELLED 2026-08-17

*Not required for a personal project on already-public data. Kept for the record; if this ever
becomes formal research or touches a real person's data, these are the questions to ask.*

1. Does secondary analysis of an **existing, de-identified, licensed** dataset (DAIC-WOZ) require IRB
   review at this institution, or does it qualify for exemption?
2. Does scraped **public social-media text** (Dreaddit/SDCNL) count as human-subjects research here?
   Policies differ sharply between institutions on exactly this point.
3. Is there a **safety/escalation** expectation if the system is ever demonstrated on a real person's
   data — even in a thesis demo?
4. Who signs the DAIC-WOZ EULA — you, or does it need a faculty PI signature? *Ask this first; it may
   sit on the critical path ahead of the application itself.*

Record the answers here when you get them.

---

## 4. Bias & reporting obligations (`X.1`)

Mental-LLM's limitations section flags known **racial and gender bias** in this task family.
Obligation carried into this project: report per-group metrics where demographics exist, and
**state their absence explicitly** where they don't. DAIC-WOZ demographic coverage is limited —
check what the archive actually provides before promising a subgroup breakdown.

---

## Sources

- [DAIC-WOZ Database (USC ICT)](https://dcapswoz.ict.usc.edu/)
- [Extended DAIC blank EULA](https://www.ihp-lab.org/downloads/Extended-DAIC-BLANK_EULA.pdf)
- [Mental-LLM repo (neuhai)](https://github.com/neuhai/Mental-LLM)
- [SDCNL repo (ayaanzhaque)](https://github.com/ayaanzhaque/SDCNL)
- [Dreaddit on Kaggle](https://www.kaggle.com/datasets/ruchi798/stress-analysis-in-social-media)
- [Dreaddit paper (ACL Anthology D19-6213)](https://aclanthology.org/D19-6213/)
