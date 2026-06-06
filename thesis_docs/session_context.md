# DeepPEF — Session Context (For New Chat Continuation)

**Last updated:** June 6, 2026 (evening session)

## LATEST DECISIONS (this session):
- Created thesis_docs/ folder with PowerPoint (13 slides) + analysis doc + this file
- Deep research confirmed: NO method gets >0.65 without pretrained backbone EXCEPT DDMut (anti-symmetry + Siamese)
- ProteinMPNN weights are PUBLIC (could replicate ThermoMPNN directly for 0.75)
- Plan: GNN-SM + anti-symmetry + serial fusion + ensemble = target 0.60-0.68
- Data filtering is intentional (professor approved, quality over quantity)
- User demands: consistent probability estimates, no bullshit, hard work is fine, evolution approach
- Research complete: DDMut NOT truly from-scratch (heavy feature engineering), ProteinMPNN weights PUBLIC
- Anti-symmetry works on existing data (no extra data needed), automatic with subtract-mut output
- One-hot+embeddings: recommended nn.Embedding(20,64) + projected PLM addition
- Inverse folding pretraining NOT feasible (368 proteins too few), but ESM-IF1 features already provide this
- Created levels_progression.md showing all experiment levels with learning from each
- Created DeepPEF_Evolution_Training.ipynb — comprehensive Colab notebook with Levels 7-13
- 14 professor questions formulated, top priority: can we use ProteinMPNN public weights?
**Purpose:** Give a new Claude session ALL context needed to continue this work without any prior conversation.

---

## PROJECT IDENTITY

- **Repo:** `C:\Users\I763940\DeepPEF` (local Windows) + GitHub `shaharec/DeepPEF`
- **GPU machine:** `/home/nissimb/workspace/DeepPEF` (Linux, 8GB NVIDIA GPU)
- **User:** M.Sc. thesis student, needs PCC >= 0.70 for ddG prediction
- **Deadline:** Active, time pressure
- **Current best PCC:** 0.5259 (single seed), target 0.55-0.57 ensemble, stretch goal 0.70

---

## CURRENT STATE (June 6, 2026)

### What's on GitHub (pushed):
- Full codebase with all architecture changes
- `run_final_pipeline.sh` — ready to execute 3 experiments (5 seeds each)
- `Megascale-fineTuning/train_subtract_mut.py` — MLP-only subtract-mut (never run)
- `DeepPEF_Training_Final.ipynb` — Colab notebook backup
- Args added to pnas_train.py: `--emb_type`, `--mini_batch_size`, `--emb_projection`

### What's NOT done:
- `run_final_pipeline.sh` has NOT been executed on GPU yet
- GNN subtract-mut (the novel architecture) has NOT been coded yet
- No experiment has been run since v3 failure (PCC 0.4232)
- The user's suggestion "embeddings first / serial fusion" has NOT been implemented

### GPU Machine Status (last known):
- Previous v3 training jobs were killed
- ESM-IF1 features generated for all 368 proteins
- ProtT5 embeddings exist for all proteins
- Repository should be up to date with GitHub (user ran `git pull`)

---

## THE BEST KNOWN CONFIGURATION

```bash
python Megascale-fineTuning/pnas_train.py \
    --model_name baseline_prott5_seed42 \
    --seed 42 \
    --dataset_type pnas \
    --epochs 15 \
    --no_pretrained \
    --loss_type huber_rank \
    --ranking_weight 0.1 \
    --use_knn_gat \
    --one_mut \
    --dg_ml \
    --cosine_lr \
    --lr_min 1e-6 \
    --weight_decay 1e-5 \
    --emb_type prott5 \
    --mini_batch_size 64 \
    --emb_projection none
```
**Result: PCC = 0.5259**

---

## KEY BUGS FIXED (don't re-introduce these)

1. **`.pdb` suffix bug:** `mega_test.csv` has `WT_name` like `3DKM.pdb` but folder names are `3DKM`. Fix: `.str.replace('.pdb', '', regex=False)` in new_dataset.py line ~77.

2. **OOM at 1536-dim:** dual_esmif (1536 features) forces mini_batch_size from 64→8 on 8GB GPU. Fix: use `--emb_projection mlp` to project 1536→16 before GNN.

3. **Stale `__pycache__`:** Background processes (setsid) use cached .pyc files. Fix: always `find . -name '__pycache__' -exec rm -rf {} +` before training.

4. **Duplicate processes:** Check `ps aux | grep pnas_train` before launching. Kill with `pkill -f pnas_train.py`.

5. **SaProt wrong shape:** saprot_emb.pt was [1, 1280] (pooled) instead of [seq_len, 1280]. Never fixed — SaProt abandoned.

---

## WHAT HELPED (proven, keep these):
- k-NN GAT (k=30, cutoff 12A): +0.043 PCC
- Huber loss (delta=1.0): +0.016 PCC
- Cosine LR (1e-4 → 1e-6): +0.015 PCC
- Weight decay 1e-5: +0.008 PCC
- From scratch (no pretrained): +0.005 PCC
- Ranking loss (lambda=0.1): +0.003 PCC

## WHAT FAILED (don't repeat):
- 1536-dim without projection → OOM → batch=8 → PCC 0.42
- ESM-IF1 only (no ProtT5) → PCC 0.28
- WT-only embeddings → PCC 0.25
- Fine-tuning pretrained model → PCC 0.48
- Full-dim DSM → gradient vanishing

---

## NEXT STEPS (agreed plan)

### Immediate (Phase 1): Run existing scripts on GPU
```bash
cd /home/nissimb/workspace/DeepPEF && git pull
setsid bash run_final_pipeline.sh > logs/final_pipeline.log 2>&1 & disown
```
Expected: PCC 0.55-0.57 from 5-seed ensemble.

### Short-term (Phase 2): Implement GNN Subtract-Mut
- Add `fc2_sm = nn.Linear(128, 20)` to `model/hydro_net.py`
- Create `pnas_train_sm.py` (1 forward pass per protein, [L,20] output)
- Create `dataset_sm.py` (WT-only graph + mutation metadata)
- Expected: PCC 0.55-0.65

### Medium-term (Phase 3): User's suggestion — embeddings first
- Put ProtT5 embeddings INTO the GNN (serial fusion)
- Train one-hot together with embeddings
- Expected: +0.05-0.10 PCC on top of Phase 2

---

## USER PREFERENCES (important for working with this user)

1. **Be honest about probabilities.** Never say 90% unless you truly mean it. The user was burned by a false 90% confidence claim (actual was 30-40%).
2. **Verify before promising.** Don't say "it will work" — say "here's the probability and why."
3. **Do exactly what they ask.** Don't add extra features, don't over-engineer, don't create files they didn't ask for.
4. **Be professional and concise.** No filler, no fluff, no unnecessary explanations of things they already know.
5. **Their suggestions matter.** They proposed "embeddings first" and "one-hot with embeddings" multiple times — this should be prioritized.
6. **Evolution approach.** Build on what works. One change at a time. Measure. Keep or revert.
7. **ALWAYS STAY AVAILABLE.** Never launch so many agents that you can't respond. Keep capacity free for user questions at ALL times. **NEVER BLOCK ON AN AGENT. LAUNCH IN BACKGROUND AND RESPOND IMMEDIATELY.**
8. **Don't ask stupid questions to professor.** Only ask what truly blocks progress and can't be resolved independently. PCC targets are OUR problem. Only legal/academic integrity questions are appropriate.

---

## KEY FILE PATHS

| File | Purpose |
|------|---------|
| `model/hydro_net.py` | PEM model (the GNN architecture) |
| `model/model_cfg.py` | Config (emb_projection, proj_dim, etc.) |
| `Megascale-fineTuning/pnas_train.py` | Main training script |
| `Megascale-fineTuning/new_dataset.py` | Dataset loader (6 embedding types) |
| `Megascale-fineTuning/train_subtract_mut.py` | MLP subtract-mut (untested) |
| `train_utils.py` | get_graph(), get_unfolded_graph() |
| `run_final_pipeline.sh` | GPU launch script (5-seed baselines) |
| `data/MsDs/training_data/` | 368 protein folders |
| `data/ThermoMPNN/mega_test.csv` | Test split definition |
| `thesis_docs/architecture_diagram.pptx` | Visual architecture (13 slides) |
| `thesis_docs/analysis_document.md` | Full analysis (experiments, suggestions, plan) |

---

## DATA SITUATION

- **Training:** 340 proteins, ~50K-100K single-point mutations (from MegaScale/PNAS)
- **Testing:** 28 proteins (ThermoMPNN benchmark)
- **Full MegaScale available:** ~776K mutations / 580 proteins (we use ~10%)
- **Discarded:** insertions, deletions, multi-site mutations, outliers
- **ThermoMPNN data advantage:** They likely use more of MegaScale than we do
- **Embeddings available:** ProtT5 [L, 1024] for all mutations; ESM-IF1 [L, 512] for all proteins

---

## ARCHITECTURE SUMMARY (one paragraph)

PEM takes a protein graph (backbone coordinates → pairwise atom distances → Gaussian kernel → per-residue distance features [48-dim] + ProtT5 embeddings [1024-dim] + one-hot AA [20-dim] = 1092-dim per node). It splits into GCN branch (52-dim, sequential edges, 3 layers) and GAT branch (36-dim, k-NN spatial edges, 3 layers with 8-head attention). Outputs are concatenated [88-dim], ProtT5 appended [1112-dim], processed by Light Attention (Conv1d + softmax weighting), then FC(1112→128→1) gives per-residue energy, summed to get total energy E. For each mutation: build folded graph + unfolded graph, compute dG = E_unf - E_fold, then ddG = dG_mut - dG_wt. Cost: 400 forward passes per protein (the main bottleneck).
