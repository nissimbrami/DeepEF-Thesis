# DeepPEF — Session Context (For New Chat Continuation)

**Last updated:** June 10, 2026

**Purpose:** Give a new Claude session ALL context needed to continue this work without any prior conversation.

---

## LATEST DECISIONS (this session):
- **DECISION: GNN-SM is our thesis. We train from scratch. No external pretrained models.**
- Plan: GNN-SM + anti-symmetry + serial fusion + ensemble = target 0.60-0.68
- **CODE COMPLETE: hydro_net.py modified, dataset_sm.py + pnas_train_sm.py + run_evolution_pipeline.sh created**
- **Old files REMOVED: run_final_pipeline.sh, run_experiments.sh, DeepPEF_Training_Final.ipynb (superseded)**
- **All pushed to GitHub (commit 0ef6d3d + a5f8724)**
- **GPU training was STARTED then CANCELLED BY USER on purpose. Do NOT touch GPU or check its status.**
- Verification on GPU PASSED (shapes, gradients, antisymmetry all checked)
- Professor questions: NONE needed. We know what to do.

---

## PROJECT IDENTITY

- **Repo:** `C:\Users\I763940\DeepPEF` (local Windows) + GitHub `nissimbrami/DeepEF-Thesis`
- **CRITICAL: NEVER use `shaharec/DeepPEF` for pushing. That's Shahar's (professor/advisor) repo. OUR repo is `https://github.com/nissimbrami/DeepEF-Thesis`**
- **GPU machine:** `/home/nissimb/workspace/DeepPEF` (Linux, 8GB NVIDIA GPU) — training cancelled by user
- **Colab:** Premium account (A100/V100 available). Notebook: `DeepPEF_Evolution_Training.ipynb`
- **HuggingFace data:** `nissimb/deepef-megascale` (75GB training data being uploaded)
- **User:** M.Sc. thesis student, needs PCC >= 0.70 for ddG prediction
- **Deadline:** Active, time pressure
- **Current best PCC:** 0.5259 (single seed), target 0.55-0.57 ensemble, stretch goal 0.70

---

## CURRENT STATE (June 6, 2026)

### What's on GitHub (latest commits):
- `model/hydro_net.py` — PEM model with BOTH output heads (energy + subtract-mut [L,20])
- `Megascale-fineTuning/pnas_train_sm.py` — GNN-SM training (1 forward pass/protein, verification gate)
- `Megascale-fineTuning/dataset_sm.py` — WT-only dataset with mutation metadata
- `run_evolution_pipeline.sh` — Master GPU script (baseline 5-seed + GNN-SM 5-seed + dual_esmif 5-seed)
- `Megascale-fineTuning/train_subtract_mut.py` — MLP subtract-mut baseline (comparison)
- Args added to pnas_train.py: `--emb_type`, `--mini_batch_size`, `--emb_projection`

### What's DONE (code-complete, verified):
- GNN-SM architecture (fc2_sm in hydro_net.py, subtract_mut forward branch)
- dataset_sm.py (WT-only loading, mutation parsing, 3 embedding types)
- pnas_train_sm.py (training loop, verification gate, damage control)
- run_evolution_pipeline.sh (8-step master script with exit-code gates)
- Verification PASSED on GPU: shapes [1,L,20], gradients non-zero, antisymmetry confirmed

### What's NOT done:
- **NO training results yet** — GPU was cancelled by user on purpose
- The user's suggestion "embeddings first / serial fusion" has NOT been implemented
- nn.Embedding(20,64) to replace one-hot has NOT been implemented
- No 5-seed ensemble has been run

### GPU Machine Status:
- **ACCESS METHOD: WSL via `wsl -e bash -c "cd /home/nissimb/workspace/DeepPEF && ..."`**
- **Also accessible at: `//wsl.localhost/Ubuntu/home/nissimb/workspace/DeepPEF/`**
- **USER CANCELLED TRAINING ON PURPOSE — DO NOT TOUCH OR CHECK GPU**
- **CRLF fix applied to all scripts (Windows git adds \r, must sed -i 's/\r$//' on Linux side)**
- Code is at commit 0ef6d3d (GNN-SM + evolution pipeline)
- ESM-IF1 features generated for all 368 proteins
- ProtT5 embeddings exist for all proteins
- All verification checks PASSED (shapes, gradients, antisymmetry)

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

6. **CRLF line endings:** Windows git adds `\r` to .sh/.py files. On Linux: `sed -i 's/\r$//' *.sh Megascale-fineTuning/*.py` before running anything.

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

### Immediate: Run evolution pipeline on GPU
```bash
cd /home/nissimb/workspace/DeepPEF
sed -i 's/\r$//' run_evolution_pipeline.sh Megascale-fineTuning/*.py
chmod +x run_evolution_pipeline.sh
tmux new-session -d -s deepef 'bash run_evolution_pipeline.sh 2>&1 | tee logs/evolution_pipeline.log'
```
Expected: Level 7 baseline PCC ~0.52/seed, ensemble 0.55-0.57. GNN-SM PCC 0.45-0.65.

### Short-term: User's suggestions (NOT YET IMPLEMENTED)
1. **Serial fusion** — Put ProtT5 embeddings INTO the GNN as input (not post-concatenated)
2. **nn.Embedding(20,64)** — Replace static one-hot with learnable embeddings + projected PLM
3. Expected: +0.05-0.10 PCC on top of GNN-SM

### Medium-term: Optimize based on results
- Anti-symmetric data augmentation (free, doubles effective data)
- Richer edge features (RBF distances, orientations)
- 5-10 seed ensemble of best config

---

## USER PREFERENCES (CRITICAL — follow these exactly)

1. **Be honest about probabilities.** Never say 90% unless you truly mean it. User was burned by false confidence claims.
2. **Verify before promising.** Don't say "it will work" — say "here's the probability and why."
3. **Do exactly what they ask.** Don't add extra features, don't over-engineer, don't create files they didn't ask for.
4. **Be professional and concise.** No filler, no fluff, no unnecessary explanations.
5. **Their suggestions matter.** "Embeddings first" and "one-hot with embeddings" should be prioritized.
6. **Evolution approach.** Build on what works. One change at a time. Measure. Keep or revert.
7. **ALWAYS STAY AVAILABLE.** Never launch so many agents that you can't respond. **NEVER BLOCK ON AN AGENT. LAUNCH IN BACKGROUND AND RESPOND IMMEDIATELY.**
8. **Don't ask stupid questions to professor.** PCC targets are OUR problem. Only legal/academic integrity questions are appropriate.
9. **Update docs after every significant action.** New chat needs to understand everything.
10. **GPU was cancelled on purpose.** Don't touch it, don't check it, don't ask about it unless user brings it up.
11. **NEVER DELETE FILES WITHOUT EXPLICIT USER APPROVAL.** Always verify first, ask before deleting anything. No exceptions.
12. **NEVER UPLOAD/WRITE/PUSH DATA WITHOUT EXPLICIT USER APPROVAL.** Don't upload to HuggingFace, Drive, or anywhere without being told to. Read context before every response.
13. **ONLY USE USER'S GITHUB: `https://github.com/nissimbrami/DeepEF-Thesis`**. NEVER push to `shaharec/DeepPEF`. That is Shahar's (professor/advisor) repo. We only READ from it if needed. All pushes go to nissimbrami/DeepEF-Thesis.
14. **Record all instructions in this context file.** When user gives a rule, write it here immediately so future sessions know.

---

## KEY FILE PATHS

| File | Purpose |
|------|---------|
| `model/hydro_net.py` | PEM model — has fc2_sm + forward(f_type='subtract_mut') for GNN-SM |
| `model/model_cfg.py` | Config (emb_projection, proj_dim, etc.) |
| `Megascale-fineTuning/pnas_train.py` | Original training script (energy-difference approach) |
| `Megascale-fineTuning/pnas_train_sm.py` | GNN-SM training (subtract-mut, 1 pass/protein) |
| `Megascale-fineTuning/dataset_sm.py` | WT-only dataset with mutation metadata |
| `Megascale-fineTuning/new_dataset.py` | Original dataset loader (6 embedding types) |
| `Megascale-fineTuning/train_subtract_mut.py` | MLP subtract-mut baseline (comparison) |
| `train_utils.py` | get_graph(), get_unfolded_graph() |
| `run_evolution_pipeline.sh` | Master GPU script with verification gates |
| `data/MsDs/training_data/` | 368 protein folders |
| `data/ThermoMPNN/mega_test.csv` | Test split definition |
| `thesis_docs/architecture_diagram.pptx` | Visual architecture (13 slides) |
| `thesis_docs/analysis_document.md` | Full analysis (experiments, suggestions, plan) |
| `thesis_docs/levels_progression.md` | All experiments as levels with learnings |
| `DeepPEF_Evolution_Training.ipynb` | Colab notebook — full pipeline (Level 7/9/10) |
| `validate_all.py` | 9-test validation suite (shapes, gradients, antisymmetry, compat) |
| `thesis_docs/session_context.md` | This file — bootstrap for new chats |

---

## DATA SITUATION

- **Training:** 340 proteins, ~50K-100K single-point mutations (from MegaScale/PNAS)
- **Testing:** 28 proteins (ThermoMPNN benchmark)
- **Full MegaScale available:** ~776K mutations / 580 proteins (we use ~10%)
- **Discarded:** insertions, deletions, multi-site mutations, outliers
- **ThermoMPNN data advantage:** They likely use more of MegaScale than we do
- **Embeddings available:** ProtT5 [L, 1024] for all mutations; ESM-IF1 [L, 512] for all proteins

---

## ARCHITECTURE SUMMARY

### Original (energy-difference, pnas_train.py):
PEM takes a protein graph (backbone coords → pairwise atom distances → Gaussian kernel → per-residue distance features [48-dim] + ProtT5 [1024-dim] + one-hot [20-dim] = 1092-dim per node). Splits into GCN branch (52-dim, sequential edges, 3 layers) and GAT branch (36-dim, k-NN spatial edges, 3 layers with 8-head attention). Outputs concatenated [88-dim], ProtT5 appended [1112-dim], Light Attention, FC(1112→128→1) per-residue energy, summed to total E. For each mutation: folded + unfolded graph, dG = E_unf - E_fold, ddG = dG_mut - dG_wt. Cost: ~400 forward passes per protein.

### NEW (subtract-mut, pnas_train_sm.py):
Same GNN backbone but output FC(128→20) gives [L,20] amino acid scores. ddG = score[pos, mut_aa] - score[pos, wt_aa]. Only 1 forward pass per protein (ALL mutations scored instantly). Anti-symmetry is automatic. This is the novel thesis contribution.
