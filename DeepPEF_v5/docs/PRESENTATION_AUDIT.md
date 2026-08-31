# Presentation Audit — architecture_diagram.pptx vs REAL code (100% validation)

Validated all 13 slides against the verified source (hydro_net.py, pnas_train.py, train_utils.py,
new_dataset.py). Legend: ✅ correct · ⚠️ misleading/imprecise · ❌ wrong.

## Slide-by-slide

**Slide 1 — Title "PEM Architecture … ddG"**
⚠️ Says the model predicts ddG. TECHNICALLY the model predicts **dG**; ddG is computed only at
validation (pnas_train.py:624). For a thesis talk, clarify: "predicts dG, derives ddG."

**Slide 2 — What the model does**
✅ Graph, k-NN k=30 / 12 Å, node features, E_folded/E_unfolded framing — all correct.
⚠️ Frames output as ddG (same nuance as slide 1).

**Slide 3 — Input data**
✅ coords [L,4,3], mask, one_hot, emb 1024, esmif_enc [L,512], deltaG — matches new_dataset.py.
⚠️ one_hot shown as [n_muts+1, L, 21]; code uses **20** AA (train_utils.get_one_hot → 20 dims;
   pnas_train normalize_batch even trims one_hot to :-1). The "21" is a data-file artifact; the
   model uses 20. Note this.

**Slide 4 — get_graph**
✅ 16 distances (4×4 atoms), Gaussian exp(-0.08·D²), per-residue aggregation. Matches train_utils.

**Slide 5 — folded vs unfolded**
✅ Core thermodynamics correct. dG = E_unfolded − E_folded.
⚠️ Says unfolded keeps "i, i±1, i±2". Code (`zero_except_udiagonal`) keeps the **diagonal and
   immediate ±1 neighbors** (f1, f2, diag) — i.e., i±1, NOT i±2. Minor imprecision.

**Slide 6 — edges**
✅ GCN sequential (i,i+1) [2, L-1]; GAT fully-connected O(L²) vs k-NN. Matches get_edge_index.

**Slide 7 — dual GNN split**
✅ x_gcn = bonded(32)+non_bonded(16)+one_hot(20)=... , x_gat = 16+20=36. Matches forward().
⚠️ Note: real code layout is [dist:16 | bonded:32 | emb | one_hot:20] and GCN uses dist(16)+
   bonded FIRST-16(16)+aa — slide's "52-dim" grouping is consistent enough but the ordering
   wording could confuse. Acceptable.

**Slide 8 — Light Attention + output**
✅ Conv1d kernel=9, feature*softmax(attention). Output FC 1112→128→1, per-residue energy.
❌ **"Energy = SUM"** — this is the KEY point for your thesis: the sum is EXTENSIVE (length-biased).
   The slide states the sum as if final/correct. v5 changes this to per-residue (length_norm).
   → ADD a slide noting the extensive-sum limitation and the v5 fix.

**Slide 9 — end-to-end flow**
✅ Data flow diagram consistent with code (get_graph → [1,L,1092] → split → GNN → concat → LA → FC).

**Slide 10 — training loop**
⚠️ "mini-batch of 64 mutations" — the ARGPARSE DEFAULT is 64, but the PROVEN/working runs use
   **16** (OOM lesson). Slide should say 16 (or note 64 caused OOM). Also "[128, seq_len, 1092]"
   assumes batch 64 → with 16 it's [32, …].
⚠️ Loss shown as "Huber(dG_pred, …)" — correct it also includes ranking (+ energy reg). OK-ish.

**Slide 11 — emb_projection**
✅ Accurately explains none vs mlp and that GNN doesn't see PLM in message-passing by default.
   This is exactly what v5 serial_fusion addresses — GOOD, tie it to v5.

**Slide 12 — "your suggestion: embeddings first"**
✅ This is conceptually **serial_fusion**, which is now implemented and default-on in v5.
   → Update slide: "This idea is now implemented as serial_fusion in v5."

**Slide 13 — summary numbers**
✅ 340 train / 28 test, ~68k mutations, 1092-dim (32+16+1024+20), 1604 with ESM-IF1.
⚠️ Mini-batch 64 → should be 16 (see slide 10).
⚠️ Param counts (~22M / ~700K) not verified against code; treat as approximate.

## VERDICT — is the presentation "perfect"? NO (but close, ~85%).
It is architecturally accurate and well-structured. Fixes needed for 100%:
1. dG vs ddG framing (slides 1, 2): model predicts dG, derives ddG at validation.
2. Extensive SUM limitation (slide 8): flag it + mention v5 length_norm fix.
3. mini_batch 64 → 16 (slides 10, 13): 64 causes OOM; proven runs use 16.
4. Unfolded keeps i±1, not i±2 (slide 5).
5. one_hot uses 20 (not 21) in the model (slide 3).
6. Tie slides 11–12 to v5 serial_fusion (now implemented).

None are catastrophic; all are precision/consistency fixes plus the important extensive-sum point
(which is the scientific core of the v5 improvement).
