# Questions for the Professor

## Question 1: Pretraining with CASP12 data

Shahar's HuggingFace dataset (`shaharec/deepef-data`) has CASP12 structures (~110GB) with ProtT5 embeddings, designed for native vs decoy pretraining. Our current approach trains from scratch (no pretraining) because an earlier test showed only +0.005 PCC benefit from pretraining.

**Question:** Should we re-investigate pretraining our GNN-SM model on CASP12 native-vs-decoy discrimination BEFORE fine-tuning on MegaScale mutations? The earlier test was with the old energy-difference architecture — GNN-SM's [L,20] output head is different and might benefit more from structural pretraining.

**Context:**
- Data is already available on HuggingFace (Shahar uploaded it)
- Original DeepEF paper used this pretraining step
- Our current best PCC is 0.5259 without pretraining
- Risk: Pretraining adds complexity but may not transfer to subtract-mut output head

---

## Previously decided: No other questions needed
- GNN-SM is our novel architecture (nobody did GNN + subtract-mut before)
- We train from scratch — no external pretrained models (unless professor says otherwise on Q1)
- If we beat ThermoMPNN, great. If not, the novelty is the contribution.
