# THE CHECKLIST — 10 tasks, each with its own 3-stage checklist

**Created 2026-09-10.** Derived from the planner's work-order (Part E) and executed under
`PROTOCOL.md`. **One task active at a time. No parallelism except background GPU jobs.**

Each task carries the same inner checklist:
`[ ] PLAN (question / hypothesis / falsifier / metric / config / "if X then Y")`
`[ ] EXECUTE (end to end, no deviation)`
`[ ] VERIFY (criterion met? did it train? did the mechanism occur?)`
`[ ] DONE (mean AND median + training-health line reported)`

Status: `[ ]` open · `[x]` done · `[~]` blocked (reason recorded) · `[>]` in progress

---

## T1 — Training curves for the two Flory arms and the descriptor arm
- [x] PLAN — Q: did these arms train? Falsifier registered in advance: *a flat or diverging
      curve is an optimisation failure and the rejection is void.* Metric: val RMSE + train loss
      per epoch. If flat -> rejection void; if monotone -> rejection stands on merit.
- [x] EXECUTE — read 4 training logs
- [x] VERIFY — Flory 1.450->1.086 and 2.313->1.628, loss falling = HEALTHY.
      Descriptors RMSE 2.541/2.529 x15, loss constant to 5 decimals = NEVER TRAINED.
- [x] DONE -> `W6_ROOT_CAUSE.md` §Task 1. **Cost: 10 min. Outcome: the wave produced 2 findings, not 3.**

## T2 — Descriptor block: folded vs unfolded, `torch.equal`
- [x] PLAN — H: the block is state-independent and cancels in dG. Falsifier: if `torch.equal`
      is False the hypothesis dies. If True -> root cause, and no descriptor arm can ever work as wired.
- [x] EXECUTE — direct tensor comparison
- [x] VERIFY — W6 `torch.equal`=True, max|d|=0.000e+00; W12 False, max|d|=1.501e+00.
      Corroborated by `train_utils.py:734` comment and the 5-decimal-constant loss.
- [x] DONE -> `W6_ROOT_CAUSE.md` + `scripts/gate_state_dependence.py` (W5/W12/W15 PASS, both W6 modes FAIL)

## T3 — Verify the distance kernel is monotone in the W7 runs
- [x] PLAN — Falsifier: if k(2A)>k(8A)>k(15A) fails, the model is distance-blind and both W7
      nulls are void. If monotone -> the nulls stand on merit.
- [x] EXECUTE — kernel read from train_utils.py:390/:652, evaluated at 2-20 A
- [x] VERIFY — **falsifier did NOT fire**: strictly decreasing (0.726 -> 1.5e-08). But it
      SATURATES: past 12 A it is at float32 eps. Also CORRECTED the review: span and bidir are
      separate flags (hydro_net.py:720-722 + branch), so span4 is NOT confounded, and the
      requested bidirectional span-1 control already exists and ran (u10bidir, also null).
- [x] DONE -> `T3_KERNEL_AND_SPAN.md`. **New actionable lever: `gaussian_coef`, not `gcn_span`.**

## T4 — BSA vs `b_p` and `a_p` on the 27 test proteins
- [x] PLAN - Falsifier: if no interface feature is significant vs b_p, the direction is closed.
- [x] EXECUTE - every structural feature vs b_p/a_p/r, 27 proteins, 58 healthy runs, 201 tests
- [x] VERIFY - falsifier FIRED: interface/SASA null for b_p (best n_hbond -0.446 p=0.020, vs
      Bonferroni 0.00025). NEW: packing_frac vs a_p -0.662 (p=0.0002) and void_vol_per_res
      +0.652 both SURVIVE Bonferroni. Tightly packed proteins compress hardest.
- [x] DONE -> T4_INTERFACE_NULL.md. Predicts W12 helps most on tightly packed proteins (T8).

## T5 — `pub_w5_ddg_s42`: separate the objective from the block
- [~] BLOCKED — arm is training (GPU, background). Not an active task.

## T6 — `slope 0.7` at 5 seeds (the only positive candidate the wave produced)
- [ ] PLAN
- [ ] EXECUTE
- [ ] VERIFY
- [ ] DONE

## T7 — W12 pooled score, 2 seeds, canonical basis
- [ ] PLAN
- [ ] EXECUTE
- [ ] VERIFY
- [ ] DONE

## T8 — W12 mechanism: slope gap buried vs exposed
- [ ] PLAN
- [ ] EXECUTE
- [ ] VERIFY
- [ ] DONE

## T9 — W15 real side-chain reconstruction
- [~] BLOCKED — `pub_w15b_s42` resuming on GPU.

## T10 — Distogram head at weight 0.1
- [~] BLOCKED — `disto0.01/0.1/1.0` running/queued on GPU.

---

**Rule: nothing below the active task is touched until it is DONE.**
