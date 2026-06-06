# Question for the Professor

**One question that blocks our architecture decision:**

### Is using publicly available pretrained ProteinMPNN weights (MIT license, github.com/dauparas/ProteinMPNN) as a frozen feature encoder acceptable for the thesis?

**Context:** ThermoMPNN (Li et al., 2023) achieves PCC=0.754 on MegaScale by adding a small MLP on top of frozen ProteinMPNN encoder features. The weights are public, MIT-licensed, ~7MB.

**If YES:** We can use these as input features (like we already use ESM-IF1 encoder features). Our GNN then processes these richer features → likely PCC 0.70+.

**If NO:** We continue with our from-scratch GNN-SM approach targeting PCC 0.60-0.65, which is still novel (first GNN + subtract-mut architecture).

**Why this matters:** This is a legal/academic integrity question. Using someone else's pretrained model vs training everything ourselves — the professor should decide what's appropriate for the thesis scope.
