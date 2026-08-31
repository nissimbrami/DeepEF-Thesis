"""Generate rendered diagrams for the DeepPEF_v5 teaching presentation.

Produces PNG figures (matplotlib, no network / no data needed) that visualize:
  fig1_dg_concept       — folded vs unfolded energy difference (the core idea)
  fig2_node_layout      — the node-feature vector layout + where each lever inserts
  fig3_architecture     — the two-tower GCN+GATv2 pipeline, input -> energy
  fig4_levers_grid      — a 2x3 grid, one panel per lever A-F (before/after sketch)
  fig5_flow_compose     — the dimension-contract flow + composability matrix

Run:  python DeepPEF_v5/docs/presentation/make_figures.py
Output: DeepPEF_v5/docs/presentation/figures/*.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# palette
C_GEOM = "#4C72B0"    # distance/geometry - blue
C_BOND = "#55A868"    # bonded - green
C_BURIAL = "#C44E52"  # burial - red (lever C)
C_EMB = "#8172B2"     # embedding - purple
C_OH = "#CCB974"      # one-hot - sand
C_ACC = "#DD8452"     # accent / lever - orange
C_GCN = "#55A868"
C_GAT = "#4C72B0"
C_BG = "#F5F5F5"


def _box(ax, x, y, w, h, text, fc, ec="#333333", fs=10, tc="white", lw=1.4, round=0.02):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={round}",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, wrap=True)


def _arrow(ax, x1, y1, x2, y2, color="#333333", lw=2, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                        color=color, lw=lw, zorder=1)
    ax.add_patch(a)


# ---------------------------------------------------------------- FIG 1: dG concept
def fig1():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis("off")
    ax.set_title("The core idea: folding free energy as an energy difference",
                 fontsize=15, weight="bold")

    # folded blob (compact contacts)
    np.random.seed(3)
    cx, cy = 2.2, 3.0
    pts = np.random.randn(9, 2) * 0.5 + [cx, cy]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) < 1.0:
                ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                        color=C_GAT, lw=1, alpha=0.5, zorder=1)
    ax.scatter(pts[:, 0], pts[:, 1], s=180, c=C_GEOM, edgecolors="white", zorder=3)
    ax.text(cx, cy - 1.4, "FOLDED\n(real 3D contacts)", ha="center", fontsize=11, weight="bold")
    ax.text(cx, cy + 1.5, r"$E_{folded}$", ha="center", fontsize=14, color=C_GAT)

    # unfolded chain (linear)
    ux = np.linspace(6.2, 9.0, 9); uy = 3.0 + 0.12 * np.sin(np.arange(9))
    ax.plot(ux, uy, color="#999999", lw=1.5, zorder=1)
    ax.scatter(ux, uy, s=180, c=C_BOND, edgecolors="white", zorder=3)
    ax.text(7.6, cy - 1.4, "UNFOLDED\n(random coil / chain-local)", ha="center", fontsize=11, weight="bold")
    ax.text(7.6, cy + 1.5, r"$E_{unfolded}$", ha="center", fontsize=14, color=C_BOND)

    _arrow(ax, 3.6, 3.0, 6.0, 3.0, color=C_ACC, lw=2.5)
    ax.text(4.8, 3.35, "same sequence,\ntwo states", ha="center", fontsize=10, color=C_ACC)

    _box(ax, 3.0, 0.3, 5.0, 0.8,
         r"$\Delta G = E_{unfolded} - E_{folded}$      $\Delta\Delta G = \Delta G_{mut} - \Delta G_{wt}$",
         fc="#2F2F2F", fs=13, round=0.05)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_dg_concept.png"), dpi=150, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- FIG 2: node layout
def fig2():
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.set_title("Node-feature layout & where each lever changes it",
                 fontsize=15, weight="bold")

    # baseline row
    y0 = 3.6
    segs = [("D\n16\n(distance)", 2.4, C_GEOM),
            ("Fb\n32\n(bonded)", 2.4, C_BOND),
            ("emb\nE=1024\n(ProtT5)", 3.2, C_EMB),
            ("one_hot\n20\n(identity)", 2.2, C_OH)]
    x = 0.6
    ax.text(0.6, y0 + 1.15, "BASELINE", fontsize=11, weight="bold")
    for txt, w, c in segs:
        _box(ax, x, y0, w, 0.9, txt, fc=c, fs=9)
        x += w + 0.05

    # lever-C row (burial inserted before emb)
    y1 = 1.5
    ax.text(0.6, y1 + 1.15, "+ Lever C (burial)", fontsize=11, weight="bold", color=C_BURIAL)
    segsC = [("D 16", 2.0, C_GEOM), ("Fb 32", 2.0, C_BOND),
             ("burial\nb=1", 1.1, C_BURIAL), ("emb E", 3.0, C_EMB), ("one_hot 20", 2.1, C_OH)]
    x = 0.6
    for txt, w, c in segsC:
        _box(ax, x, y1, w, 0.9, txt, fc=c, fs=9)
        x += w + 0.05
    # highlight arrow to burial insertion
    _arrow(ax, 4.9, y0, 4.9, y1 + 0.95, color=C_BURIAL, lw=2, style="-|>")

    ax.text(6.0, 0.55,
            "one_hot always stays LAST (model slices it from the right).\n"
            "Burial is inserted BEFORE emb; every width derives from CFG so nothing drifts.",
            ha="center", fontsize=9.5, style="italic", color="#333333")

    # tags for other levers (which part they touch)
    ax.text(1.8, y0 - 0.15, "B: kernel of D\nF: also builds edges", ha="center", fontsize=8, color=C_ACC)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_node_layout.png"), dpi=150, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- FIG 3: architecture
def fig3():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.set_title("Two-tower architecture: from graph to per-residue energy",
                 fontsize=15, weight="bold")

    _box(ax, 0.3, 2.7, 1.9, 1.0, "coords\n[L,4,3]\n+ ProtT5\n+ one-hot", fc="#2F2F2F", fs=9)
    _box(ax, 2.6, 2.7, 1.9, 1.0, "build graph\n(train_utils)\nnodes+edges", fc=C_GEOM, fs=9)

    # two towers
    _box(ax, 5.1, 4.0, 2.1, 1.0, "GCN tower\n(sequential\ni..i+S edges)", fc=C_GCN, fs=9)
    _box(ax, 5.1, 1.4, 2.1, 1.0, "GATv2 tower\n(spatial k-NN\n+ edge feats)", fc=C_GAT, fs=9)

    _box(ax, 7.7, 2.7, 1.7, 1.0, "concat +\nlight attn\n+ fc1(128)", fc=C_EMB, fs=9)
    _box(ax, 9.7, 2.7, 2.0, 1.0, "fc2 -> K terms\nsum -> energy", fc=C_ACC, fs=9)

    _arrow(ax, 2.2, 3.2, 2.6, 3.2)
    _arrow(ax, 4.5, 3.2, 5.1, 4.5)
    _arrow(ax, 4.5, 3.2, 5.1, 1.9)
    _arrow(ax, 7.2, 4.5, 7.7, 3.4)
    _arrow(ax, 7.2, 1.9, 7.7, 3.0)
    _arrow(ax, 9.4, 3.2, 9.7, 3.2)

    # denoise head branch (lever E)
    _box(ax, 7.7, 0.3, 1.7, 0.75, "denoise head\n(Lever E, aux)", fc=C_BURIAL, fs=8.5)
    _arrow(ax, 8.55, 2.7, 8.55, 1.05, color=C_BURIAL, style="-|>")

    # annotations for levers
    ax.text(6.15, 5.15, "F: span S", ha="center", fontsize=8, color=C_ACC)
    ax.text(6.15, 1.25, "F: edge_attr 41-d", ha="center", fontsize=8, color=C_ACC)
    ax.text(10.7, 3.85, "A: K terms", ha="center", fontsize=8, color=C_ACC)
    ax.text(3.55, 2.55, "B kernel, C burial,\nD flory (unfolded)", ha="center", fontsize=8, color=C_ACC)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_architecture.png"), dpi=150, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- FIG 4: levers grid
def _mini_axes(ax, title, color):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(2)
    ax.set_title(title, fontsize=11, weight="bold", color=color, pad=8)
    ax.set_facecolor(C_BG)


def fig4():
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    fig.suptitle("The six levers (default = OFF = baseline)", fontsize=16, weight="bold")

    # A: energy decomposition — draw everything in axis-fraction coords for a clean layout.
    a = axs[0, 0]; _mini_axes(a, "A  Energy decomposition", C_ACC)
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.text(0.5, 0.92, "fc2: 128 -> K per-residue terms", ha="center", fontsize=9,
           transform=a.transAxes)
    for i, (lbl, w) in enumerate([("contact", 0.55), ("solvation", 0.45), ("backbone", 0.35)]):
        y = 0.62 - i * 0.16
        a.add_patch(Rectangle((0.32, y - 0.05), w, 0.1, transform=a.transAxes,
                              fc=C_GEOM, ec="white"))
        a.text(0.30, y, lbl, ha="right", va="center", fontsize=8, transform=a.transAxes)
    a.text(0.5, 0.06, "K=1 == today's single scalar", ha="center", fontsize=8,
           style="italic", transform=a.transAxes)

    # B: RBF bank
    b = axs[0, 1]; _mini_axes(b, "B  RBF distance bank", C_ACC)
    xx = np.linspace(0, 20, 200)
    b.plot(xx, np.exp(-0.08 * xx ** 2), color="#999", lw=2, label="single Gaussian")
    for c in np.linspace(2, 16, 6):
        b.plot(xx, np.exp(-0.08 * (xx - c) ** 2), color=C_GEOM, lw=1, alpha=0.7)
    b.set_xlim(0, 20); b.set_ylim(0, 1.05); b.legend(fontsize=7, loc="upper right")
    b.text(0.5, -0.12, "M RBFs summed -> width 16 (dim-preserving)", ha="center",
           fontsize=8, style="italic", transform=b.transAxes)

    # C: burial
    c = axs[0, 2]; _mini_axes(c, "C  Burial / solvation", C_BURIAL)
    th = np.linspace(0, 2 * np.pi, 60)
    c.plot(1.15 * np.cos(th), 1.15 * np.sin(th), color="#bbb")
    np.random.seed(1)
    p = np.random.randn(14, 2) * 0.5
    dens = np.array([np.sum(np.linalg.norm(p - q, axis=1) < 0.55) for q in p])
    c.scatter(p[:, 0], p[:, 1], c=dens, cmap="Reds", s=90, edgecolors="k", lw=0.4)
    c.set_xlim(-1.4, 1.4); c.set_ylim(-1.4, 1.4); c.set_aspect("equal")
    c.text(0.5, -0.1, "buried (dark) vs surface (light)", ha="center", fontsize=8,
           style="italic", transform=c.transAxes)

    # D: flory
    d = axs[1, 0]; _mini_axes(d, "D  Flory unfolded reference", C_ACC)
    sep = np.arange(1, 15)
    d.plot(sep, sep ** 0.5, color=C_BOND, lw=2, label=r"coil $|i-j|^{0.5}$")
    d.plot(sep, np.where(sep <= 1, 1.0, 0.0), "o", color="#999", label="tridiagonal")
    d.set_xlim(0, 15); d.set_ylim(-0.3, 4.2); d.set_xlabel("|i - j|", fontsize=8)
    d.legend(fontsize=7, loc="upper left")
    d.text(0.5, -0.2, "random-coil distances (value-only)", ha="center", fontsize=8,
           style="italic", transform=d.transAxes)

    # E: denoise
    e = axs[1, 1]; _mini_axes(e, "E  Decoys + denoising head", C_BURIAL)
    np.random.seed(2)
    base = np.random.randn(8, 2) * 0.35 + [0.5, 0.55]
    noisy = base + np.random.randn(8, 2) * 0.1
    e.scatter(base[:, 0], base[:, 1], s=70, c=C_GAT, label="native", zorder=3)
    e.scatter(noisy[:, 0], noisy[:, 1], s=70, c=C_ACC, marker="x", label="+noise", zorder=3)
    for i in range(len(base)):
        e.annotate("", xy=base[i], xytext=noisy[i],
                   arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))
    e.set_xlim(-0.2, 1.4); e.set_ylim(-0.2, 1.5); e.legend(fontsize=7, loc="upper right")
    e.text(0.5, -0.12, "head predicts injected noise (train only)", ha="center",
           fontsize=8, style="italic", transform=e.transAxes)

    # F: edges
    f = axs[1, 2]; _mini_axes(f, "F  Edge features + connectivity", C_ACC)
    xs = np.arange(6)
    f.scatter(xs, np.zeros(6), s=160, c=C_GEOM, zorder=3)
    for i in range(5):
        f.annotate("", xy=(xs[i + 1], 0), xytext=(xs[i], 0),
                   arrowprops=dict(arrowstyle="-", color=C_BOND, lw=2))
    for i in range(4):  # span=2 extra edges
        f.annotate("", xy=(xs[i + 2], 0.3), xytext=(xs[i], 0.3),
                   arrowprops=dict(arrowstyle="-", color=C_ACC, lw=1.5, linestyle="--"))
    f.set_xlim(-0.5, 5.5); f.set_ylim(-0.6, 0.8)
    f.text(0.5, -0.18, "span S both dirs; edge_attr=[oh|oh|dist]=41", ha="center",
           fontsize=8, style="italic", transform=f.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0)
    fig.savefig(os.path.join(FIG, "fig4_levers_grid.png"), dpi=150, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- FIG 5: flow + compose
def fig5():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

    # flow: single CFG feeds builder + model
    ax1.set_xlim(0, 6); ax1.set_ylim(0, 5.4); ax1.axis("off")
    ax1.set_title("The dimension contract", fontsize=14, weight="bold")
    _box(ax1, 2.0, 4.2, 2.0, 0.8, "model_cfg_v5.CFG\n(single source)", fc="#2F2F2F", fs=9)
    _box(ax1, 0.4, 2.2, 2.2, 0.9, "graph builder\n(train_utils)", fc=C_GEOM, fs=9)
    _box(ax1, 3.4, 2.2, 2.2, 0.9, "model\n(hydro_net_v5)", fc=C_GAT, fs=9)
    _arrow(ax1, 2.6, 4.2, 1.6, 3.1, color=C_ACC)
    _arrow(ax1, 3.4, 4.2, 4.4, 3.1, color=C_ACC)
    _box(ax1, 1.4, 0.6, 3.2, 0.9, "SAME widths ->\ntensors & math agree", fc=C_BOND, fs=9)
    _arrow(ax1, 1.5, 2.2, 2.6, 1.5, color="#333")
    _arrow(ax1, 4.5, 2.2, 3.4, 1.5, color="#333")

    # composability matrix
    ax2.set_title("Composability (why levers stack cleanly)", fontsize=14, weight="bold")
    levers = ["A", "B", "C", "D", "E", "F"]
    kind = {"A": "head", "B": "dim-preserve", "C": "node width",
            "D": "value-only", "E": "aux head", "F": "edges"}
    colors = {"head": C_ACC, "dim-preserve": C_GEOM, "value-only": C_BOND,
              "node width": C_BURIAL, "aux head": C_EMB, "edges": C_GAT}
    for i, lv in enumerate(levers):
        k = kind[lv]
        ax2.add_patch(Rectangle((0, 5 - i), 1.2, 0.8, fc=colors[k], ec="k"))
        ax2.text(0.6, 5.4 - i, lv, ha="center", va="center", color="white",
                 fontsize=12, weight="bold")
        ax2.text(1.4, 5.4 - i, f"{k}", va="center", fontsize=10)
    ax2.text(0.0, -0.4,
             "A/E orthogonal to widths | B preserves width | D value-only\n"
             "C changes node width, F changes edges -> compose via CFG",
             fontsize=9, style="italic")
    ax2.set_xlim(0, 5); ax2.set_ylim(-0.8, 6.2); ax2.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_flow_compose.png"), dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("Wrote figures to", FIG)
    for f in sorted(os.listdir(FIG)):
        print("  ", f)
