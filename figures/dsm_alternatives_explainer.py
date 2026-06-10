"""
Explainer plots for three DSM alternatives:
1. Direct Score Prediction Head
2. Fisher Divergence with Hutchinson Trace
3. Noise Conditional Score Network (NCSN)

Each panel shows architecture, math, and why it helps.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig = plt.figure(figsize=(22, 28))

# ========================================================================
# Helper: draw a rounded box with text
# ========================================================================
def draw_box(ax, xy, w, h, text, color='#E8F4FD', ec='#2196F3', fontsize=10,
             text_color='black', lw=1.5, alpha=1.0, bold=False):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=ec, linewidth=lw, alpha=alpha)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, weight=weight)

def draw_arrow(ax, start, end, color='#333', lw=1.5, style='->', connectionstyle='arc3,rad=0'):
    arrow = FancyArrowPatch(start, end, arrowstyle=style, color=color,
                            lw=lw, connectionstyle=connectionstyle,
                            mutation_scale=15)
    ax.add_patch(arrow)

# Colors
C_BLUE = '#E3F2FD'
C_GREEN = '#E8F5E9'
C_ORANGE = '#FFF3E0'
C_RED = '#FFEBEE'
C_PURPLE = '#F3E5F5'
C_GRAY = '#F5F5F5'
EC_BLUE = '#1976D2'
EC_GREEN = '#388E3C'
EC_ORANGE = '#F57C00'
EC_RED = '#D32F2F'
EC_PURPLE = '#7B1FA2'


# ========================================================================
# PANEL 1: Direct Score Prediction Head
# ========================================================================
ax1 = fig.add_axes([0.03, 0.68, 0.94, 0.30])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
ax1.axis('off')
ax1.set_title('1. Direct Score Prediction Head', fontsize=16, fontweight='bold',
              pad=15, loc='left', color=EC_BLUE)

# --- Current (broken) approach on the left ---
ax1.text(2.5, 5.7, 'CURRENT (broken)', ha='center', fontsize=12, fontweight='bold', color=EC_RED)

# Input
draw_box(ax1, (1.5, 4.8), 2, 0.5, 'X̃  (noisy input)', C_GRAY, '#666')
# Model
draw_box(ax1, (1.5, 3.8), 2, 0.7, 'GNN Backbone\n+ Energy Head', C_BLUE, EC_BLUE, bold=True)
# E output
draw_box(ax1, (1.7, 2.9), 1.6, 0.5, 'E(X̃)  scalar', C_ORANGE, EC_ORANGE)
# Autograd
draw_box(ax1, (1.3, 1.9), 2.4, 0.6, '∇ₓE  via autograd\n(1st order)', C_GREEN, EC_GREEN, fontsize=9)
# DSM loss
draw_box(ax1, (1.3, 0.8), 2.4, 0.7, 'DSM Loss:\n||∇ₓE + noise/σ²||²', C_RED, EC_RED, fontsize=9)
# Backprop label
draw_box(ax1, (1.1, 0.05), 2.8, 0.5, '∂L/∂θ needs ∂²E/∂x∂θ\nHESSIAN → vanishes!', C_RED, EC_RED,
         fontsize=9, text_color=EC_RED, bold=True)

# Arrows
draw_arrow(ax1, (2.5, 4.8), (2.5, 4.55))
draw_arrow(ax1, (2.5, 3.8), (2.5, 3.45))
draw_arrow(ax1, (2.5, 2.9), (2.5, 2.55))
draw_arrow(ax1, (2.5, 1.9), (2.5, 1.55))
draw_arrow(ax1, (2.5, 0.8), (2.5, 0.6), color=EC_RED)

# --- Proposed (works) approach on the right ---
ax1.text(7.5, 5.7, 'PROPOSED (works)', ha='center', fontsize=12, fontweight='bold', color=EC_GREEN)

# Input
draw_box(ax1, (6.5, 4.8), 2, 0.5, 'X̃  (noisy input)', C_GRAY, '#666')
# Shared backbone
draw_box(ax1, (6.5, 3.8), 2, 0.7, 'GNN Backbone\n(shared weights)', C_BLUE, EC_BLUE, bold=True)

# Two heads branching
# Energy head (left branch)
draw_box(ax1, (5.2, 2.7), 1.8, 0.7, 'Energy Head\nE(X̃) → scalar', C_ORANGE, EC_ORANGE, fontsize=9)
# Score head (right branch)
draw_box(ax1, (8.0, 2.7), 1.8, 0.7, 'Score Head\ns_θ(X̃) → R^D', C_GREEN, EC_GREEN, fontsize=9)

# InfoNCE loss (left)
draw_box(ax1, (5.2, 1.5), 1.8, 0.7, 'InfoNCE (lossd)\nfirst-order ✓', C_BLUE, EC_BLUE, fontsize=9)
# DSM loss (right)
draw_box(ax1, (8.0, 1.5), 1.8, 0.7, 'DSM on s_θ\n||s_θ + noise/σ²||²', C_GREEN, EC_GREEN, fontsize=9)

# Backprop label
draw_box(ax1, (6.2, 0.3), 2.6, 0.8, '∂L/∂θ = ∂s_θ/∂θ\nFIRST-ORDER only!\nNo Hessian needed ✓', C_GREEN, EC_GREEN,
         fontsize=9, text_color=EC_GREEN, bold=True)

# Consistency loss
draw_box(ax1, (5.5, 0.3), 0.5, 0.3, '≈', '#FFF', '#999', fontsize=14)

# Arrows
draw_arrow(ax1, (7.5, 4.8), (7.5, 4.55))
draw_arrow(ax1, (7.5, 3.8), (7.5, 3.55))
# Branch to energy head
draw_arrow(ax1, (6.8, 3.8), (6.1, 3.45), color=EC_ORANGE)
# Branch to score head
draw_arrow(ax1, (8.2, 3.8), (8.9, 3.45), color=EC_GREEN)
# Down to losses
draw_arrow(ax1, (6.1, 2.7), (6.1, 2.25))
draw_arrow(ax1, (8.9, 2.7), (8.9, 2.25))
# Down to explanation
draw_arrow(ax1, (8.0, 1.5), (7.8, 1.15), color=EC_GREEN)

# Divider
ax1.axvline(x=5.0, ymin=0.05, ymax=0.95, color='#CCC', linestyle='--', lw=1)

# Key insight text
ax1.text(5.0, 5.7, '→', ha='center', fontsize=20, color='#666')


# ========================================================================
# PANEL 2: Fisher Divergence with Hutchinson Trace
# ========================================================================
ax2 = fig.add_axes([0.03, 0.36, 0.94, 0.28])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.axis('off')
ax2.set_title('2. Fisher Divergence with Hutchinson Trace Estimator', fontsize=16,
              fontweight='bold', pad=15, loc='left', color=EC_PURPLE)

# Math explanation on top
ax2.text(5.0, 5.5,
    'Implicit Score Matching (no noise needed!):   $L_{ISM} = \\frac{1}{2}||\\nabla_x E||^2 + \\mathrm{tr}(\\nabla^2_x E)$',
    ha='center', fontsize=13, style='italic',
    bbox=dict(boxstyle='round', facecolor=C_PURPLE, edgecolor=EC_PURPLE, alpha=0.3))

# Left: Full Hessian (expensive)
ax2.text(2.5, 4.7, 'Full Hessian trace (impossible)', ha='center', fontsize=11,
         fontweight='bold', color=EC_RED)

# Draw Hessian matrix
hess_x, hess_y = 1.2, 2.5
hess_w, hess_h = 2.5, 2.0
rect = plt.Rectangle((hess_x, hess_y), hess_w, hess_h, fill=True,
                       facecolor=C_RED, edgecolor=EC_RED, alpha=0.3, lw=2)
ax2.add_patch(rect)
ax2.text(hess_x + hess_w/2, hess_y + hess_h/2 + 0.3,
         '$H = \\nabla^2_x E$\n1092 × 1092', ha='center', fontsize=11, fontweight='bold')
ax2.text(hess_x + hess_w/2, hess_y + 0.3,
         'tr(H) = Σᵢ Hᵢᵢ\nNeeds D backward passes!', ha='center', fontsize=9, color=EC_RED)

# Diagonal highlight
for i in range(8):
    frac = i / 8
    x = hess_x + 0.15 + frac * (hess_w - 0.3)
    y = hess_y + hess_h - 0.15 - frac * (hess_h - 0.3)
    ax2.plot(x, y, 's', color=EC_RED, markersize=6)

# Right: Hutchinson estimator (cheap)
ax2.text(7.5, 4.7, 'Hutchinson estimator (one random vector)', ha='center', fontsize=11,
         fontweight='bold', color=EC_GREEN)

# Random vector v
draw_box(ax2, (5.8, 3.5), 1.2, 0.6, 'v ~ N(0,I)\nrandom', C_GREEN, EC_GREEN, fontsize=9)

# Steps
ax2.text(7.5, 4.15,
    '1) Compute  g = grad_x E    (one backward pass)\n'
    '2) Compute  v.g = scalar    (dot product)\n'
    '3) Differentiate  d(v.g)/dx = Hv    (one more backward)\n'
    '4) tr(H) ~ v . (Hv)    (dot product)',
    ha='center', va='top', fontsize=10, fontfamily='monospace',
    bbox=dict(boxstyle='round', facecolor=C_GREEN, edgecolor=EC_GREEN, alpha=0.2))

# Cost comparison
draw_box(ax2, (0.8, 0.3), 3.3, 0.9, 'Full trace: D=1092 backward passes\nCost: O(D²) — impossible',
         C_RED, EC_RED, fontsize=10, text_color=EC_RED)
draw_box(ax2, (5.5, 0.3), 3.8, 0.9,
         'Hutchinson: 2 backward passes total\nCost: O(1) — same as DSM!\nBUT: still 2nd-order through E(x)',
         C_ORANGE, EC_ORANGE, fontsize=10, text_color='#333')

# Arrow between
ax2.text(4.8, 0.75, 'vs', ha='center', fontsize=14, fontweight='bold', color='#666')

# Divider
ax2.axvline(x=5.0, ymin=0.15, ymax=0.82, color='#CCC', linestyle='--', lw=1)


# ========================================================================
# PANEL 3: Noise Conditional Score Network (NCSN)
# ========================================================================
ax3 = fig.add_axes([0.03, 0.02, 0.94, 0.30])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 6.5)
ax3.axis('off')
ax3.set_title('3. Noise Conditional Score Network (NCSN)', fontsize=16,
              fontweight='bold', pad=15, loc='left', color=EC_ORANGE)

# The key concern
ax3.text(5.0, 6.1,
    'User concern: "Doesn\'t this change the network from predicting energy to predicting score?"  → YES.',
    ha='center', fontsize=11, style='italic', color=EC_RED,
    bbox=dict(boxstyle='round', facecolor=C_RED, edgecolor=EC_RED, alpha=0.2))

# Left: standard NCSN
ax3.text(2.5, 5.4, 'Standard NCSN (Song & Ermon 2019)', ha='center', fontsize=11, fontweight='bold')

# Multi-scale noise illustration
sigmas = [3.0, 1.5, 0.5, 0.1]
colors_sigma = ['#FFCDD2', '#EF9A9A', '#E57373', '#D32F2F']
x_base = np.linspace(-3, 3, 200)

for i, (sig, col) in enumerate(zip(sigmas, colors_sigma)):
    y_offset = 4.6 - i * 0.55
    # Smoothed distribution
    y = np.exp(-x_base**2 / (2*sig**2))
    y = y / y.max() * 0.4
    ax3.fill_between(x_base * 0.3 + 2.5, y_offset, y_offset + y, alpha=0.5, color=col)
    ax3.text(4.2, y_offset + 0.1, f'σ={sig}', fontsize=9, color=col)

ax3.text(2.5, 2.4, 'Network: $s_\\theta(x, \\sigma) \\rightarrow \\mathbb{R}^D$\nOutputs score VECTOR directly\n⚠ Not an energy function!',
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor=C_ORANGE, edgecolor=EC_ORANGE, alpha=0.3))

# Right: comparison table
ax3.text(7.5, 5.4, 'How it compares to our setup', ha='center', fontsize=11, fontweight='bold')

table_data = [
    ('', 'Our PEM', 'NCSN'),
    ('Output', 'E(x) ∈ R¹', 's(x) ∈ R^D'),
    ('Score', '−∇ₓE (autograd)', 'Direct output'),
    ('DSM grad', '2nd order (broken)', '1st order (works)'),
    ('Physics', 'Energy = interpretable', 'Score ≠ energy'),
    ('Inference', 'E(native) vs E(decoy)', '???'),
]

for i, row in enumerate(table_data):
    y = 5.1 - i * 0.45
    color = '#E0E0E0' if i == 0 else ('#FFF' if i % 2 == 0 else '#F5F5F5')
    weight = 'bold' if i == 0 else 'normal'
    ax3.text(5.8, y, row[0], fontsize=9, fontweight=weight, va='center')
    ax3.text(7.2, y, row[1], fontsize=9, fontweight=weight, va='center', ha='center',
             color=EC_BLUE if i > 0 else 'black')
    ax3.text(8.8, y, row[2], fontsize=9, fontweight=weight, va='center', ha='center',
             color=EC_ORANGE if i > 0 else 'black')

# Bottom: verdict for each
ax3.plot([0.3, 9.7], [1.5, 1.5], '-', color='#CCC', lw=1)

ax3.text(0.5, 1.1, 'VERDICT:', fontsize=12, fontweight='bold', color='#333')

verdicts = [
    ('Score Head', '✓ Best fit', 'Keeps E(x) for InfoNCE.\nAdds s_θ for DSM. First-order.\nShared backbone learns from both.', C_GREEN, EC_GREEN),
    ('Hutchinson', '~ Partial fix', 'Reduces cost O(D²)→O(1).\nStill 2nd-order through E.\nSame vanishing Hessian problem.', C_ORANGE, EC_ORANGE),
    ('NCSN', '✗ Wrong tool', 'Replaces energy with score.\nBreaks InfoNCE pipeline.\nNot what we need.', C_RED, EC_RED),
]

for i, (name, verdict, detail, fc, ec) in enumerate(verdicts):
    x = 1.2 + i * 3.0
    draw_box(ax3, (x, 0.05), 2.5, 1.3, '', fc, ec, alpha=0.3)
    ax3.text(x + 1.25, 1.15, f'{name}: {verdict}', ha='center', fontsize=10, fontweight='bold', color=ec)
    ax3.text(x + 1.25, 0.55, detail, ha='center', va='center', fontsize=8.5, color='#333')


plt.savefig('/Users/shahar/Documents/Personal/DeepEF/figures/dsm_alternatives.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.savefig('/Users/shahar/Documents/Personal/DeepEF/figures/dsm_alternatives.pdf', bbox_inches='tight',
            facecolor='white')
print("Saved to figures/dsm_alternatives.png and .pdf")
