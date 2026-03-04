"""
Framework figure - Layout like reference: Left big module + Right top/bottom modules
3 Innovations: dt-Norm, Attention Gate, 6-Channel Fusion (includes YOLO mod)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np


def _box(ax, x, y, w, h, text, *, fc="white", ec="black", lw=1.0, fontsize=9, weight=None):
    r = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.008",
        facecolor=fc, edgecolor=ec, linewidth=lw
    )
    ax.add_patch(r)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight)
    return r


def _arrow(ax, x0, y0, x1, y1, *, lw=1.0, color="black", style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle=style, lw=lw, color=color))


def _label(ax, x, y, text, fontsize=8, color="black", ha="center", va="center", **kwargs):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color, **kwargs)


def main() -> int:
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    fig.patch.set_facecolor("white")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ==================== LEFT: dt-Normalization (big module) ====================
    _label(ax, 0.22, 0.97, "Temporal-Normalized Flow (TNF)", fontsize=14, fontweight="bold", color="#2E7D32")
    
    mod1 = FancyBboxPatch((0.02, 0.03), 0.42, 0.92,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=2.0)
    ax.add_patch(mod1)

    # ---- Problem Section ----
    _label(ax, 0.23, 0.92, "Problem: Variable Frame Interval", fontsize=11, fontweight="bold", color="#2E7D32")
    
    # Input frames illustration
    _box(ax, 0.05, 0.78, 0.08, 0.10, "$I_{t-1}$", fc="#E8E8E8", fontsize=10)
    _box(ax, 0.16, 0.78, 0.08, 0.10, "$I_t$", fc="#E8E8E8", fontsize=10)
    _box(ax, 0.30, 0.78, 0.08, 0.10, "$I_{t+1}$", fc="#E8E8E8", fontsize=10)
    
    # Timeline with variable dt
    ax.plot([0.09, 0.09], [0.72, 0.76], 'k-', lw=2)
    ax.plot([0.20, 0.20], [0.72, 0.76], 'k-', lw=2)
    ax.plot([0.34, 0.34], [0.72, 0.76], 'k-', lw=2)
    ax.plot([0.09, 0.34], [0.74, 0.74], 'k-', lw=1.0)
    
    _label(ax, 0.145, 0.76, "$\\Delta t_1$=0.1s", fontsize=8, color="red")
    _label(ax, 0.27, 0.76, "$\\Delta t_2$=0.3s", fontsize=8, color="red")
    _label(ax, 0.23, 0.70, "Different intervals cause inconsistent flow magnitude!", fontsize=8, color="red")

    # ---- RAFT Section ----
    _label(ax, 0.23, 0.64, "Step 1: Extract Optical Flow (RAFT)", fontsize=10, fontweight="bold")
    
    _box(ax, 0.08, 0.52, 0.12, 0.08, "RAFT\n(pretrained)", fc="#FFF3E0", ec="#E65100", fontsize=8)
    _arrow(ax, 0.13, 0.78, 0.13, 0.60)
    _arrow(ax, 0.20, 0.52, 0.24, 0.52)
    
    # Flow output
    _box(ax, 0.24, 0.48, 0.14, 0.12, "$F = (u, v)$\nraw flow", fc="white", ec="#E65100", fontsize=9)
    _label(ax, 0.31, 0.45, "[H,W,2]", fontsize=7, color="gray")

    # ---- Normalization Section ----
    _label(ax, 0.23, 0.42, "Step 2: TNF (divide by $\\Delta t$)", fontsize=10, fontweight="bold")
    
    # Formula box
    norm_box = FancyBboxPatch((0.06, 0.26), 0.30, 0.12,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        facecolor="white", edgecolor="#2E7D32", linewidth=1.5)
    ax.add_patch(norm_box)
    
    _label(ax, 0.21, 0.34, "$\\hat{u} = \\frac{u}{\\Delta t}, \\quad \\hat{v} = \\frac{v}{\\Delta t}$", fontsize=12)
    _label(ax, 0.21, 0.28, "Convert to velocity (pixels/second)", fontsize=8, color="gray")

    # ---- Effect Section ----
    _label(ax, 0.23, 0.22, "Effect: Consistent Velocity Representation", fontsize=10, fontweight="bold")
    
    # Before
    _box(ax, 0.05, 0.10, 0.08, 0.08, "slow\n$\\Delta t$=0.1", fc="#FFCDD2", fontsize=7)
    _box(ax, 0.14, 0.10, 0.08, 0.08, "fast\n$\\Delta t$=0.3", fc="#FFCDD2", fontsize=7)
    _label(ax, 0.135, 0.06, "Before: different scale", fontsize=7, color="gray")
    
    _arrow(ax, 0.24, 0.14, 0.28, 0.14)
    
    # After
    _box(ax, 0.29, 0.10, 0.08, 0.08, "norm\nvel", fc="#C8E6C9", fontsize=7)
    _box(ax, 0.38, 0.10, 0.08, 0.08, "norm\nvel", fc="#C8E6C9", fontsize=7)
    _label(ax, 0.375, 0.06, "After: same scale", fontsize=7, color="gray")

    # ==================== RIGHT TOP: TNF Consistency Alignment ====================
    _label(ax, 0.72, 0.97, "Temporal Consistency Alignment (TNF)", fontsize=14, fontweight="bold", color="#1565C0")
    
    mod2 = FancyBboxPatch((0.46, 0.52), 0.52, 0.43,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=2.0)
    ax.add_patch(mod2)

    # Cross-Δt illustration
    _label(ax, 0.72, 0.92, "Cross-Δt Inconsistency", fontsize=10, fontweight="bold", color="#1565C0")
    _box(ax, 0.48, 0.82, 0.12, 0.07, "Δt=1\nraw F", fc="#FFCDD2", fontsize=8)
    _box(ax, 0.62, 0.82, 0.12, 0.07, "Δt=5\nraw F", fc="#FFCDD2", fontsize=8)
    _arrow(ax, 0.60, 0.80, 0.72, 0.80)
    _label(ax, 0.73, 0.80, "magnitude shift", fontsize=7, color="gray")

    # Histograms before/after TNF (schematic)
    _label(ax, 0.58, 0.74, "Before TNF", fontsize=9, fontweight="bold")
    opt1 = FancyBboxPatch((0.48, 0.64), 0.20, 0.08,
        boxstyle="round,pad=0.005,rounding_size=0.007",
        facecolor="white", edgecolor="#1565C0", linewidth=1.2)
    ax.add_patch(opt1)
    for i, h in enumerate([0.02, 0.06, 0.10, 0.05, 0.08]):
        ax.add_patch(Rectangle((0.495 + i*0.035, 0.645), 0.02, h, facecolor="#90CAF9", edgecolor="#1565C0", linewidth=0.8))

    _label(ax, 0.85, 0.74, "After TNF", fontsize=9, fontweight="bold")
    opt2 = FancyBboxPatch((0.74, 0.64), 0.22, 0.08,
        boxstyle="round,pad=0.005,rounding_size=0.007",
        facecolor="white", edgecolor="#1565C0", linewidth=1.2)
    ax.add_patch(opt2)
    for i, h in enumerate([0.01, 0.04, 0.11, 0.04, 0.01]):
        ax.add_patch(Rectangle((0.755 + i*0.035, 0.645), 0.02, h, facecolor="#42A5F5", edgecolor="#1565C0", linewidth=0.8))

    _label(ax, 0.72, 0.58, "Aligned distribution → Δt-invariant features", fontsize=9, fontweight="bold")

    # ==================== RIGHT BOTTOM: Attention Gating + Detector Integration ====================
    _label(ax, 0.72, 0.49, "Attention Gating + Detector Integration", fontsize=14, fontweight="bold", color="#C2185B")
    
    mod3 = FancyBboxPatch((0.46, 0.03), 0.52, 0.44,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor="#FCE4EC", edgecolor="#C2185B", linewidth=2.0)
    ax.add_patch(mod3)

    # Attention options (magnitude & coherence)
    _label(ax, 0.58, 0.44, "Magnitude Gate", fontsize=9, fontweight="bold")
    _box(ax, 0.48, 0.36, 0.20, 0.07, "$A = \\|\\hat{F}\\|^\\gamma$", fc="white", ec="#C2185B", fontsize=10)
    _label(ax, 0.58, 0.33, "$\\gamma \\in (0,1]$", fontsize=7, color="gray")

    _label(ax, 0.84, 0.44, "Coherence Gate", fontsize=9, fontweight="bold")
    _box(ax, 0.74, 0.36, 0.22, 0.07, "$A = M \\cdot \\sigma(C)$", fc="white", ec="#C2185B", fontsize=10)
    _label(ax, 0.85, 0.33, "$M$=magnitude, $C$=local coherence", fontsize=7, color="gray")

    # Integration choices
    _label(ax, 0.72, 0.29, "Detector Integration", fontsize=9, fontweight="bold", color="#C2185B")
    _box(ax, 0.52, 0.22, 0.16, 0.06, "Motion-only (3ch)", fc="#C8E6C9", fontsize=8)
    _box(ax, 0.74, 0.22, 0.20, 0.06, "RGB+Motion (6ch, optional)", fc="#E1BEE7", fontsize=8)
    _arrow(ax, 0.60, 0.25, 0.66, 0.13)
    _arrow(ax, 0.84, 0.25, 0.86, 0.13)

    # YOLO block
    _box(ax, 0.66, 0.10, 0.12, 0.06, "YOLOv8\n(backbone+head)", fc="#FFF9C4", fontsize=7)
    _label(ax, 0.72, 0.07, "train with\nModality-Aware Aug.", fontsize=6, color="gray")

    plt.tight_layout()
    fig.savefig("framework_figure.png", dpi=300, bbox_inches="tight")
    fig.savefig("framework_figure.pdf", bbox_inches="tight")
    fig.savefig("framework_fig.png", dpi=300, bbox_inches="tight")
    print("Saved: framework_figure.png, framework_figure.pdf, framework_fig.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
