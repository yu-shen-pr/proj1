"""
Motivation figure v4 - Left image + Right text box style
(a) Traditional method problem
(b) Our method advantage
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches


def _image_box(ax, x, y, w, h, label):
    """Draw an image placeholder box"""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#EEEEEE",
        edgecolor="black",
        linewidth=1.0
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)


def _text_box(ax, x, y, w, h, lines, *, highlights=None):
    """
    Draw a text explanation box with optional highlighted keywords.
    lines: list of str
    highlights: dict mapping keyword -> color (e.g., {"motion": "red"})
    """
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="white",
        edgecolor="black",
        linewidth=1.0
    )
    ax.add_patch(rect)

    if highlights is None:
        highlights = {}

    line_height = h / (len(lines) + 1)
    for i, line in enumerate(lines):
        y_pos = y + h - (i + 1) * line_height
        ax.text(x + 0.02, y_pos, line, ha="left", va="center", fontsize=9, wrap=True)


def _arrow(ax, x0, y0, x1, y1, style="->", color="black", lw=1.0):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle=style, lw=lw, color=color)
    )


def main():
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    fig.patch.set_facecolor("white")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ============ (a) Traditional YOLO Detection Problem ============
    ax.text(0.5, 0.97, "(a) Traditional YOLO Detection", fontsize=12, ha="center", fontweight="bold")

    # Left: two consecutive frames + detection result
    _image_box(ax, 0.03, 0.72, 0.12, 0.18, "[Frame $I_{t-1}$]")
    _image_box(ax, 0.17, 0.72, 0.12, 0.18, "[Frame $I_t$]")
    
    # Arrow showing only I_t is used
    _arrow(ax, 0.29, 0.81, 0.32, 0.81)
    
    # Detection result placeholder
    _image_box(ax, 0.32, 0.72, 0.12, 0.18, "[Detection\nResult]")

    # Right: text explanation box
    problem_text = FancyBboxPatch(
        (0.48, 0.68), 0.50, 0.26,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="white",
        edgecolor="black",
        linewidth=1.0
    )
    ax.add_patch(problem_text)

    ax.text(0.50, 0.90, "Problems:", fontsize=10, fontweight="bold", va="center")
    ax.text(0.50, 0.84, "1. Only uses ", fontsize=9, va="center")
    ax.text(0.62, 0.84, "single-frame RGB", fontsize=9, va="center", style="italic", color="red")
    ax.text(0.50, 0.78, "2. Ignores ", fontsize=9, va="center")
    ax.text(0.59, 0.78, "temporal motion information", fontsize=9, va="center", style="italic", color="red")
    ax.text(0.50, 0.72, "3. May ", fontsize=9, va="center")
    ax.text(0.55, 0.72, "miss moving objects", fontsize=9, va="center", style="italic", color="red")
    ax.text(0.70, 0.72, " or have", fontsize=9, va="center")
    ax.text(0.50, 0.66, "    ", fontsize=9, va="center")
    ax.text(0.52, 0.66, "false positives in static regions", fontsize=9, va="center", style="italic", color="red")

    # Dashed line separator
    ax.plot([0.02, 0.98], [0.62, 0.62], "k--", lw=0.8)

    # ============ (b) Our Motion-Augmented YOLO ============
    ax.text(0.5, 0.58, "(b) Our Motion-Augmented YOLO", fontsize=12, ha="center", fontweight="bold")

    # Left: frames + motion + detection
    _image_box(ax, 0.03, 0.32, 0.12, 0.18, "[Frame $I_{t-1}$]")
    _image_box(ax, 0.17, 0.32, 0.12, 0.18, "[Frame $I_t$]")

    # Arrow to motion
    _arrow(ax, 0.10, 0.32, 0.235, 0.28)
    _arrow(ax, 0.24, 0.32, 0.235, 0.28)

    # Motion image
    _image_box(ax, 0.17, 0.08, 0.12, 0.18, "[Motion\nu,v,attn]")

    # Arrow to detection
    _arrow(ax, 0.29, 0.41, 0.32, 0.41)
    _arrow(ax, 0.29, 0.17, 0.32, 0.30)

    # Detection result
    _image_box(ax, 0.32, 0.25, 0.12, 0.18, "[Detection\nResult]")

    # Right: text explanation box
    solution_text = FancyBboxPatch(
        (0.48, 0.08), 0.50, 0.46,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="white",
        edgecolor="black",
        linewidth=1.0
    )
    ax.add_patch(solution_text)

    ax.text(0.50, 0.50, "Our Solution:", fontsize=10, fontweight="bold", va="center")
    
    ax.text(0.50, 0.44, "1. ", fontsize=9, va="center")
    ax.text(0.52, 0.44, "RAFT optical flow", fontsize=9, va="center", style="italic", color="green")
    ax.text(0.68, 0.44, " extracts motion between", fontsize=9, va="center")
    ax.text(0.52, 0.39, "   consecutive frames $I_{t-1}$ and $I_t$", fontsize=9, va="center")

    ax.text(0.50, 0.33, "2. ", fontsize=9, va="center")
    ax.text(0.52, 0.33, "dt-normalization", fontsize=9, va="center", style="italic", color="green")
    ax.text(0.67, 0.33, " compensates variable", fontsize=9, va="center")
    ax.text(0.52, 0.28, "   frame intervals: $\\hat{F} = F / \\Delta t$", fontsize=9, va="center")

    ax.text(0.50, 0.22, "3. ", fontsize=9, va="center")
    ax.text(0.52, 0.22, "Attention gate", fontsize=9, va="center", style="italic", color="green")
    ax.text(0.65, 0.22, " filters noisy motion via", fontsize=9, va="center")
    ax.text(0.52, 0.17, "   magnitude or coherence gating", fontsize=9, va="center")

    ax.text(0.50, 0.11, "4. ", fontsize=9, va="center")
    ax.text(0.52, 0.11, "6-channel fusion", fontsize=9, va="center", style="italic", color="green")
    ax.text(0.67, 0.11, " concatenates RGB +", fontsize=9, va="center")
    ax.text(0.52, 0.06, "   motion(u,v,attn) as YOLO input", fontsize=9, va="center")

    plt.tight_layout()
    fig.savefig("motivation_figure_v4.png", dpi=300, bbox_inches="tight")
    fig.savefig("motivation_figure_v4.pdf", bbox_inches="tight")
    print("Saved: motivation_figure_v4.png and motivation_figure_v4.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
