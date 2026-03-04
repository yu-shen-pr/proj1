"""
Motivation figure for the paper:
(a) Traditional approaches and their problems
(b) Our proposed solution with key innovations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set up figure
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.patch.set_facecolor('white')

# Colors
COLOR_RGB = '#4CAF50'       # Green for RGB
COLOR_MOTION = '#2196F3'    # Blue for motion
COLOR_PROBLEM = '#F44336'   # Red for problems
COLOR_SOLUTION = '#FF9800'  # Orange for solutions
COLOR_BOX = '#E3F2FD'       # Light blue for boxes
COLOR_ARROW = '#333333'     # Dark gray for arrows
COLOR_HIGHLIGHT = '#FFEB3B' # Yellow for highlights

def draw_box(ax, x, y, w, h, text, color='white', edgecolor='black', fontsize=10, bold=False):
    """Draw a rounded box with text"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                         facecolor=color, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, 
            fontweight=weight, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color=COLOR_ARROW, style='->'):
    """Draw an arrow"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=2))

def draw_crossed_arrow(ax, x1, y1, x2, y2):
    """Draw a crossed-out arrow (indicating problem)"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=COLOR_PROBLEM, lw=2, linestyle='--'))
    # Draw X mark
    mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
    ax.plot([mid_x-0.02, mid_x+0.02], [mid_y-0.03, mid_y+0.03], color=COLOR_PROBLEM, lw=3)
    ax.plot([mid_x-0.02, mid_x+0.02], [mid_y+0.03, mid_y-0.03], color=COLOR_PROBLEM, lw=3)

# ============ (a) Traditional Approaches & Problems ============
ax1 = axes[0]
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.set_title('(a) Traditional Approaches: Key Challenges', fontsize=14, fontweight='bold', pad=10)

# Row 1: RGB-only approach
draw_box(ax1, 0.02, 0.7, 0.12, 0.2, 'RGB\nFrame\n$I_t$', COLOR_RGB, fontsize=9)
draw_arrow(ax1, 0.14, 0.8, 0.22, 0.8)
draw_box(ax1, 0.22, 0.72, 0.13, 0.16, 'YOLO\nDetector', '#E0E0E0', fontsize=9)
draw_arrow(ax1, 0.35, 0.8, 0.43, 0.8)
draw_box(ax1, 0.43, 0.72, 0.13, 0.16, 'Detection\nResult', '#C8E6C9', fontsize=9)

# Problem annotation for RGB-only
ax1.text(0.60, 0.82, '[X] Ignores motion cues', fontsize=10, color=COLOR_PROBLEM, fontweight='bold')
ax1.text(0.60, 0.74, '[X] May miss moving objects', fontsize=10, color=COLOR_PROBLEM)

# Row 2: Motion-only approach (replacement)
draw_box(ax1, 0.02, 0.38, 0.12, 0.2, 'Motion\nImage\n$(u, v)$', COLOR_MOTION, fontsize=9)
draw_arrow(ax1, 0.14, 0.48, 0.22, 0.48)
draw_box(ax1, 0.22, 0.4, 0.13, 0.16, 'YOLO\n(3ch)', '#E0E0E0', fontsize=9)
draw_arrow(ax1, 0.35, 0.48, 0.43, 0.48)
draw_box(ax1, 0.43, 0.4, 0.13, 0.16, 'Detection\nResult', '#FFCDD2', fontsize=9)

# Problem annotation for Motion-only
ax1.text(0.60, 0.52, '[X] Loses appearance (texture, color)', fontsize=10, color=COLOR_PROBLEM, fontweight='bold')
ax1.text(0.60, 0.44, '[X] Precision drops', fontsize=10, color=COLOR_PROBLEM)

# Row 3: Naive motion integration problems
draw_box(ax1, 0.02, 0.06, 0.12, 0.2, 'Frame\n$I_{t-1}$\n$I_t$', '#BBDEFB', fontsize=8)
draw_arrow(ax1, 0.14, 0.16, 0.22, 0.16)
draw_box(ax1, 0.22, 0.08, 0.13, 0.16, 'RAFT\nFlow', COLOR_MOTION, fontsize=9)
draw_arrow(ax1, 0.35, 0.16, 0.43, 0.16)
draw_box(ax1, 0.43, 0.08, 0.18, 0.16, 'Motion\nMagnitude\n$||F||$', '#90CAF9', fontsize=9)

# Problem annotations for naive integration
problem_box = FancyBboxPatch((0.65, 0.02), 0.32, 0.28, boxstyle="round,pad=0.02",
                              facecolor='#FFEBEE', edgecolor=COLOR_PROBLEM, linewidth=2)
ax1.add_patch(problem_box)
ax1.text(0.81, 0.26, 'Key Problems:', fontsize=10, ha='center', fontweight='bold', color=COLOR_PROBLEM)
ax1.text(0.67, 0.19, '① $\\Delta t$ varies → magnitude drifts', fontsize=9)
ax1.text(0.67, 0.12, '② HSV augmentation corrupts motion', fontsize=9)
ax1.text(0.67, 0.05, '③ 3ch metadata hardcoded in YOLO', fontsize=9)


# ============ (b) Our Proposed Solution ============
ax2 = axes[1]
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.set_title('(b) Our Solution: Motion-Augmented YOLO with Key Innovations', fontsize=14, fontweight='bold', pad=10)

# Input frames
draw_box(ax2, 0.01, 0.55, 0.08, 0.35, 'Frame\nPair\n$I_{t-1}$\n$I_t$', '#E8F5E9', fontsize=8)

# RAFT + dt-norm branch
draw_arrow(ax2, 0.09, 0.72, 0.14, 0.72)
draw_box(ax2, 0.14, 0.64, 0.1, 0.16, 'RAFT\nFlow', COLOR_MOTION, fontsize=9)

# dt-normalization (innovation 1)
draw_arrow(ax2, 0.24, 0.72, 0.29, 0.72)
innovation_box1 = FancyBboxPatch((0.29, 0.62), 0.12, 0.20, boxstyle="round,pad=0.01",
                                  facecolor='#FFF3E0', edgecolor=COLOR_SOLUTION, linewidth=2.5)
ax2.add_patch(innovation_box1)
ax2.text(0.35, 0.77, '★ Innovation 1', fontsize=8, ha='center', fontweight='bold', color=COLOR_SOLUTION)
ax2.text(0.35, 0.70, 'dt-Norm', fontsize=10, ha='center', fontweight='bold')
ax2.text(0.35, 0.64, '$\\tilde{F}=F/\\Delta t$', fontsize=9, ha='center')

# Attention (innovation 2)
draw_arrow(ax2, 0.41, 0.72, 0.46, 0.72)
innovation_box2 = FancyBboxPatch((0.46, 0.62), 0.12, 0.20, boxstyle="round,pad=0.01",
                                  facecolor='#FFF3E0', edgecolor=COLOR_SOLUTION, linewidth=2.5)
ax2.add_patch(innovation_box2)
ax2.text(0.52, 0.77, '★ Innovation 2', fontsize=8, ha='center', fontweight='bold', color=COLOR_SOLUTION)
ax2.text(0.52, 0.70, 'Attention', fontsize=10, ha='center', fontweight='bold')
ax2.text(0.52, 0.64, 'Gate', fontsize=9, ha='center')

# RGB branch
draw_arrow(ax2, 0.09, 0.55, 0.58, 0.4, style='->')
ax2.text(0.25, 0.42, 'RGB preserved', fontsize=9, color=COLOR_RGB, fontweight='bold', rotation=0)

# 6ch Fusion (innovation 3)
innovation_box3 = FancyBboxPatch((0.60, 0.35), 0.14, 0.50, boxstyle="round,pad=0.01",
                                  facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2.5)
ax2.add_patch(innovation_box3)
ax2.text(0.67, 0.82, '★ Innovation 3', fontsize=8, ha='center', fontweight='bold', color='#2E7D32')
ax2.text(0.67, 0.74, '6-Channel', fontsize=10, ha='center', fontweight='bold')
ax2.text(0.67, 0.66, 'Fusion', fontsize=10, ha='center', fontweight='bold')

# 6ch breakdown
ax2.text(0.67, 0.56, 'R G B', fontsize=9, ha='center', color=COLOR_RGB, fontweight='bold')
ax2.text(0.67, 0.49, 'u  v  attn', fontsize=9, ha='center', color=COLOR_MOTION, fontweight='bold')
ax2.text(0.67, 0.40, '(motion)', fontsize=8, ha='center', color='gray')

# Arrows to fusion
draw_arrow(ax2, 0.58, 0.72, 0.60, 0.65)

# YOLO detector
draw_arrow(ax2, 0.74, 0.60, 0.80, 0.60)
draw_box(ax2, 0.80, 0.50, 0.10, 0.20, 'YOLO\n(6ch)', '#E0E0E0', fontsize=10, bold=True)

# Output
draw_arrow(ax2, 0.90, 0.60, 0.95, 0.60)
draw_box(ax2, 0.92, 0.72, 0.07, 0.18, '↑R\n↑mAP50', '#C8E6C9', fontsize=9, bold=True)
draw_box(ax2, 0.92, 0.50, 0.07, 0.18, '↑P\nRobust', '#C8E6C9', fontsize=9, bold=True)

# Key results annotation
result_box = FancyBboxPatch((0.01, 0.02), 0.45, 0.28, boxstyle="round,pad=0.02",
                             facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2)
ax2.add_patch(result_box)
ax2.text(0.235, 0.26, 'Key Results (vs. Raw RGB Baseline)', fontsize=10, ha='center', fontweight='bold', color='#2E7D32')
ax2.text(0.03, 0.18, '• RAFT+Attn+dt: Best Recall (0.792 vs 0.756)', fontsize=9)
ax2.text(0.03, 0.11, '• RAFT+Attn+dt: Best mAP50 (0.893 vs 0.885)', fontsize=9)
ax2.text(0.03, 0.04, '• 6ch Fusion: Highest Precision (0.967), appearance-robust', fontsize=9)

# Engineering innovations
eng_box = FancyBboxPatch((0.50, 0.02), 0.48, 0.28, boxstyle="round,pad=0.02",
                          facecolor='#FFF8E1', edgecolor=COLOR_SOLUTION, linewidth=2)
ax2.add_patch(eng_box)
ax2.text(0.74, 0.26, 'Engineering Framework', fontsize=10, ha='center', fontweight='bold', color='#E65100')
ax2.text(0.52, 0.18, '• Disable HSV/erasing augmentation for motion', fontsize=9)
ax2.text(0.52, 0.11, '• 3→6 channel first-conv expansion', fontsize=9)
ax2.text(0.52, 0.04, '• Validation warmup fix for 6ch metadata', fontsize=9)

plt.tight_layout()
plt.savefig('motivation_figure.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('motivation_figure.pdf', bbox_inches='tight', facecolor='white')
print("Saved: motivation_figure.png and motivation_figure.pdf")
plt.show()
