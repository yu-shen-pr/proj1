"""
Motivation figure - clean style matching the reference image
(a) Traditional approach problem
(b) Our proposed solution
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# Set up figure - similar aspect ratio to reference
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
fig.patch.set_facecolor('white')

# ============ (a) Traditional YOLO Detection ============
ax1 = axes[0]
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.text(0.5, 0.95, '(a) Traditional YOLO Detection', fontsize=12, ha='center', fontweight='bold')

# Image placeholder (frame t-1)
rect1 = Rectangle((0.05, 0.35), 0.15, 0.45, fill=True, facecolor='#EEEEEE', edgecolor='black', linewidth=1)
ax1.add_patch(rect1)
ax1.text(0.125, 0.575, '[Image\n$I_{t-1}$]', fontsize=9, ha='center', va='center')
ax1.text(0.125, 0.28, '$I_{t-1}$', fontsize=10, ha='center')

# Image placeholder (frame t)
rect2 = Rectangle((0.25, 0.35), 0.15, 0.45, fill=True, facecolor='#EEEEEE', edgecolor='black', linewidth=1)
ax1.add_patch(rect2)
ax1.text(0.325, 0.575, '[Image\n$I_t$]', fontsize=9, ha='center', va='center')
ax1.text(0.325, 0.28, '$I_t$', fontsize=10, ha='center')

# Arrow to YOLO
ax1.annotate('', xy=(0.52, 0.575), xytext=(0.42, 0.575),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# YOLO box
rect3 = Rectangle((0.52, 0.45), 0.12, 0.25, fill=True, facecolor='white', edgecolor='black', linewidth=1)
ax1.add_patch(rect3)
ax1.text(0.58, 0.575, 'YOLO\n(RGB)', fontsize=10, ha='center', va='center')

# Arrow to output
ax1.annotate('', xy=(0.76, 0.575), xytext=(0.66, 0.575),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Detection result placeholder
rect4 = Rectangle((0.76, 0.35), 0.18, 0.45, fill=True, facecolor='#EEEEEE', edgecolor='black', linewidth=1)
ax1.add_patch(rect4)
ax1.text(0.85, 0.575, '[Detection\nResult]', fontsize=9, ha='center', va='center')

# Problem text box on the right
problem_box = Rectangle((0.55, 0.02), 0.42, 0.25, fill=True, facecolor='white', edgecolor='black', linewidth=1)
ax1.add_patch(problem_box)
ax1.text(0.76, 0.22, 'Only uses single RGB frame', fontsize=9, ha='center', style='italic')
ax1.text(0.76, 0.13, 'Ignores ', fontsize=9, ha='center')
ax1.text(0.82, 0.13, 'motion information', fontsize=9, ha='left', color='red', style='italic')
ax1.text(0.76, 0.04, 'between consecutive frames', fontsize=9, ha='center')

# Dashed line separator
ax1.axhline(y=0.0, color='gray', linestyle='--', linewidth=1)

# ============ (b) Our Motion-Augmented YOLO ============
ax2 = axes[1]
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.text(0.5, 0.95, '(b) Our Motion-Augmented YOLO', fontsize=12, ha='center', fontweight='bold')

# Frame t-1 placeholder
rect_t1 = Rectangle((0.02, 0.55), 0.1, 0.3, fill=True, facecolor='#EEEEEE', edgecolor='black', linewidth=1)
ax2.add_patch(rect_t1)
ax2.text(0.07, 0.7, '[Image\n$I_{t-1}$]', fontsize=8, ha='center', va='center')

# Frame t placeholder  
rect_t = Rectangle((0.02, 0.2), 0.1, 0.3, fill=True, facecolor='#EEEEEE', edgecolor='black', linewidth=1)
ax2.add_patch(rect_t)
ax2.text(0.07, 0.35, '[Image\n$I_t$]', fontsize=8, ha='center', va='center')

# Arrow from frames to RAFT
ax2.annotate('', xy=(0.18, 0.52), xytext=(0.13, 0.65),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
ax2.annotate('', xy=(0.18, 0.52), xytext=(0.13, 0.40),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# RAFT Optical Flow box
rect_raft = Rectangle((0.18, 0.42), 0.12, 0.2, fill=True, facecolor='white', edgecolor='black', linewidth=1)
ax2.add_patch(rect_raft)
ax2.text(0.24, 0.52, 'RAFT\nOptical\nFlow', fontsize=8, ha='center', va='center')

# Arrow to dt-norm
ax2.annotate('', xy=(0.36, 0.52), xytext=(0.31, 0.52),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# dt-normalization box (Innovation 1)
rect_dt = Rectangle((0.36, 0.42), 0.11, 0.2, fill=True, facecolor='white', edgecolor='black', linewidth=1)
ax2.add_patch(rect_dt)
ax2.text(0.415, 0.55, 'dt-Norm', fontsize=9, ha='center', va='center', fontweight='bold')
ax2.text(0.415, 0.46, '$F/\\Delta t$', fontsize=9, ha='center', va='center')

# Arrow to attention
ax2.annotate('', xy=(0.53, 0.52), xytext=(0.48, 0.52),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# Attention box (Innovation 2)
rect_attn = Rectangle((0.53, 0.42), 0.11, 0.2, fill=True, facecolor='white', edgecolor='black', linewidth=1)
ax2.add_patch(rect_attn)
ax2.text(0.585, 0.55, 'Attention', fontsize=9, ha='center', va='center', fontweight='bold')
ax2.text(0.585, 0.46, 'Gate', fontsize=9, ha='center', va='center')

# Motion image output (u, v, attn)
ax2.annotate('', xy=(0.70, 0.52), xytext=(0.65, 0.52),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# Motion image placeholder
rect_motion = Rectangle((0.70, 0.42), 0.1, 0.2, fill=True, facecolor='#EEEEEE', edgecolor='black', linewidth=1)
ax2.add_patch(rect_motion)
ax2.text(0.75, 0.52, '[Motion\nu, v, attn]', fontsize=8, ha='center', va='center')

# RGB bypass arrow (Innovation 3 - 6ch fusion)
ax2.annotate('', xy=(0.82, 0.35), xytext=(0.13, 0.35),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
ax2.text(0.45, 0.38, 'RGB preserved (6-channel fusion)', fontsize=9, ha='center', style='italic')

# Concat symbol
ax2.text(0.825, 0.47, '+', fontsize=16, ha='center', va='center', fontweight='bold')

# Arrow to YOLO
ax2.annotate('', xy=(0.88, 0.42), xytext=(0.84, 0.42),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# YOLO 6ch box
rect_yolo = Rectangle((0.88, 0.32), 0.1, 0.2, fill=True, facecolor='white', edgecolor='black', linewidth=1)
ax2.add_patch(rect_yolo)
ax2.text(0.93, 0.42, 'YOLO\n(6ch)', fontsize=9, ha='center', va='center')

# Key results text box at bottom
result_box = Rectangle((0.02, 0.02), 0.96, 0.15, fill=True, facecolor='white', edgecolor='black', linewidth=1)
ax2.add_patch(result_box)

ax2.text(0.5, 0.135, 'Key Improvements vs. RGB Baseline:', fontsize=10, ha='center', fontweight='bold')
ax2.text(0.18, 0.065, 'RAFT+Attn+dt:', fontsize=9, ha='center')
ax2.text(0.18, 0.03, 'Best ', fontsize=9, ha='right')
ax2.text(0.18, 0.03, 'Recall', fontsize=9, ha='left', color='red', style='italic')
ax2.text(0.26, 0.03, ' & ', fontsize=9, ha='left')
ax2.text(0.28, 0.03, 'mAP50', fontsize=9, ha='left', color='red', style='italic')

ax2.text(0.55, 0.065, '6ch Fusion:', fontsize=9, ha='center')
ax2.text(0.55, 0.03, 'Highest ', fontsize=9, ha='right')
ax2.text(0.55, 0.03, 'Precision', fontsize=9, ha='left', color='red', style='italic')
ax2.text(0.64, 0.03, ', preserves appearance', fontsize=9, ha='left')

plt.tight_layout()
plt.savefig('motivation_figure_v2.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('motivation_figure_v2.pdf', bbox_inches='tight', facecolor='white')
print("Saved: motivation_figure_v2.png and motivation_figure_v2.pdf")
plt.show()
