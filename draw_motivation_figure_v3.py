"""
Motivation figure - clean style with detailed explanations
(a) Traditional approach problem
(b) Our proposed solution
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# Set up figure
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.patch.set_facecolor('white')

# ============ (a) Traditional YOLO Detection ============
ax1 = axes[0]
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.text(0.5, 0.95, '(a) Traditional YOLO Detection', fontsize=13, ha='center', fontweight='bold')

# Image placeholder (frame t-1)
rect1 = Rectangle((0.03, 0.4), 0.12, 0.4, fill=True, facecolor='#EEEEEE', edgecolor='black', linewidth=1)
ax1.add_patch(rect1)
ax1.text(0.09, 0.6, '[Image\n$I_{t-1}$]', fontsize=9, ha='center', va='center')
ax1.text(0.09, 0.33, '$I_{t-1}$', fontsize=10, ha='center')

# Image placeholder (frame t)
rect2 = Rectangle((0.18, 0.4), 0.12, 0.4, fill=True, facecolor='#EEEEEE', edgecolor='black', linewidth=1)
ax2_img = ax1.add_patch(rect2)
ax1.text(0.24, 0.6, '[Image\n$I_t$]', fontsize=9, ha='center', va='center')
ax1.text(0.24, 0.33, '$I_t$', fontsize=10, ha='center')

# Arrow to YOLO (only from I_t)
ax1.annotate('', xy=(0.38, 0.6), xytext=(0.31, 0.6),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# YOLO box
rect3 = Rectangle((0.38, 0.48), 0.12, 0.24, fill=True, facecolor='white', edgecolor='black', linewidth=1)
ax1.add_patch(rect3)
ax1.text(0.44, 0.6, 'YOLO\n(RGB)', fontsize=10, ha='center', va='center')

# Arrow to output
ax1.annotate('', xy=(0.58, 0.6), xytext=(0.51, 0.6),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Detection result placeholder
rect4 = Rectangle((0.58, 0.4), 0.15, 0.4, fill=True, facecolor='#EEEEEE', edgecolor='black', linewidth=1)
ax1.add_patch(rect4)
ax1.text(0.655, 0.6, '[Detection\nResult]', fontsize=9, ha='center', va='center')

# Problem text box on the right - more detailed
problem_box = Rectangle((0.76, 0.25), 0.22, 0.55, fill=True, facecolor='white', edgecolor='black', linewidth=1)
ax1.add_patch(problem_box)
ax1.text(0.87, 0.73, 'Problems:', fontsize=10, ha='center', fontweight='bold')
ax1.text(0.78, 0.62, '1. Only uses ', fontsize=9, ha='left')
ax1.text(0.895, 0.62, 'single frame', fontsize=9, ha='left', color='red', style='italic')
ax1.text(0.78, 0.52, '2. Ignores temporal', fontsize=9, ha='left')
ax1.text(0.78, 0.44, '   ', fontsize=9, ha='left')
ax1.text(0.80, 0.44, 'motion information', fontsize=9, ha='left', color='red', style='italic')
ax1.text(0.78, 0.34, '3. May miss moving', fontsize=9, ha='left')
ax1.text(0.78, 0.27, '   objects or have', fontsize=9, ha='left')
# ax1.text(0.78, 0.20, '   lower recall', fontsize=9, ha='left')

# Note: I_{t-1} is not used
ax1.plot([0.09, 0.09], [0.4, 0.35], 'k--', lw=0.8)
ax1.text(0.09, 0.18, '(not used)', fontsize=8, ha='center', color='gray', style='italic')

# Dashed line separator
ax1.axhline(y=0.08, color='gray', linestyle='--', linewidth=1)

# ============ (b) Our Motion-Augmented YOLO ============
ax2 = axes[1]
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.text(0.5, 0.96, '(b) Our Motion-Augmented YOLO', fontsize=13, ha='center', fontweight='bold')

# Frame t-1 placeholder
rect_t1 = Rectangle((0.02, 0.58), 0.08, 0.28, fill=True, facecolor='#EEEEEE', edgecolor='black', linewidth=1)
ax2.add_patch(rect_t1)
ax2.text(0.06, 0.72, '[Image\n$I_{t-1}$]', fontsize=8, ha='center', va='center')

# Frame t placeholder  
rect_t = Rectangle((0.02, 0.25), 0.08, 0.28, fill=True, facecolor='#EEEEEE', edgecolor='black', linewidth=1)
ax2.add_patch(rect_t)
ax2.text(0.06, 0.39, '[Image\n$I_t$]', fontsize=8, ha='center', va='center')

# Arrow from frames to RAFT
ax2.annotate('', xy=(0.145, 0.55), xytext=(0.11, 0.68),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
ax2.annotate('', xy=(0.145, 0.55), xytext=(0.11, 0.42),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# RAFT Optical Flow box
rect_raft = Rectangle((0.145, 0.45), 0.09, 0.2, fill=True, facecolor='white', edgecolor='black', linewidth=1)
ax2.add_patch(rect_raft)
ax2.text(0.19, 0.55, 'RAFT\nFlow', fontsize=9, ha='center', va='center')

# Arrow to dt-norm
ax2.annotate('', xy=(0.28, 0.55), xytext=(0.24, 0.55),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# dt-normalization box (Innovation 1)
rect_dt = Rectangle((0.28, 0.45), 0.1, 0.2, fill=True, facecolor='white', edgecolor='black', linewidth=1.5)
ax2.add_patch(rect_dt)
ax2.text(0.33, 0.58, 'dt-Norm', fontsize=9, ha='center', va='center', fontweight='bold')
ax2.text(0.33, 0.49, '$\\tilde{F}=F/\\Delta t$', fontsize=8, ha='center', va='center')

# Innovation 1 label
ax2.text(0.33, 0.68, 'Innovation 1', fontsize=8, ha='center', color='#1565C0', fontweight='bold')

# Arrow to attention
ax2.annotate('', xy=(0.43, 0.55), xytext=(0.39, 0.55),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# Attention box (Innovation 2)
rect_attn = Rectangle((0.43, 0.45), 0.1, 0.2, fill=True, facecolor='white', edgecolor='black', linewidth=1.5)
ax2.add_patch(rect_attn)
ax2.text(0.48, 0.58, 'Attention', fontsize=9, ha='center', va='center', fontweight='bold')
ax2.text(0.48, 0.49, 'Gate', fontsize=9, ha='center', va='center')

# Innovation 2 label
ax2.text(0.48, 0.68, 'Innovation 2', fontsize=8, ha='center', color='#1565C0', fontweight='bold')

# Motion image output
ax2.annotate('', xy=(0.58, 0.55), xytext=(0.54, 0.55),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# Motion image placeholder
rect_motion = Rectangle((0.58, 0.45), 0.09, 0.2, fill=True, facecolor='#EEEEEE', edgecolor='black', linewidth=1)
ax2.add_patch(rect_motion)
ax2.text(0.625, 0.55, '[Motion\nu,v,attn]', fontsize=8, ha='center', va='center')

# RGB bypass arrow (Innovation 3 - 6ch fusion)
ax2.annotate('', xy=(0.69, 0.39), xytext=(0.11, 0.39),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
ax2.text(0.40, 0.33, 'RGB channel preserved', fontsize=9, ha='center', style='italic')

# Innovation 3 label
ax2.text(0.69, 0.78, 'Innovation 3', fontsize=8, ha='center', color='#1565C0', fontweight='bold')
ax2.text(0.69, 0.73, '6-Channel Fusion', fontsize=9, ha='center', fontweight='bold')

# Concat box
rect_concat = Rectangle((0.68, 0.42), 0.04, 0.26, fill=True, facecolor='white', edgecolor='black', linewidth=1.5)
ax2.add_patch(rect_concat)
ax2.text(0.70, 0.55, '+', fontsize=14, ha='center', va='center', fontweight='bold')

# Arrow to YOLO
ax2.annotate('', xy=(0.77, 0.55), xytext=(0.73, 0.55),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# YOLO 6ch box
rect_yolo = Rectangle((0.77, 0.45), 0.09, 0.2, fill=True, facecolor='white', edgecolor='black', linewidth=1)
ax2.add_patch(rect_yolo)
ax2.text(0.815, 0.55, 'YOLO\n(6ch)', fontsize=9, ha='center', va='center')

# Arrow to result
ax2.annotate('', xy=(0.91, 0.55), xytext=(0.87, 0.55),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# Detection result placeholder
rect_result = Rectangle((0.91, 0.45), 0.07, 0.2, fill=True, facecolor='#EEEEEE', edgecolor='black', linewidth=1)
ax2.add_patch(rect_result)
ax2.text(0.945, 0.55, '[Result]', fontsize=8, ha='center', va='center')

# Detailed explanation boxes at bottom
# Innovation 1 explanation
box1 = Rectangle((0.02, 0.02), 0.30, 0.18, fill=True, facecolor='white', edgecolor='black', linewidth=1)
ax2.add_patch(box1)
ax2.text(0.17, 0.17, 'dt-Normalization', fontsize=9, ha='center', fontweight='bold')
ax2.text(0.04, 0.11, 'Compensates for variable frame', fontsize=8, ha='left')
ax2.text(0.04, 0.06, 'intervals. Normalizes flow by $\\Delta t$', fontsize=8, ha='left')
ax2.text(0.04, 0.01, 'to ensure ', fontsize=8, ha='left')
ax2.text(0.115, 0.01, 'consistent magnitude', fontsize=8, ha='left', color='red', style='italic')

# Innovation 2 explanation
box2 = Rectangle((0.35, 0.02), 0.30, 0.18, fill=True, facecolor='white', edgecolor='black', linewidth=1)
ax2.add_patch(box2)
ax2.text(0.50, 0.17, 'Attention Gate', fontsize=9, ha='center', fontweight='bold')
ax2.text(0.37, 0.11, 'Filters noisy motion via magnitude', fontsize=8, ha='left')
ax2.text(0.37, 0.06, 'or coherence gating. Suppresses', fontsize=8, ha='left')
ax2.text(0.37, 0.01, 'unreliable flow in ', fontsize=8, ha='left')
ax2.text(0.52, 0.01, 'background regions', fontsize=8, ha='left', color='red', style='italic')

# Innovation 3 explanation
box3 = Rectangle((0.68, 0.02), 0.30, 0.18, fill=True, facecolor='white', edgecolor='black', linewidth=1)
ax2.add_patch(box3)
ax2.text(0.83, 0.17, '6-Channel Fusion', fontsize=9, ha='center', fontweight='bold')
ax2.text(0.70, 0.11, 'Concatenates RGB + motion (u,v,attn).', fontsize=8, ha='left')
ax2.text(0.70, 0.06, 'Preserves ', fontsize=8, ha='left')
ax2.text(0.76, 0.06, 'appearance cues', fontsize=8, ha='left', color='red', style='italic')
ax2.text(0.88, 0.06, ' while', fontsize=8, ha='left')
ax2.text(0.70, 0.01, 'adding motion. Best Precision.', fontsize=8, ha='left')

plt.tight_layout()
plt.savefig('motivation_figure_v3.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('motivation_figure_v3.pdf', bbox_inches='tight', facecolor='white')
print("Saved: motivation_figure_v3.png and motivation_figure_v3.pdf")
plt.show()
