"""F1 — study schematic. Layer 1 (development history + four failure modes)
and Layer 2 (the controlled factorial, frozen calibration, one-shot test).
Pure drawing: no data is read, nothing is computed — this one always rebuilds."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = "."
INK, MUTED = '#0b0b0b', '#52514e'
BLUE, ORANGE, RED = '#2a78d6', '#eb6834', '#c0392b'
PAPER, BAND = '#fbfaf8', '#eceae4'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.0,
                     'text.color': INK, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})

fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 108); ax.axis('off')


def box(x, y, w, h, label, sub=None, fc=PAPER, ec=MUTED, lw=0.9,
        fs=8.0, bold=False, subfs=6.2, ytitle=0.68, ysub=0.30):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0,rounding_size=1.4',
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                zorder=5))
    kw = dict(ha='center', va='center', zorder=6)
    if sub:
        ax.text(x + w / 2, y + h * ytitle, label, fontsize=fs,
                fontweight='bold' if bold else 'normal', linespacing=1.25, **kw)
        ax.text(x + w / 2, y + h * ysub, sub, fontsize=subfs, color=MUTED,
                linespacing=1.5, **kw)
    else:
        ax.text(x + w / 2, y + h / 2, label, fontsize=fs,
                fontweight='bold' if bold else 'normal', linespacing=1.25, **kw)


def arrow(x1, y1, x2, y2, col=MUTED, lw=0.9, head=True):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle='-|>' if head else '-',
                                 mutation_scale=8, color=col, linewidth=lw,
                                 shrinkA=0, shrinkB=0, zorder=3))


# ══ LAYER 1 ═════════════════════════════════════════════════════════════════
ax.add_patch(Rectangle((0, 74), 100, 34, facecolor=BAND, edgecolor='none'))
ax.text(1.5, 105, 'LAYER 1', fontsize=9.0, fontweight='bold', va='center')
ax.text(13.5, 105, 'eleven development versions — what the measurements got wrong',
        fontsize=8.0, color=MUTED, va='center')

ax.plot([6, 96], [84, 84], color=MUTED, lw=1.1, zorder=1)
VERS = [(10, 'v0.1–v0.4', '—'), (22, 'v0.5', '0.200'), (33, 'v0.6', '0.333'),
        (44, 'v0.7', '0.434'), (56, 'v0.8', '0.454'), (67, 'v0.9', '0.407'),
        (79, 'v0.10', '0.568'), (90, 'v0.10$_{50}$', '0.596')]
for x, v, d in VERS:
    ax.plot([x], [84], 'o', ms=4.0, color=INK, zorder=3)
    ax.text(x, 81.2, v, ha='center', va='center', fontsize=7.2)
    ax.text(x, 78.2, d, ha='center', va='center', fontsize=6.8, color=MUTED)
ax.text(2.2, 78.2, 'Dice', ha='left', va='center', fontsize=6.6,
        color=MUTED, style='italic')

# rows ascend left-to-right so no leader line ever crosses a label
FLAGS = [(56, '1 · RNG coupling  —  found in 3 h', 90.2),
         (67, '2 · metric artifact  —  found in 7 min', 93.8),
         (79, '3 · preprocessing mismatch  —  found in $<$1 min', 97.4),
         (90, '4 · cache provenance  —  found in zero', 101.0)]
for x, lab, yr in FLAGS:
    ax.plot([x, x], [85.0, yr], color=RED, lw=0.8, ls=(0, (2, 1.8)), zorder=2)
    ax.plot([x], [84], 'o', ms=6.4, mfc='none', mec=RED, mew=1.2, zorder=4)
    ax.text(x - 1.6, yr, lab, ha='right', va='center', fontsize=7.1, color=RED)

arrow(79, 76.4, 79, 67.5, col=RED, lw=1.0)
ax.text(77.4, 71.5, 'raises the hypothesis tested below', ha='right',
        va='center', fontsize=7.0, color=RED)

# ══ LAYER 2 ═════════════════════════════════════════════════════════════════
ax.text(1.5, 65, 'LAYER 2', fontsize=9.0, fontweight='bold', va='center')
ax.text(13.5, 65, 'a pre-specified $2\\times2\\times3$ factorial, then one look at the test set',
        fontsize=8.0, color=MUTED, va='center')

# data + split
box(0, 36, 14.5, 10, 'BraTS2020', '368 usable cases')
arrow(14.5, 41, 16.0, 41, head=False)
arrow(16.0, 51.5, 16.0, 30.5, head=False)
box(17.5, 48, 13.5, 7, 'train  258')
box(17.5, 37.5, 13.5, 7, 'val  55', 'HGG 42 / LGG 13')
box(17.5, 27, 13.5, 7, 'test  55', 'sealed until §VII', fc='#fdf1ec', ec=ORANGE)
for yy in (51.5, 41, 30.5):
    arrow(16.0, yy, 17.1, yy)

# pretraining
box(34.5, 48, 24, 10, 'SimCLR pretraining',
    '258 training patients only\n100 epochs · 5.0 h', ec=BLUE, ytitle=0.72)
arrow(31, 51.5, 34.1, 52.5, col=BLUE)

# the twelve runs
box(34.5, 19, 24, 25, '')
ax.text(46.5, 41.3, '12 fine-tuning runs', ha='center', va='center',
        fontsize=8.2, fontweight='bold', zorder=6)
ax.text(46.5, 38.3, 'identical hyperparameters', ha='center', va='center',
        fontsize=6.2, color=MUTED, zorder=6)
CELL = [('SSL\nnorm', BLUE, ''), ('scratch\nnorm', BLUE, '///'),
        ('SSL\nraw', ORANGE, ''), ('scratch\nraw', ORANGE, '///')]
for i, (lab, col, h) in enumerate(CELL):
    cx = 35.7 + (i % 2) * 11.4
    cy = 29.4 - (i // 2) * 7.2
    ax.add_patch(Rectangle((cx, cy), 10.6, 6.4, facecolor=col, alpha=0.15,
                           edgecolor=col, linewidth=0.8, hatch=h, zorder=6))
    ax.text(cx + 5.3, cy + 3.2, lab, ha='center', va='center',
            fontsize=6.8, linespacing=1.3, zorder=7)
arrow(46.5, 48, 46.5, 44.5, col=BLUE)
arrow(31, 41, 34.1, 38.0)
ax.text(46.5, 15.6, '$\\times$ seeds 42 · 123 · 2024   $\\rightarrow$   42.7 GPU-hours',
        ha='center', va='center', fontsize=7.0, color=MUTED)

# rehearsal + calibration
box(60.5, 44, 23, 10, 'dress rehearsal',
    'replay on validation\n12/12 reproduce, $\\Delta = 0$', ytitle=0.72)
box(60.5, 29, 23, 10, 'calibrate on validation',
    "Youden's $J$ per (arm, seed)\nthresholds frozen + hashed",
    fs=7.3, subfs=5.9, ytitle=0.72)
arrow(58.5, 40, 60.1, 47.5)
arrow(58.5, 33, 60.1, 34)

# one-shot test
box(85, 27, 15, 21, 'ONE-SHOT\nTEST',
    '55 held-out patients\nprovenance record\nblocks a second run',
    fs=8.4, bold=True, subfs=5.9, fc='#fdf1ec', ec=ORANGE, lw=1.3, ytitle=0.74)
arrow(83.5, 47.5, 84.6, 43)
arrow(83.5, 34, 84.6, 36)

# the sealed path
arrow(21, 27, 21, 8, col=ORANGE, lw=1.0, head=False)
arrow(21, 8, 92.7, 8, col=ORANGE, lw=1.0, head=False)
arrow(92.7, 8, 92.7, 26.6, col=ORANGE, lw=1.0)
ax.text(57, 5.0, 'the test split is loaded exactly once, after every decision is frozen',
        ha='center', va='center', fontsize=7.0, color=ORANGE)

for ext in ('png', 'pdf'):
    fig.savefig(f"{OUT}/F1_study_schematic.{ext}")
print("wrote F1_study_schematic")
