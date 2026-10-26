"""v0.11 paper figures — built from the frozen CSVs only. No model, no GPU."""
import csv, os, collections, statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

U = os.environ.get("MEDSSL_CSV_DIR", "./csv") + "/"
OUT = "./"
os.makedirs(OUT, exist_ok=True)

rep = list(csv.DictReader(open(U + "v0_11_test_report.csv")))
pp = list(csv.DictReader(open(U + "v0_11_test_per_patient.csv")))
vs = list(csv.DictReader(open(U + "v0_11_val_scores.csv")))
cr = list(csv.DictReader(open(U + "v0_11_classification_report.csv")))

ARMS = ['ssl_norm', 'scratch_norm', 'ssl_raw', 'scratch_raw']   # normalized pair first
LABEL = {'ssl_norm': 'SSL\nnormalized', 'scratch_norm': 'Scratch\nnormalized',
         'ssl_raw': 'SSL\nraw', 'scratch_raw': 'Scratch\nraw'}
BLUE, ORANGE = '#2a78d6', '#eb6834'          # validated pair (CVD ΔE 24.7)
COLOR = {'ssl_norm': BLUE, 'scratch_norm': BLUE, 'ssl_raw': ORANGE, 'scratch_raw': ORANGE}
HATCH = {'ssl_norm': '', 'scratch_norm': '///', 'ssl_raw': '', 'scratch_raw': '///'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d8d7d2'
BOOT_SEED, BOOT_N = 20260904, 10000

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.8,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

def style(ax, ylab=None, ymax=None):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    if ylab: ax.set_ylabel(ylab)
    if ymax: ax.set_ylim(0, ymax)

def boot(vals, iters=BOOT_N, seed=BOOT_SEED):
    a = np.asarray([v for v in vals if v == v], float)
    rs = np.random.RandomState(seed)
    m = a[rs.randint(0, a.size, (iters, a.size))].mean(1)
    return a.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

def seeds_of(arm, key):
    return [float(r[key]) for r in rep if r['arm'] == arm]

def per_patient(arm, key):
    """seed-averaged value per patient (one number per patient per arm)"""
    d = collections.defaultdict(list)
    for r in pp:
        if r['arm'] == arm:
            v = float(r[key]) if r[key] not in ('', 'nan') else float('nan')
            if v == v: d[r['patient_id']].append(v)
    return [float(np.mean(v)) for v in d.values()]

def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(f"{OUT}{name}.{ext}")
    plt.close(fig); print("wrote", name)

# ── F2 — headline factorial, BOTH uncertainties shown separately ──────────────
fig, ax = plt.subplots(figsize=(6.4, 4.0))
x = np.arange(4)
for i, arm in enumerate(ARMS):
    s = seeds_of(arm, 'dice'); m = float(np.mean(s)); sd = st.stdev(s)
    bm, lo, hi = boot(per_patient(arm, 'dice'))
    ax.bar(i - 0.13, m, 0.42, color=COLOR[arm], hatch=HATCH[arm],
           edgecolor='white', linewidth=1.2, zorder=2)
    ax.errorbar(i - 0.13, m, yerr=sd, fmt='none', ecolor=INK, capsize=5,
                elinewidth=1.4, capthick=1.4, zorder=4)
    ax.plot([i - 0.13] * 3, s, 'o', ms=3.5, mfc='white', mec=INK, mew=0.9, zorder=5)
    ax.plot([i + 0.24, i + 0.24], [lo, hi], color=MUTED, lw=1.6, zorder=3,
            solid_capstyle='butt')
    ax.plot([i + 0.24], [bm], 'D', ms=4, color=MUTED, zorder=4)
    ax.text(i - 0.13, m + sd + 0.028, f"{m:.3f}", ha='center', fontsize=8.5, color=INK)
style(ax, 'Test Dice (4-class)', 0.72)
ax.set_xticks(x); ax.set_xticklabels([LABEL[a] for a in ARMS])
ax.axhline(0, color=MUTED, lw=0.8)
ax.legend(handles=[
    Patch(facecolor=BLUE, edgecolor='white', label='Normalized input'),
    Patch(facecolor=ORANGE, edgecolor='white', label='Raw input'),
    Patch(facecolor='#b9b8b3', hatch='///', edgecolor='white', label='Scratch (vs. solid = SSL)'),
    Line2D([], [], color=INK, marker='_', ls='none', ms=12, mew=1.4,
           label='seed Mean ± SD (n=3), with the 3 seeds'),
    Line2D([], [], color=MUTED, marker='D', ls='-', ms=4, lw=1.6,
           label='patient bootstrap 95% CI (n=55)'),
], loc='upper right', frameon=False, fontsize=7.6, handlelength=1.6)
ax.set_title('Intensity normalization moves test Dice by ~0.22;\nSSL pretraining does not move it at all',
             fontsize=10.5, loc='left', pad=10)
save(fig, 'F2_factorial_test_dice')

# ── F3 — per-patient distributions ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 3.8))
data = [per_patient(a, 'dice') for a in ARMS]
bp = ax.boxplot(data, positions=x, widths=0.5, patch_artist=True, showfliers=False,
                medianprops=dict(color=INK, lw=1.4),
                whiskerprops=dict(color=MUTED, lw=1), capprops=dict(color=MUTED, lw=1),
                boxprops=dict(edgecolor='white', lw=1.2))
for patch, arm in zip(bp['boxes'], ARMS):
    patch.set_facecolor(COLOR[arm]); patch.set_alpha(0.35); patch.set_hatch(HATCH[arm])
rs = np.random.RandomState(7)
for i, (arm, d) in enumerate(zip(ARMS, data)):
    ax.plot(i + rs.uniform(-0.16, 0.16, len(d)), d, 'o', ms=2.6,
            color=COLOR[arm], alpha=0.75, mew=0, zorder=3)
style(ax, 'Test Dice per patient', 1.0)
ax.set_xticks(x); ax.set_xticklabels([LABEL[a] for a in ARMS])
ax.set_title('Per-patient test Dice (55 patients, averaged over the 3 seeds)',
             fontsize=10.5, loc='left', pad=8)
save(fig, 'F3_per_patient_dice')

# ── F4 — why 0.5 was the wrong cut (validation scores) ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
for ax, arm in zip(axes, ['ssl_norm', 'scratch_norm']):
    thr = float([r['threshold'] for r in cr
                 if r['arm'] == arm and r['seed'] == '42' and r['threshold_kind'] == 'youden_val'][0])
    rows = [r for r in vs if r['arm'] == arm and r['seed'] == '42']
    rs2 = np.random.RandomState(3)
    for lab, cls, col, mk in (('HGG', '1', BLUE, 'o'), ('LGG', '0', ORANGE, '^')):
        s = [float(r['p_hgg']) for r in rows if r['y_true'] == cls]
        ax.plot(s, rs2.uniform(-0.35, 0.35, len(s)) + (0.6 if cls == '1' else -0.6),
                mk, ms=4.5, color=col, alpha=0.85, mew=0, label=f"{lab} (n={len(s)})")
    rescued = sum(1 for r in rows if r['y_true'] == '0' and 0.5 <= float(r['p_hgg']) < thr)
    ax.axvspan(0.5, thr, color='#f2c94c', alpha=0.22, zorder=0)
    ax.axvline(0.5, color=MUTED, ls='--', lw=1.2)
    ax.axvline(thr, color=INK, ls='-', lw=1.4)
    ax.text(0.5, 1.30, ' default 0.5', color=MUTED, fontsize=7.5, ha='left')
    ax.text(thr, -1.52, f"Youden {thr:.3f} ", color=INK, fontsize=7.5, ha='right')
    ax.annotate(f"{rescued} LGG patients\nlive in this band —\nmisread at 0.5,\ncaught after\nrecalibration",
                xy=((0.5 + thr) / 2, -0.15), xycoords='data', ha='center', va='center',
                fontsize=7.2, color=INK)
    ax.set_yticks([0.6, -0.6]); ax.set_yticklabels(['HGG', 'LGG'])
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-1.7, 1.5); ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel('P(HGG) on validation')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(LABEL[arm].replace('\n', ' ') + ', seed 42', fontsize=9.5, loc='left')
axes[0].legend(frameon=False, fontsize=8, loc='center left')
fig.suptitle('The model ranks patients well but is badly calibrated: almost every score sits near 1.0,\n'
             'so a 0.5 cut labels most LGG patients HGG',
             fontsize=10.5, x=0.02, ha='left', y=1.06)
save(fig, 'F4_threshold_calibration')

# ── F5 — region breakdown ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 3.8))
regions = [('wt', 'WT'), ('tc', 'TC'), ('et', 'ET (n=53)')]
w = 0.2
for i, arm in enumerate(ARMS):
    means, sds = [], []
    for key, _ in regions:
        s = seeds_of(arm, key); means.append(np.mean(s)); sds.append(st.stdev(s))
    pos = np.arange(3) + (i - 1.5) * w
    ax.bar(pos, means, w * 0.88, color=COLOR[arm], hatch=HATCH[arm],
           edgecolor='white', linewidth=1.0, zorder=2, label=LABEL[arm].replace('\n', ' '))
    ax.errorbar(pos, means, yerr=sds, fmt='none', ecolor=INK, capsize=3,
                elinewidth=1.0, capthick=1.0, zorder=4)
style(ax, 'Test Dice', 1.0)
ax.set_xticks(np.arange(3)); ax.set_xticklabels([r[1] for r in regions])
ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper right')
ax.set_title('Region-wise test Dice — the normalization gain grows as the structure shrinks\n(WT +19%, TC +45%, ET +67% relative)',
             fontsize=10.5, loc='left', pad=8)
save(fig, 'F5_region_dice')

# ── numbers for the paper tables ─────────────────────────────────────────────
print("\n=== T5 (test, ddof=1 + bootstrap) ===")
for key, name in [('dice', 'Dice'), ('wt', 'WT'), ('tc', 'TC'), ('et', 'ET')]:
    print(f"-- {name}")
    for arm in ARMS:
        s = seeds_of(arm, key)
        bm, lo, hi = boot(per_patient(arm, key))
        print(f"   {arm:<14} {np.mean(s):.4f} ± {st.stdev(s):.4f}   CI [{lo:.4f}, {hi:.4f}]  (n={len(per_patient(arm,key))})")
