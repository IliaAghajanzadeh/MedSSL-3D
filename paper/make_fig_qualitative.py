"""F6 — qualitative test-set comparison. Same encoder, same seed, same patient;
only the input preprocessing differs. Built by re-composing the two overlay PNGs
the pipeline already produced — no model is run.

INPUTS YOU MUST SUPPLY (they are not in the project; re-upload them):
  SRC['norm'] — the ssl_norm overlay render  (grey background: z-scoring
                re-centres tissue around zero, so the exact-zero background
                renders mid-grey)
  SRC['raw']  — the ssl_raw overlay render   (black background)
Each is a 3-panel strip: FLAIR slice | ground truth | prediction.
If the panel geometry of the new renders differs, re-measure XS / YS.
"""
from PIL import Image
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SRC = {'norm': "overlay_ssl_norm.png", 'raw': "overlay_ssl_raw.png"}
XS = [(34, 582), (626, 1174), (1219, 1767)]     # the three panels, measured
YS = (95, 515)                                   # cropped to the brain, margin kept
INK, MUTED = '#0b0b0b', '#52514e'

# per-patient 3D Dice for BraTS20_Training_186, seed 42 (v0_11_test_per_patient.csv)
DICE = {'norm': 0.7056, 'raw': 0.5995}
ROWLAB = {'norm': 'Normalized input', 'raw': 'Raw input'}
COLTITLE = ['FLAIR slice (model input)', 'Ground truth', 'Prediction']

panels = {k: [np.asarray(Image.open(p).convert('RGB'))[YS[0]:YS[1], x0:x1]
              for x0, x1 in XS] for k, p in SRC.items()}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 300,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0))
for r, key in enumerate(['norm', 'raw']):
    for c in range(3):
        ax = axes[r, c]
        ax.imshow(panels[key][c]); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor('#c9c8c3'); s.set_linewidth(0.8)
        if r == 0:
            ax.set_title(COLTITLE[c], fontsize=9, color=INK, pad=6)
    axes[r, 0].set_ylabel(f"{ROWLAB[key]}\nDice {DICE[key]:.3f}",
                          fontsize=9.5, color=INK, labelpad=8)

fig.suptitle('Same encoder, same seed, same test patient — only the input preprocessing differs',
             fontsize=10.5, x=0.012, ha='left', y=1.005, color=INK)
fig.legend(handles=[Patch(facecolor='#f2dd5e', label='Oedema (ED)'),
                    Patch(facecolor='#a04d4d', label='Necrotic core (NCR)'),
                    Patch(facecolor='#69cbe8', label='Enhancing tumour (ET)')],
           loc='lower center', bbox_to_anchor=(0.55, -0.035), ncol=3,
           frameon=False, fontsize=7.8)
fig.text(0.012, -0.075,
         'SSL-pretrained arm, seed 42, test patient BraTS20_Training_186 — never seen during training or calibration.\n'
         'The left column also shows why the two rows differ in appearance: z-scoring re-centres tissue around zero, so\n'
         'the exact-zero background renders mid-grey rather than black. Dice values are the 3D scores for this patient.',
         fontsize=7.4, color=MUTED, ha='left', va='top')
fig.subplots_adjust(hspace=0.02, wspace=0.04)
for ext in ('png', 'pdf'):
    fig.savefig(f"F6_qualitative_test_patient.{ext}")
print("wrote F6_qualitative_test_patient")
