"""Draw the validation-strategy diagram (docs/source/validation.png).

Each row is one training iteration and each square one patient; orange
squares are held out for validation. Purely schematic, no data involved.

Run from the repository root:  python docs/figures/make_validation.py
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
OUT = ROOT / 'docs/source/validation.png'
INK, MUTED = '#111111', '#555555'
N = 12
FONT = 'Inter' if any('Inter' in f.name for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family': FONT})

rng = np.random.default_rng(4)
perm = rng.permutation(N)
fixed = [set(range(9, 12))]
boot = [set(rng.choice(N, 3, replace=False)) for _ in range(3)]
kfold = [set(perm[i::3]) for i in range(3)]
site_of = np.repeat([0, 1, 2], 4)
preserved = [set(np.where(site_of == s)[0]) for s in range(3)]

panels = [
    ('Fixed', 'One held-out fraction', fixed, False, False),
    ('Bootstrap', 'Random hold-outs, repeated', boot, False, True),
    ('k-fold', 'Every patient validated once', kfold, False, False),
    ('k-fold, preserved site', 'Each site held out together', preserved, True, False),
]

fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), dpi=150)
for ax, (title, sub, rows, show_sites, more) in zip(axes, panels):
    ax.set_xlim(-2.2, N + 0.4); ax.set_ylim(-6.3, 2.1); ax.axis('off')
    ax.text(-2.2, 1.55, title, fontsize=13, weight='bold', color=INK, va='center')
    ax.text(-2.2, 0.75, sub, fontsize=10, color=MUTED, va='center')
    if show_sites:
        for s, lab in enumerate('ABC'):
            ax.plot([s * 4 + 0.15, s * 4 + 0.15, s * 4 + 3.85, s * 4 + 3.85], [-0.15, -0.3, -0.3, -0.15], color=INK, lw=1.2)
            ax.text(s * 4 + 2, -0.05, f'site {lab}', fontsize=9.5, color=INK, ha='center', va='bottom')
    for r, val in enumerate(rows):
        y = -1.35 - r * 1.25
        ax.text(-0.5, y + 0.45, str(r + 1) if len(rows) > 1 else '', fontsize=9.5, color=MUTED, ha='right', va='center')
        for p in range(N):
            ax.add_patch(Rectangle((p + 0.12, y), 0.76, 0.9, fc=INK if p in val else 'white', ec=INK, lw=1.1))
    if more:
        ax.text(N / 2, -1.35 - len(rows) * 1.25 + 0.35, '...', fontsize=14, color=MUTED, ha='center')

fig.text(0.012, 0.045, '□ training    ■ validation', fontsize=11, color=INK)
fig.text(0.16, 0.045, 'Each row is one training iteration; each square is one patient.', fontsize=10.5, color=MUTED)
fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.12, wspace=0.12)
fig.savefig(OUT, dpi=150, facecolor='white')
print('wrote', OUT)
