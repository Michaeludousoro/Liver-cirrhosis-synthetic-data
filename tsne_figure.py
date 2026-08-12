"""Standalone t-SNE projection figure.

Extracted from notebook 02 because that notebook also regenerates synthetic
data with an older consensus implementation, which corrupts output/data. This
script only reads what master_runner.py has written, so it is safe to run.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os, sys
sys.path.insert(0, os.getcwd())
from src.data_loader import ALL_FEATURE_COLS

_S = 2.0

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11*_S,
    'axes.titlesize':    12*_S,
    'axes.labelsize':    11*_S,
    'xtick.labelsize':   10*_S,
    'ytick.labelsize':   10*_S,
    'legend.fontsize':   10*_S,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.spines.left':  False,
    'axes.spines.bottom': False,
    'figure.dpi':        130,
    'savefig.dpi':       300,
})

real_df   = pd.read_csv('output/data/train_real.csv')
gan_df    = pd.read_csv('output/data/filtered_gan.csv')
ctgan_df  = pd.read_csv('output/data/filtered_ctgan.csv')
tvae_df   = pd.read_csv('output/data/filtered_tvae.csv')
cons_df   = pd.read_csv('output/data/consensus_equalised.csv')

FEAT_COLS = [c for c in ALL_FEATURE_COLS if c in real_df.columns]

comparisons = [
    ('GAN (filtered)',   gan_df,   '#0066FF', '#003399'),
    ('cGAN (filtered)', ctgan_df, '#FF4500', '#8B2000'),
    ('VAE (filtered)',  tvae_df,  '#00B43C', '#004D1A'),
    ('Consensus',        cons_df,  '#9900CC', '#4B0066'),
]

REAL_COLOR      = '#404040'
REAL_EDGE_COLOR = '#FFFFFF'

def add_kde_contours(ax, coords, color, levels=5, alpha=0.35):
    """Overlay smooth KDE density contours on a scatter panel."""
    if len(coords) < 10:
        return
    try:
        kde = gaussian_kde(coords.T, bw_method=0.35)
        x_min, x_max = coords[:, 0].min() - 2, coords[:, 0].max() + 2
        y_min, y_max = coords[:, 1].min() - 2, coords[:, 1].max() + 2
        xi, yi = np.mgrid[x_min:x_max:80j, y_min:y_max:80j]
        zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        ax.contour(xi, yi, zi, levels=levels, colors=color,
                   alpha=alpha, linewidths=1.2)
    except Exception:
        pass

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

for ax, (method_name, synth_df, synth_color, synth_dark) in zip(axes, comparisons):

    real_part  = real_df[FEAT_COLS].fillna(real_df[FEAT_COLS].mean())
    synth_part = synth_df[FEAT_COLS].fillna(real_df[FEAT_COLS].mean())

    combined = pd.concat([real_part, synth_part], ignore_index=True)

    scaler          = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    perplexity = min(30, len(combined_scaled) // 4)
    tsne       = TSNE(n_components=2, perplexity=perplexity,
                      random_state=42, n_iter=1000,
                      learning_rate='auto', init='pca')
    coords = tsne.fit_transform(combined_scaled)

    n_real        = len(real_part)
    real_coords   = coords[:n_real]
    synth_coords  = coords[n_real:]

    # Light background tint for this panel
    ax.set_facecolor('#F7F9FC')
    ax.grid(True, color='white', linewidth=1.0, zorder=0)

    # KDE contour lines — plotted below scatter points
    add_kde_contours(ax, real_coords,  REAL_COLOR,  levels=5, alpha=0.30)
    add_kde_contours(ax, synth_coords, synth_color, levels=5, alpha=0.40)

    # Real patients — solid grey circles, drawn first (below)
    ax.scatter(real_coords[:, 0], real_coords[:, 1],
               c=REAL_COLOR, s=48, alpha=0.65, label='Real patients',
               edgecolors=REAL_EDGE_COLOR, linewidths=0.5, zorder=3)

    # Synthetic records — vibrant coloured triangles on top
    ax.scatter(synth_coords[:, 0], synth_coords[:, 1],
               c=synth_color, s=55, alpha=0.80, label=method_name,
               edgecolors='white', linewidths=0.5, zorder=4, marker='^')

    # Panel title and overlap note
    n_real_shown  = len(real_coords)
    n_synth_shown = len(synth_coords)
    ax.set_title(f'Real vs {method_name}',
                 fontsize=14*_S, fontweight='bold', pad=10)
    ax.set_xlabel('t-SNE dimension 1', fontsize=10*_S, labelpad=6)
    ax.set_ylabel('t-SNE dimension 2', fontsize=10*_S, labelpad=6)
    ax.tick_params(labelsize=9, length=0)

    legend = ax.legend(fontsize=10*_S, frameon=True,
                       edgecolor='#CCCCCC', facecolor='white',
                       loc='upper right', markerscale=1.3)
    legend.get_frame().set_linewidth(0.8)

    # Sample size annotation in lower-left corner
    ax.text(0.02, 0.02,
            f'n real = {n_real_shown}   n synthetic = {n_synth_shown}',
            transform=ax.transAxes, fontsize=8.5*_S,
            color='#555555', va='bottom')

plt.tight_layout(pad=2.5)
plt.savefig('output/figures/tsne_real_vs_synthetic.png', bbox_inches='tight')

print('t-SNE figure saved.')
