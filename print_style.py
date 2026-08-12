"""
Print-Safe Figure Styling for IEEE Two-Column Layout
====================================================

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

The problem this solves
-----------------------
Matplotlib font sizes are expressed in points relative to the figure canvas,
not relative to the printed page. A figure created with figsize=(16, 5) and
9 pt tick labels is 16 inches wide on its own canvas; when LaTeX places it at
\\columnwidth (about 3.5 inches in IEEEtran) the whole thing is scaled down by
16 / 3.5 = 4.6x, and those 9 pt labels reach the reader at roughly 2 pt. That is
unreadable in print, which is exactly the problem reported on the first draft.

The rule
--------
    printed_pt = design_pt * (display_width / figsize_width)

so to hit a target size on the page:

    design_pt = target_pt * (figsize_width / display_width)

set_print_style() applies that correction. Give it the width the figure was
designed at and the width LaTeX will display it at, and it scales every text
element so the result lands at TARGET_PT on the printed page.

Display widths in IEEEtran
--------------------------
    single column   \\columnwidth   ~3.5 in   -> figure   environment
    full page       \\textwidth     ~7.16 in  -> figure*  environment

Multi-panel figures should use the full page width. Three panels inside a 3.5
inch column leaves about 1.1 inch per panel, which no font size can rescue.

Usage
-----
    from print_style import set_print_style
    set_print_style(figsize_width=16, display_width=7.16)   # a figure* row
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
"""

import matplotlib.pyplot as plt

# Width available to a figure in IEEEtran, in inches.
COLUMN_WIDTH = 3.5     # \columnwidth  -> figure
TEXT_WIDTH   = 7.16    # \textwidth    -> figure*

# Target size of body text on the printed page. IEEE captions run at 8 pt and
# figure text should not be smaller than the caption it sits under.
TARGET_PT = 8.0


def set_print_style(figsize_width, display_width=TEXT_WIDTH, target_pt=TARGET_PT):
    """
    Scale all matplotlib text so it prints at target_pt.

    Parameters
    ----------
    figsize_width  : width passed to plt.subplots(figsize=(W, H)), in inches
    display_width  : width LaTeX will render it at (COLUMN_WIDTH or TEXT_WIDTH)
    target_pt      : desired size of tick labels on the printed page

    Returns
    -------
    float : the scale factor applied, useful for sizing annotations by hand
    """
    scale = figsize_width / float(display_width)

    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         target_pt * scale,
        "axes.titlesize":    (target_pt + 1.0) * scale,
        "axes.labelsize":    target_pt * scale,
        "xtick.labelsize":   (target_pt - 0.5) * scale,
        "ytick.labelsize":   (target_pt - 0.5) * scale,
        "legend.fontsize":   (target_pt - 0.5) * scale,

        # High contrast for print: black text on white, no tinted panels.
        "text.color":        "black",
        "axes.labelcolor":   "black",
        "xtick.color":       "black",
        "ytick.color":       "black",
        "axes.edgecolor":    "black",
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",

        # Lines and spines also shrink with the figure, so scale them too.
        "axes.linewidth":    0.8 * scale,
        "lines.linewidth":   1.4 * scale,
        "xtick.major.width": 0.8 * scale,
        "ytick.major.width": 0.8 * scale,
        "grid.linewidth":    0.5 * scale,
        "patch.linewidth":   0.6 * scale,

        "axes.spines.top":   False,
        "axes.spines.right": False,
        "savefig.dpi":       300,
        "figure.dpi":        100,
    })
    return scale


def printed_size(design_pt, figsize_width, display_width=TEXT_WIDTH):
    """Return the size design_pt will appear at once LaTeX scales the figure."""
    return design_pt * (display_width / float(figsize_width))
