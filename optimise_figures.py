"""
Optimise Figures for LaTeX Compilation
======================================

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

Purpose of this module
-----------------------
Matplotlib saves the publication figures at 300 DPI on large canvases, which
produces PNG files of up to several megabytes and tens of megapixels. LaTeX has
to decompress and embed every one of them, and on a metered service such as
Overleaf's free tier the resulting compile can exceed the time limit.

An IEEE single-column figure is about 3.5 inches wide, so even at 600 DPI it
only needs roughly 2100 pixels across. Anything beyond that adds file size and
compile time without adding any detail a reader or printer can resolve.

This script downsamples the numbered paper figures in paper/figures/ so that
neither side exceeds MAX_EDGE pixels, using high-quality Lanczos resampling,
and re-encodes them with maximum PNG compression. Run it after
organise_figures.py:

    python organise_figures.py
    python optimise_figures.py
"""

import os
import glob

from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(BASE_DIR, "paper", "figures")

# Longest edge in pixels. Every figure in the paper is placed at
# width=\columnwidth, i.e. about 3.5 inches, so 1600 pixels still corresponds
# to roughly 450 DPI in print — well above the 300 DPI publication standard.
MAX_EDGE = 1600


def optimise(path):
    """Downsample one figure if oversized and recompress it. Returns (before, after) bytes."""
    before = os.path.getsize(path)

    with Image.open(path) as img:
        img = img.convert("RGB") if img.mode in ("RGBA", "P") else img
        width, height = img.size

        if max(width, height) > MAX_EDGE:
            scale      = MAX_EDGE / float(max(width, height))
            new_size   = (int(width * scale), int(height * scale))
            img        = img.resize(new_size, Image.LANCZOS)
        else:
            new_size = (width, height)

        img.save(path, "PNG", optimize=True, compress_level=9)

    after = os.path.getsize(path)
    return before, after, (width, height), new_size


def main():
    paths = sorted(glob.glob(os.path.join(FIG_DIR, "fig[0-2][0-9]_*.png")))
    if not paths:
        print(f"  No numbered figures found in {FIG_DIR}. Run organise_figures.py first.")
        return

    total_before = total_after = 0
    for path in paths:
        before, after, old_size, new_size = optimise(path)
        total_before += before
        total_after  += after
        name   = os.path.basename(path)
        change = "resized" if old_size != new_size else "recompressed"
        print(f"  {name:<40} {before/1048576:5.2f} -> {after/1048576:5.2f} MB  "
              f"({old_size[0]}x{old_size[1]} -> {new_size[0]}x{new_size[1]}, {change})")

    saved = 100.0 * (1 - total_after / total_before) if total_before else 0.0
    print(f"\n  Total: {total_before/1048576:.2f} MB -> {total_after/1048576:.2f} MB "
          f"({saved:.0f} percent smaller)")


if __name__ == "__main__":
    main()
