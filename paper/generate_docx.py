"""
Generate the Word (.docx) Version of the Paper
================================================

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

Why this file changed
----------------------
This used to be a ~900-line script that hand-typed every paragraph, table,
and number from main.tex a second time using python-docx. That duplication
is exactly the kind of thing that silently drifts: by the time this
correction pass started, the hand-typed version was missing entire sections
that already existed in main.tex (the CTGAN comparison, the federated IoMT
implications, the practical guidance section, the threats-to-validity
section) — it had not been kept in sync even before the N_Days leakage fix
that motivated this rewrite.

Instead, this script converts main.tex directly via pandoc, so the .docx is
always a rendering of the actual paper source, not a second copy of it that
someone has to remember to update. Two preprocessing fixes are needed
because pandoc's LaTeX reader does not understand plain-IEEEtran macros or
silently drops content it can't parse:

  1. \\resizebox{\\columnwidth}{!}{ ... } around a table causes pandoc to
     drop the table's content entirely, with no warning. The wrapper is
     stripped (Word doesn't need manual column-width shrinking); the table
     itself converts fine once unwrapped.
  2. \\author{\\IEEEauthorblockN{...}\\IEEEauthorblockA{...}...} uses
     IEEEtran-specific macros pandoc doesn't recognise, so the entire
     author/affiliation block is silently dropped. The \\author{} command
     is removed and the same information is inserted as an ordinary
     paragraph immediately after \\maketitle instead, which pandoc reads
     as plain body text.

Requires the `pypandoc_binary` package (bundles its own pandoc, no system
dependency): pip install pypandoc_binary
"""

import os
import re
import pypandoc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_TEX = os.path.join(BASE_DIR, "main.tex")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
OUTPUT_PATH = os.path.join(BASE_DIR, "liver_cirrhosis_paper.docx")
TMP_TEX = os.path.join(BASE_DIR, "..", "output", "data", "_main_docx_build.tex")

AUTHOR_PARAGRAPH = (
    r"Michael Udousoro, Mohammad Farhan Khan, Fakhreldin Saeed, and M. Mursaleen"
    r"\\ Department of Computing, School of Engineering, Computing and Design, "
    r"University of Roehampton, London, United Kingdom"
    r"\\ Department of Medical Research, China Medical University Hospital, "
    r"China Medical University, Taichung, Taiwan"
    r"\\ Corresponding author: Michael Udousoro (michaeludousoro13@gmail.com)"
)


def strip_resizebox(text):
    """Remove \\resizebox{\\columnwidth}{!}{ ... } wrappers around tables,
    which pandoc's LaTeX reader silently drops rather than parses."""
    lines = text.split("\n")
    out = []
    skip_next_close = False
    for line in lines:
        if line.strip() == r"\resizebox{\columnwidth}{!}{%":
            continue
        if skip_next_close:
            skip_next_close = False
            if line.strip() == "}":
                continue
        if line.strip() == r"\end{tabular}%":
            out.append(line)
            skip_next_close = True
            continue
        out.append(line)
    return "\n".join(out)


def replace_ieee_author_block(text):
    """Remove the IEEEtran \\author{...} block (unrecognised macros cause
    pandoc to drop it silently) and insert the same information as a plain
    paragraph right after \\maketitle instead."""
    author_pattern = re.compile(
        r"\\author\{\\IEEEauthorblockN\{.*?\\thanks\{.*?\}\}\}", re.DOTALL
    )
    text, n = author_pattern.subn("", text)
    if n != 1:
        raise ValueError(
            f"Expected exactly one IEEEtran \\author{{}} block, found {n}. "
            "main.tex's author block format may have changed; update the "
            "regex in replace_ieee_author_block() to match."
        )
    text = text.replace(r"\maketitle", r"\maketitle" + "\n\n" + AUTHOR_PARAGRAPH + "\n")
    return text


def build_docx():
    with open(MAIN_TEX) as f:
        text = f.read()

    text = strip_resizebox(text)
    text = replace_ieee_author_block(text)

    os.makedirs(os.path.dirname(TMP_TEX), exist_ok=True)
    with open(TMP_TEX, "w") as f:
        f.write(text)

    pypandoc.convert_file(
        TMP_TEX, "docx", outputfile=OUTPUT_PATH,
        extra_args=[f"--resource-path={FIGURES_DIR}"]
    )
    os.remove(TMP_TEX)
    print(f"DOCX saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_docx()
