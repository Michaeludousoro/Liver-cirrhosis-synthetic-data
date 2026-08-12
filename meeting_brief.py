"""
Build the supervisor meeting brief as a PDF.

Renders an HTML source document through PyMuPDF's Story engine, which handles
pagination, headings and tables without needing LaTeX or pandoc installed.

Usage:
    python meeting_brief.py
"""

import os
import fitz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "meeting_brief.pdf")

CSS = """
body { font-family: sans-serif; font-size: 9.5pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 16pt; margin: 0 0 2pt 0; color: #111; }
.sub { font-size: 9pt; color: #555; margin: 0 0 14pt 0; }
h2 { font-size: 11.5pt; margin: 15pt 0 5pt 0; color: #0b3d66; }
h3 { font-size: 9.8pt; margin: 9pt 0 2pt 0; color: #222; }
p { margin: 0 0 6pt 0; }
ul { margin: 0 0 6pt 0; }
li { margin: 0 0 3pt 0; }
table { width: 100%; border-collapse: collapse; margin: 5pt 0 9pt 0; font-size: 8.6pt; }
th { background: #eef3f8; text-align: left; padding: 3.5pt 5pt; border-bottom: 1px solid #b8ccdd; }
td { padding: 3.5pt 5pt; border-bottom: 0.5px solid #dde5ec; }
.num { text-align: center; }
b { color: #111; }
.q { margin: 0 0 7pt 0; }
.note { font-size: 8.6pt; color: #555; margin-top: 12pt; border-top: 0.5px solid #ccc; padding-top: 6pt; }
"""

HTML = """
<h1>Synthetic Data for Liver Cirrhosis Survival Prediction</h1>
<p class="sub">Supervisor meeting brief &middot; Michael Udousoro &middot; 1 August 2026</p>

<h2>What the paper does</h2>
<p>Three generative models (Vanilla GAN, conditional tabular GAN, TVAE) built from scratch in
TensorFlow, trained on 193 real PBC patients. Two quality filters applied: IQR Tukey fences for
clinical plausibility, then consensus voting requiring agreement across all three architectures.
Evaluated on 83 real held-out patients using four statistical methods together: held-out test,
5-fold cross-validation, 1000-resample bootstrap, and McNemar's exact test.</p>

<h2>Main result</h2>
<p>No augmentation scenario beats the real-data baseline. All McNemar p-values above 0.22, all
bootstrap confidence intervals overlap, cross-validation AUC neither rises nor stabilises.
Consistent across all four statistical methods.</p>

<h2>Achievements</h2>

<h3>Synthetic data substitutes for real data even where it cannot augment it</h3>
<p>Classifiers trained on CTGAN records alone, with zero real patients in training, reach AUC
0.82 to 0.84 on real held-out patients. Directly relevant to federated settings where records
cannot be shared across sites.</p>

<h3>Consensus voting increases disclosure risk rather than reducing it</h3>
<p>21.1% near-duplicates against 7.3% and 9.6% for the individual adversarial pools.
Counterintuitive, and a useful warning: agreement concentrates records in the dense centre of the
distribution, which is exactly where real patients sit.</p>

<h3>Similarity does not imply utility, demonstrated twice independently</h3>
<p>CTGAN has the best FID (0.0724) and delivers no classifier benefit. Separately, Cohen's d ranks
the TVAE best on all eleven features while every distributional-shape measure ranks it worst.</p>

<h3>A mechanism, not just an observation</h3>
<p>A model trained on 193 patients is a function of those patients, so by the data processing
inequality it cannot add information the training data did not already contain. This turns the
null into a principled statement with a testable boundary condition.</p>

<h3>Leakage-free evaluation</h3>
<p>Cross-validation folds on real patients only, with synthetic records confined to the training
side of every fold. Mixing synthetic data into validation inflates apparent benefit, and this is a
common failure in the literature.</p>

<h3>Full reproducibility and complete reporting</h3>
<p>Fixed seeds throughout; a clean re-run reproduced all 19 result tables byte-for-byte, verified
by checksum. The supplementary reports 44 KS tests, 44 Jensen-Shannon divergences and 44 Cohen's d
values rather than aggregates alone.</p>

<h2>Known weaknesses</h2>
<ul>
<li><b>Underpowered.</b> At n = 83 held out, McNemar has roughly 9% power to detect a 5-point
accuracy gain. Around 300 patients would be needed for 80%. The null cannot rule out a small real effect.</li>
<li><b>Single cohort, single disease.</b> No external validation.</li>
<li><b>The IoMT framing is asserted, not demonstrated.</b> The dataset is a Mayo Clinic trial with
no sensors or streams. This is the main reviewer risk.</li>
<li><b>CTGAN is not the published CTGAN.</b> Outcome-conditioning only, with no mode-specific
normalisation or training-by-sampling. Stated in the paper, but a reviewer may still object to the name.</li>
<li><b>Formal differential privacy is unreachable.</b> Epsilon stays above 14 even at aggressive
noise at n = 193. Output perturbation at sigma 0.7 is the practical alternative.</li>
</ul>

<h2>The power limitation, explained</h2>
<p>McNemar counts only patients where the two models <b>disagree</b>. If both are right, or both
wrong, that patient contributes nothing. So although there are 83 test patients, the test works
with 4 to 11 observations.</p>

<table>
<tr><th>Comparison</th><th class="num">n10</th><th class="num">n01</th><th class="num">Discordant</th><th class="num">p</th></tr>
<tr><td>RF, A vs B</td><td class="num">2</td><td class="num">2</td><td class="num">4</td><td class="num">1.000</td></tr>
<tr><td>RF, A vs C</td><td class="num">7</td><td class="num">3</td><td class="num">10</td><td class="num">0.344</td></tr>
<tr><td>GB, A vs B</td><td class="num">4</td><td class="num">2</td><td class="num">6</td><td class="num">0.688</td></tr>
<tr><td>GB, A vs C</td><td class="num">8</td><td class="num">3</td><td class="num">11</td><td class="num">0.227</td></tr>
<tr><td>LR, A vs B</td><td class="num">7</td><td class="num">4</td><td class="num">11</td><td class="num">0.549</td></tr>
<tr><td>LR, A vs C</td><td class="num">4</td><td class="num">5</td><td class="num">9</td><td class="num">1.000</td></tr>
</table>

<p>An 11-way split of 8 to 3 is like flipping a coin 11 times and getting 8 heads: suggestive, but
consistent with a fair coin. At 9% power, a genuine 5-point gain would be detected only about 9
times in 100. This is absence of evidence, not evidence of absence.</p>

<p><b>Why it does not sink the paper.</b> The p-values are not marginal, ranging from 0.227 to
1.000, so nothing sits just above threshold awaiting more data. The direction mostly favours the
baseline, with n10 exceeding n01 in four of six comparisons. And McNemar is not the only evidence:
the bootstrap intervals overlap, cross-validation AUC does not rise, and held-out metrics do not
improve. The power limitation applies to McNemar alone and does not explain away the other three.</p>

<h2>Questions</h2>
<p class="q"><b>1. Venue.</b> I targeted IoT-J on the MF-CGAN precedent, but the IoMT link is
motivational rather than real. Is JBHI or a clinical informatics venue the better target?</p>
<p class="q"><b>2. Title.</b> If we drop IoMT, I propose <i>Evaluating the Predictive Utility of
Synthetic Patient Data in Small Clinical Cohorts</i>. IoMT appears 17 times but only in framing, so
it is about a day of editing. Does that read right?</p>
<p class="q"><b>3.</b> Is the null result strong enough to carry a paper, or does it need the
larger-cohort replication attached before submission?</p>
<p class="q"><b>4. Naming.</b> Should I rename our conditional GAN to avoid the CTGAN comparison, or
is the stated caveat sufficient?</p>
<p class="q"><b>5. Next paper.</b> I want to run the pipeline across subsampled cohort sizes
(n = 100 to 5000) to find where augmentation begins to help. Does that seem like the right follow-up?</p>
<p class="q"><b>6. Request.</b> Would you act as my PhysioNet reference for MIMIC-IV access? Having
graduated I have no institutional email, and a named academic referee is the main route through
credentialing. I have also applied for SEER, which approves in about two days.</p>

<p class="note">Current state: 11 pages, 10 figures, 5 tables, plus a 7-page supplementary with 10
figures and 8 tables. 210 numbers cross-checked against pipeline output with zero mismatches.
Code at github.com/Michaeludousoro/liver-cirrhosis-synthetic-data</p>
"""


def main():
    story = fitz.Story(html=HTML, user_css=CSS)
    writer = fitz.DocumentWriter(OUT_PATH)
    page_rect = fitz.paper_rect("a4")
    content_rect = page_rect + (56, 52, -56, -52)

    pages = 0
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(content_rect)
        story.draw(device)
        writer.end_page()
        pages += 1
    writer.close()
    print(f"  Wrote {OUT_PATH} ({pages} page{'s' if pages != 1 else ''})")


if __name__ == "__main__":
    main()
