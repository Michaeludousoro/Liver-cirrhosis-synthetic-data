"""
Competing-Risks Cumulative Incidence Figure
==============================================

Purpose
-------
The reviewer noted that coding transplantation as ordinary censoring
(alongside patients who remain alive at data cutoff) treats a competing
risk as if it were simple loss to follow-up. The survival-modelling
framing (Section IV, Cox/RSF) uses a cause-specific hazard for death,
censoring at transplant, which is a standard and legitimate treatment
when death is the endpoint of interest — but it assumes transplant timing
is non-informative about death risk, an assumption worth checking rather
than asserting. This script computes the non-parametric Aalen-Johansen
cumulative incidence function (CIF) for death and transplant as competing
events, which does not require that assumption, and quantifies how much
of the cohort each event type actually accounts for.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import AalenJohansenFitter

_S = 2.0
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5*_S, "axes.titlesize": 9.5*_S,
    "axes.labelsize": 8.5*_S, "xtick.labelsize": 7.5*_S, "ytick.labelsize": 7.5*_S,
    "legend.fontsize": 7.5*_S, "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.linewidth": 0.8*_S, "lines.linewidth": 1.4*_S, "xtick.major.width": 0.8*_S,
    "ytick.major.width": 0.8*_S, "patch.linewidth": 0.6*_S, "savefig.dpi": 300,
})


def main():
    df = pd.read_csv("data/raw/cirrhosis.csv")
    complete = df.dropna().copy()

    event_code = complete["Status"].map({"C": 0, "D": 1, "CL": 2})
    duration = complete["N_Days"].values

    n_death = int((event_code == 1).sum())
    n_transplant = int((event_code == 2).sum())
    n_censored = int((event_code == 0).sum())
    print(f"n={len(complete)}: death={n_death} ({100*n_death/len(complete):.1f}%), "
          f"transplant={n_transplant} ({100*n_transplant/len(complete):.1f}%), "
          f"censored-alive={n_censored} ({100*n_censored/len(complete):.1f}%)")

    ajf_death = AalenJohansenFitter()
    ajf_death.fit(duration, event_code, event_of_interest=1, label="Death")

    ajf_transplant = AalenJohansenFitter()
    ajf_transplant.fit(duration, event_code, event_of_interest=2, label="Transplant")

    cif_death_end = float(ajf_death.cumulative_density_.iloc[-1, 0])
    cif_transplant_end = float(ajf_transplant.cumulative_density_.iloc[-1, 0])
    print(f"Cumulative incidence by end of follow-up: death={cif_death_end:.4f}, "
          f"transplant={cif_transplant_end:.4f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ajf_death.plot(ax=ax, color="#dc2626", linewidth=2)
    ajf_transplant.plot(ax=ax, color="#1e3a5f", linewidth=2)
    ax.set_xlabel("N_Days", fontsize=11*_S)
    ax.set_ylabel("Cumulative incidence", fontsize=11*_S)
    ax.set_title("Competing-risks cumulative incidence:\ndeath vs. transplant "
                f"(n={len(complete)})", fontsize=11*_S)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9*_S)

    plt.tight_layout()
    out = "paper/figures_supplementary/figS11_competing_risks.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    summary = pd.DataFrame([{
        "n_total": len(complete), "n_death": n_death, "n_transplant": n_transplant,
        "n_censored_alive": n_censored,
        "cif_death_end_of_followup": round(cif_death_end, 4),
        "cif_transplant_end_of_followup": round(cif_transplant_end, 4),
    }])
    summary.to_csv("output/results/competing_risks_summary.csv", index=False)
    print("Saved output/results/competing_risks_summary.csv")


if __name__ == "__main__":
    main()
