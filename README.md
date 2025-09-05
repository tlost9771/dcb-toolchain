[![DOI](https://zenodo.org/badge/1051022869.svg)](https://doi.org/10.5281/zenodo.17062605)

https://doi.org/10.5281/zenodo.17062605

# Dynamical Coherence Budget: Models, Toolchain, and Case Studies

**Author:** Sajjad Saei  
**Contact:** sajjadsaei97@gmail.com

This repository contains the code to reproduce all LaTeX-ready figures and the LaTeX table used in the paper.
The script `dcb_figs_all.py` generates:
- CAC figures (exposure map; exposure–makespan trade-off),
- RCD figures (rolling windows; policy regions; ROC/PR with operating point; TTD histogram) + a LaTeX table,
- Packet-level link figures (budget/yield vs packet length; threshold regions; throughput; sensitivity heatmap; latency CDF).

## How to run

### Option A: pip
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python dcb_figs_all.py            # outputs to ./figs
# or specify another directory
python dcb_figs_all.py ./figs_out
```

### Option B: conda/mamba
```bash
mamba env create -f environment.yml
mamba activate dcb-toolchain
python dcb_figs_all.py
```

## Expected outputs (in `figs/`)
PDF + PNG for each of:
- `fig_CAC_exposure_map`, `fig_CAC_tradeoff_curve`
- `fig_RCD_roll_windows`, `fig_RCD_policy_regions`,
  `fig_RCD_ROC`, `fig_RCD_ROC_marked`,
  `fig_RCD_PR`, `fig_RCD_PR_marked`, `fig_RCD_TTD`
- `fig_Qlink_Bp_vs_tau`, `fig_Qlink_threshold_policy`,
  `fig_Qlink_throughput_tau`, `fig_Qlink_sensitivity_heatmap`,
  `fig_Qlink_latency_cdf`

Plus:
- `figs/rcd_summary_table.tex`
- `figs/manifest.json` (list of all generated files)

## Use in LaTeX
```latex
\graphicspath{{figs/}}
% ...
\includegraphics[width=.47\linewidth]{fig_CAC_exposure_map.pdf}
```
For the table:
```latex
\input{figs/rcd_summary_table.tex}
```

## License and citation
- License: MIT (see `LICENSE`).
- Please cite the repository using `CITATION.cff` (Zenodo DOI will be added after the first release).
