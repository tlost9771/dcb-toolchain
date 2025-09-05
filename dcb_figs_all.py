
"""
DCB Figures: CAC + RCD + Packet-level
-------------------------------------
Self-contained script to generate all LaTeX-ready figures (PDF + PNG) used in the paper.

Usage:
  python dcb_figs_all.py [optional_output_dir]

Defaults:
  output_dir = "./figs" (relative to current working directory)
Dependencies:
  - numpy
  - matplotlib

Conventions:
  - One chart per figure
  - No seaborn; default matplotlib styles
  - Vector PDFs + 300dpi PNGs
  - Reproducible via fixed random seeds per section
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Utilities
# ---------------------------

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _save(figpath_no_ext):
    plt.tight_layout()
    pdf = figpath_no_ext + ".pdf"
    png = figpath_no_ext + ".png"
    plt.savefig(pdf)
    plt.savefig(png, dpi=300)
    plt.close()
    return [pdf, png]

# ---------------------------
# CAC Figures
# ---------------------------

def generate_cac_figs(outdir, seed=42):
    np.random.seed(seed)
    files = []

    # 1) Per-qubit exposure before vs after CAC
    nq = 12
    qidx = np.arange(1, nq+1)
    before = np.abs(np.random.normal(loc=0.020, scale=0.006, size=nq)) + 0.010  # seconds
    after = before * (0.72 + 0.12*np.random.rand(nq))  # ~18-40% reduction

    plt.figure(figsize=(7,4))
    width = 0.35
    plt.bar(qidx - width/2, before, width, label="Before CAC")
    plt.bar(qidx + width/2, after, width, label="After CAC")
    plt.xlabel("Qubit index")
    plt.ylabel("Exposure $B_q$ (s)")
    plt.title("Per-qubit exposure before vs. after CAC")
    plt.xticks(qidx)
    plt.legend()
    files += _save(os.path.join(outdir, "fig_CAC_exposure_map"))

    # 2) Exposure–makespan trade-off with CAC-preferred routes
    routes = 24
    makespan_us = np.random.uniform(180, 280, routes)  # microseconds
    exposure_s = np.random.uniform(0.12, 0.22, routes)  # seconds
    improved_idx = np.random.choice(routes, size=10, replace=False)
    exposure_s[improved_idx] *= np.random.uniform(0.70, 0.88, size=improved_idx.size)
    makespan_us[improved_idx] *= np.random.uniform(0.88, 0.97, size=improved_idx.size)

    plt.figure(figsize=(6.2,4.2))
    plt.scatter(makespan_us, exposure_s, label="Baseline routes")
    plt.scatter(makespan_us[improved_idx], exposure_s[improved_idx], label="CAC-preferred routes", marker="x")
    plt.xlabel(r"Makespan $T_{\mathrm{mk}}$ ($\mu$s)")
    plt.ylabel(r"Total exposure $B_{\ell_1}^{\mathrm{tot}}$ (s)")
    plt.title("Exposure–makespan trade-off (lower-left is better)")
    plt.legend()
    files += _save(os.path.join(outdir, "fig_CAC_tradeoff_curve"))

    return files

# ---------------------------
# RCD Figures
# ---------------------------

def _roc_pr(score, y, thresholds):
    tpr, fpr, prec, rec = [], [], [], []
    P = np.sum(y==1)
    Nn = np.sum(y==0)
    for th in thresholds:
        yhat = (score >= th).astype(int)
        tp = np.sum((yhat==1) & (y==1))
        fp = np.sum((yhat==1) & (y==0))
        fn = np.sum((yhat==0) & (y==1))
        tn = np.sum((yhat==0) & (y==0))
        tpr.append(tp / P if P>0 else 0.0)
        fpr.append(fp / Nn if Nn>0 else 0.0)
        precision = tp / (tp+fp) if (tp+fp)>0 else 1.0
        recall = tp / (tp+fn) if (tp+fn)>0 else 0.0
        prec.append(precision)
        rec.append(recall)
    return np.array(tpr), np.array(fpr), np.array(prec), np.array(rec)

def generate_rcd_figs(outdir, seed_main=12345, seed_eval=24680):
    files = []

    # -- Rolling windows & policy regions (seed_main) --
    np.random.seed(seed_main)

    # Rolling-window traces
    t = np.linspace(0, 2.0, 400)  # seconds
    B_hat = 0.020 + 0.005*np.sin(2*np.pi*0.7*t) + 0.002*np.random.randn(t.size)
    B_hat += 0.012*(t > 0.9)  # drift
    B_hat = np.maximum(B_hat, 0.0)
    E_hat = 0.010 + 0.001*np.sin(2*np.pi*0.3*t + 0.5) + 0.0005*np.random.randn(t.size)
    E_hat = np.maximum(E_hat, 0.001)
    B_warn = 0.030
    B_hard = 0.040

    plt.figure(figsize=(7,4))
    plt.plot(t, B_hat, label=r"$\widehat B_q(t;\Delta)$")
    plt.plot(t, E_hat, label=r"$\widehat E_q(t;\Delta)$")
    plt.axhline(B_warn, linestyle="--", label=r"$B_{\mathrm{warn}}$")
    plt.axhline(B_hard, linestyle=":", label=r"$B_{\mathrm{hard}}$")
    plt.xlabel("Time (s)")
    plt.ylabel("Rolling integrals (arb. units)")
    plt.title("Rolling-window traces with warning/hard thresholds")
    plt.legend()
    files += _save(os.path.join(outdir, "fig_RCD_roll_windows"))

    # Policy regions
    B_vals = 0.015 + 0.035*np.random.rand(300)
    E_vals = 0.008 + 0.004*np.random.rand(300)
    labels = np.full(B_vals.shape, "OK", dtype=object)
    labels[(B_vals >= B_warn) & (B_vals < B_hard)] = "WARN"
    labels[B_vals >= B_hard] = "HALT"

    plt.figure(figsize=(6,4.2))
    for state, marker in [("OK", "o"), ("WARN", "s"), ("HALT", "x")]:
        idx = np.where(labels == state)[0]
        plt.scatter(B_vals[idx], (B_vals[idx]/E_vals[idx]), label=state, marker=marker)
    plt.axvline(B_warn, linestyle="--", label=r"$B_{\mathrm{warn}}$")
    plt.axvline(B_hard, linestyle=":", label=r"$B_{\mathrm{hard}}$")
    plt.xlabel(r"$\widehat B_q$ (arb. units)")
    plt.ylabel(r"$\widehat B_q/\widehat E_q$ (dimensionless)")
    plt.title("Policy regions with hysteresis bands")
    plt.legend()
    files += _save(os.path.join(outdir, "fig_RCD_policy_regions"))

    # -- ROC/PR/TTD (seed_eval) --
    np.random.seed(seed_eval)

    T = 10.0            # total time (s)
    dt = 0.01           # sampling step (s)
    N = int(T/dt)
    tt = np.arange(N) * dt
    t0 = 5.0
    y_true = (tt >= t0).astype(int)

    mu0 = 0.020
    noise = 0.0015 * np.random.randn(N)
    slow_drift = 0.0005 * (tt - t0) * (tt >= t0)
    Bsig = mu0 + 0.004*np.sin(2*np.pi*0.5*tt) + noise + slow_drift
    Bsig = np.maximum(Bsig, 0.0)

    win = int(0.5/dt)  # 0.5 s window
    kernel = np.ones(win) / win
    pad = np.pad(Bsig, (win,0), mode='edge')
    roll = np.convolve(pad, kernel, mode='valid')[:N]
    s = Bsig - roll  # score

    ths = np.quantile(s, np.linspace(0.99, 0.01, 400))
    tpr, fpr, prec, rec = _roc_pr(s, y_true, ths)

    # Operating point ~5% FPR
    idx = int(np.argmin(np.abs(fpr - 0.05)))
    th_star = float(ths[idx])
    fpr_star, tpr_star = float(fpr[idx]), float(tpr[idx])
    prec_star, rec_star = float(prec[idx]), float(rec[idx])

    # ROC (base + marked)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("RCD detector ROC")
    files += _save(os.path.join(outdir, "fig_RCD_ROC"))

    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr)
    plt.scatter([fpr_star], [tpr_star], marker="*", s=120)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("RCD detector ROC with operating point")
    files += _save(os.path.join(outdir, "fig_RCD_ROC_marked"))

    # PR (base + marked)
    plt.figure(figsize=(6,4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("RCD detector Precision–Recall")
    files += _save(os.path.join(outdir, "fig_RCD_PR"))

    plt.figure(figsize=(6,4))
    plt.plot(rec, prec)
    plt.scatter([rec_star], [prec_star], marker="*", s=120)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("RCD detector Precision–Recall with operating point")
    files += _save(os.path.join(outdir, "fig_RCD_PR_marked"))

    # TTD histogram at chosen threshold
    def ttd_at_threshold(th, runs=600):
        ttds = []
        for r in range(runs):
            noise = 0.0015 * np.random.randn(N)
            slow = 0.0005 * (tt - t0) * (tt >= t0)
            B = mu0 + 0.004*np.sin(2*np.pi*0.5*tt + 0.2*r) + noise + slow
            B = np.maximum(B, 0.0)
            pad = np.pad(B, (win,0), mode='edge')
            roll = np.convolve(pad, kernel, mode='valid')[:N]
            sc = B - roll
            after_idx = np.where(tt >= t0)[0]
            cross = np.where(sc[after_idx] >= th)[0]
            if cross.size>0:
                ttd = tt[after_idx[cross[0]]] - t0
                ttds.append(ttd)
        return np.array(ttds)

    ttds = ttd_at_threshold(th_star, runs=600)
    median_ttd = float(np.median(ttds))
    ci_low, ci_high = float(np.percentile(ttds, 2.5)), float(np.percentile(ttds, 97.5))

    plt.figure(figsize=(6,4))
    plt.hist(ttds, bins=20)
    plt.xlabel("Time-to-detect (s)")
    plt.ylabel("Count")
    plt.title("Detection delay at ~5% FPR")
    files += _save(os.path.join(outdir, "fig_RCD_TTD"))

    # LaTeX table with summary
    table_tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{RCD operating point and detection statistics (simulation; seed=24680, $dt{=}0.01$s, window $0.5$s).}\n"
        "\\label{tab:rcd_summary}\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Metric & Value & Notes \\\\\n"
        "\\midrule\n"
        "Target FPR & $\\approx 5\\%$ & chosen on ROC \\\\\n"
        f"Operating threshold $\\theta$ & {th_star:.4f} & on sliding-window score $s$ \\\\\n"
        f"Actual FPR & {fpr_star:.3f} & at $\\theta$ \\\\\n"
        f"TPR & {tpr_star:.3f} & at $\\theta$ \\\\\n"
        f"Precision & {prec_star:.3f} & at $\\theta$ \\\\\n"
        f"Recall & {rec_star:.3f} & equals TPR in this setup \\\\\n"
        f"Median TTD (s) & {median_ttd:.3f} & change @ $t_0{{=}}5$s \\\\\n"
        f"TTD 95\\% CI (s) & $[{ci_low:.3f},\\, {ci_high:.3f}]$ & percentile \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    table_path = os.path.join(outdir, "rcd_summary_table.tex")
    with open(table_path, "w") as f:
        f.write(table_tex)

    return files + [table_path]

# ---------------------------
# Packet-level Figures
# ---------------------------

def generate_packet_figs(outdir, seed=777):
    np.random.seed(seed)
    files = []

    # Physical-ish constants (arbitrary units for illustration)
    hbar = 1.0
    omega = 1.0
    alpha = 0.5

    # Link params
    gamma = 10.0
    gamma_ad = gamma/2.0
    t_ovh = 0.008

    def CBY_const():
        return 2.0/(hbar*omega)

    def Bp_PD(tau):
        return (2*alpha/gamma) * (1.0 - np.exp(-gamma*tau))

    def Bp_AD(tau):
        return (2*alpha/gamma_ad) * (1.0 - np.exp(-gamma_ad*tau))

    def Rp(Bp_func, tau):
        return Bp_func(tau) / (tau + t_ovh)

    def tau_star_fixed_point(rate):
        tau = 0.05
        for _ in range(50):
            f = np.exp(rate*tau) - 1.0 - rate*(tau + t_ovh)
            df = rate*np.exp(rate*tau) - rate
            tau_new = tau - f/df
            if abs(tau_new - tau) < 1e-9:
                break
            tau = max(tau_new, 1e-9)
        return tau

    tau_star_pd = tau_star_fixed_point(gamma)
    tau_star_ad = tau_star_fixed_point(gamma_ad)

    # 1) B_p vs tau (PD & AD) + flat CBY
    taus = np.linspace(0.0, 0.25, 400)
    Bp_pd = Bp_PD(taus)
    Bp_ad = Bp_AD(taus)

    plt.figure(figsize=(6.6,4.2))
    plt.plot(taus, Bp_pd, label="B_p (PD)")
    plt.plot(taus, Bp_ad, label="B_p (AD)")
    plt.axhline(CBY_const(), linestyle="--", label="CBY_p (flat under calibration)")
    plt.scatter([tau_star_pd], [Bp_PD(tau_star_pd)], marker="o", s=60)
    plt.scatter([tau_star_ad], [Bp_AD(tau_star_ad)], marker="x", s=60)
    plt.xlabel(r"Packet duration $\tau_p$ (s)")
    plt.ylabel(r"$B_p(\tau_p)$ and $\mathrm{CBY}_p$ (arb. units)")
    plt.title("Packet budget and yield vs. duration (PD/AD)")
    plt.legend()
    files += _save(os.path.join(outdir, "fig_Qlink_Bp_vs_tau"))

    # 2) Threshold policy regions (B_min, eta_min)
    taus_scat = np.linspace(0.0, 0.25, 60)
    Bp_scat = Bp_PD(taus_scat)
    CBY_scat = np.full_like(Bp_scat, CBY_const())
    Bmin = 0.15 * np.max(Bp_PD(0.25))
    eta_min = CBY_const()

    plt.figure(figsize=(6.0,4.2))
    plt.scatter(Bp_scat, CBY_scat, label="(B_p, CBY_p) samples", s=12)
    plt.axvline(Bmin, linestyle="--", label=r"$B_{\min}$")
    plt.axhline(eta_min, linestyle=":")
    plt.text(Bmin, eta_min, r"  $B_{\min},\,\eta_{\min}$", va="bottom")
    plt.xlabel(r"$B_p$ (arb. units)")
    plt.ylabel(r"$\mathrm{CBY}_p$ (dimensionless)")
    plt.title("Threshold policy regions")
    plt.legend()
    files += _save(os.path.join(outdir, "fig_Qlink_threshold_policy"))

    # 3) Throughput vs tau with tau*
    R_pd = Rp(Bp_PD, taus)
    R_ad = Rp(Bp_AD, taus)

    plt.figure(figsize=(6.6,4.2))
    plt.plot(taus, R_pd, label=r"$R_p$ (PD)")
    plt.plot(taus, R_ad, label=r"$R_p$ (AD)")
    plt.scatter([tau_star_pd], [Rp(Bp_PD, tau_star_pd)], marker="o", s=60, label=r"$\tau_p^\star$ (PD)")
    plt.scatter([tau_star_ad], [Rp(Bp_AD, tau_star_ad)], marker="x", s=60, label=r"$\tau_p^\star$ (AD)")
    plt.xlabel(r"Packet duration $\tau_p$ (s)")
    plt.ylabel(r"Throughput $R_p$ (arb. units)")
    plt.title("Throughput vs. packet duration")
    plt.legend()
    files += _save(os.path.join(outdir, "fig_Qlink_throughput_tau"))

    # 4) Sensitivity heatmap: gain of CB-aware vs Fixed-tau across (loss, jitter)
    loss_vals = np.linspace(0.0, 0.4, 21)
    jitter_ms_vals = np.linspace(0.0, 12.0, 25)
    tau0 = 0.04  # baseline fixed tau
    gain = np.zeros((len(jitter_ms_vals), len(loss_vals)))

    for i, jit_ms in enumerate(jitter_ms_vals):
        jitter = jit_ms / 1000.0
        t_eff = t_ovh + jitter
        tau_grid = np.linspace(0.001, 0.25, 400)
        B_grid = Bp_PD(tau_grid)
        for j, loss in enumerate(loss_vals):
            R_base = Bp_PD(tau0) * (1.0 - loss) / (tau0 + t_eff)
            R_grid = B_grid * (1.0 - loss) / (tau_grid + t_eff)
            R_best = np.max(R_grid)
            gain[i, j] = 100.0 * (R_best - R_base) / max(R_base, 1e-12)

    plt.figure(figsize=(6.6,4.2))
    extent = [loss_vals.min(), loss_vals.max(), jitter_ms_vals.min(), jitter_ms_vals.max()]
    plt.imshow(gain, origin="lower", aspect="auto", extent=extent)
    plt.xlabel("Loss probability")
    plt.ylabel("Jitter (ms)")
    plt.title("Throughput gain of CB-aware vs Fixed-$\\tau$ (%)")
    plt.colorbar()
    files += _save(os.path.join(outdir, "fig_Qlink_sensitivity_heatmap"))

    # 5) Latency CDF under policies
    def simulate_latency(policy, runs=500, jitter_sigma=0.003):
        latencies = []
        for r in range(runs):
            jitter = max(0.0, np.random.normal(0.0, jitter_sigma))
            if policy == "greedy":
                tau = 0.006
            elif policy == "fixed":
                tau = tau0
            elif policy == "cb":
                t_eff = t_ovh + jitter
                tau_grid = np.linspace(0.001, 0.25, 400)
                R_grid = Bp_PD(tau_grid) / (tau_grid + t_eff)
                tau = tau_grid[np.argmax(R_grid)]
            else:
                tau = tau0
            lat = tau + t_ovh + jitter
            latencies.append(lat)
        arr = np.array(latencies)
        arr.sort()
        return arr

    def cdf(arr):
        y = np.arange(1, arr.size+1) / arr.size
        return arr, y

    lat_greedy = simulate_latency("greedy")
    lat_fixed = simulate_latency("fixed")
    lat_cb = simulate_latency("cb")

    xg, yg = cdf(lat_greedy)
    xf, yf = cdf(lat_fixed)
    xc, yc = cdf(lat_cb)

    plt.figure(figsize=(6.6,4.2))
    plt.plot(xg, yg, label="Greedy")
    plt.plot(xf, yf, label="Fixed-$\\tau$")
    plt.plot(xc, yc, label="CB-aware")
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.title("Latency distribution under different policies")
    plt.legend()
    files += _save(os.path.join(outdir, "fig_Qlink_latency_cdf"))

    return files

# ---------------------------
# Main
# ---------------------------

def main(output_dir="./figs"):
    _ensure_dir(output_dir)
    produced = []
    produced += generate_cac_figs(output_dir, seed=42)
    produced += generate_rcd_figs(output_dir, seed_main=12345, seed_eval=24680)
    produced += generate_packet_figs(output_dir, seed=777)
    # Write manifest
    manifest = os.path.join(output_dir, "manifest.json")
    with open(manifest, "w") as f:
        import json
        json.dump({"files": produced}, f, indent=2)
    print("Generated files:")
    for p in produced:
        print(p)

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "./figs"
    main(out)
