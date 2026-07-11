#!/usr/bin/env python3
"""
drift_tracking.py -- PA drift tracking (the selling point).

Claim to prove: under a TIME-VARYING PA, a FROZEN DPD degrades, while ONLINE ILA
(the on-chip RTRL path, no gradient through the PA) tracks the drift and holds
linearization. Periodic host-retrain is the middle ground (good right after a
recal, decays in between).

Drift model (applied to the loaded DGRU PA, physically interpretable):
    drifted(xpd) = g(t) * ( PA(xpd) + beta(t) * PA(xpd)|PA(xpd)|^2 )
    * g(t) = A(t) e^{j phi(t)}   thermal-like gain droop + phase rotation ramp
    * a mid-run BIAS STEP (sudden gain/phase jump)
    * beta(t) growing 3rd-order compression (operating-point drift)
All three DPDs start from the SAME offline s4d_best (tuned for the nominal PA).

Strategies (identical ILA learning; only the CADENCE differs):
    frozen   : never updates
    periodic : big ILA burst every K blocks (host recal), else nothing
    online   : a few ILA steps every block (continuous, small-lr tracking)

Run:  uv run experiments/drift_tracking.py   (saves results/drift_tracking.png)
"""
import copy
import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "results")

from online_adaptation import (make_pa, make_dpd, target_gain_of, segment,
                           DATASET, DEVICE, FS, BW, NSUB, NPERSEG)
from modules.data_collector import load_dataset
from utils.metrics import NMSE, EVM, ACLR

N_BLOCKS = 30
STEP_BLK = 15                    # bias step happens here
RETRAIN_EVERY = 6                # periodic host recal cadence (blocks)
rng = np.random.default_rng(0)


import drift_models

# Which PA-drift operator this run uses.  'memoryless' is the historical behaviour,
# bit-for-bit; see drift_models.py for why the memory-drifting arms exist.
DRIFT_KIND = "memoryless"


def drift_params(blk):
    return drift_models.drift_params(blk, N_BLOCKS, STEP_BLK)


def make_pa_fn(pa, blk):
    return drift_models.make_drift(pa, blk, N_BLOCKS, STEP_BLK, kind=DRIFT_KIND)


@torch.no_grad()
def eval_drift(M, pa_fn, X_np, tg, nperseg=NPERSEG):
    M.eval()
    Xs = segment(X_np, nperseg)
    yb = pa_fn(M(torch.tensor(Xs).to(DEVICE))).cpu().numpy().astype(np.float64)
    gt = tg * Xs.astype(np.float64)
    return dict(ACLR=float(np.mean(ACLR(yb, fs=FS, nperseg=nperseg, bw_main_ch=BW, n_sub_ch=NSUB))),
                EVM=float(EVM(yb, gt, sample_rate=FS, bw_main_ch=BW, n_sub_ch=NSUB, nperseg=nperseg)))


def ila_update(M, pa_fn, x_full, opt, n_frames, fl=500):
    """A few ILA post-inverse steps against the current (drifted) PA. No PA grad."""
    M.eval()
    with torch.no_grad():
        xpd = M(x_full)
        z = pa_fn(xpd)
    z = z.detach(); xpd_t = xpd.detach()
    N = x_full.shape[1]
    lossf = torch.nn.MSELoss()
    M.train()
    for s in rng.integers(0, N - fl, size=n_frames):
        s = int(s)
        loss = lossf(M(z[:, s:s + fl, :]), xpd_t[:, s:s + fl, :])
        opt.zero_grad(); loss.backward(); opt.step()


def main():
    torch.manual_seed(0)
    print(f"device = {DEVICE}  |  PA drift tracking")

    X_tr, y_tr, *_rest, X_te, y_te = load_dataset(dataset_name=DATASET)
    X_tr = np.asarray(X_tr, np.float32); X_te = np.asarray(X_te, np.float32)
    tg = target_gain_of(X_tr, np.asarray(y_tr, np.float32))
    x_full = torch.tensor(X_tr[None]).to(DEVICE)

    pa = make_pa()

    # three DPDs, all starting from the SAME offline s4d_best (tuned for nominal PA)
    Mf = make_dpd(load_offline=True)                       # frozen
    Mo = make_dpd(load_offline=True)                       # online ILA
    Mp = make_dpd(load_offline=True)                       # periodic retrain
    opt_o = torch.optim.Adam(Mo.parameters(), lr=5e-4)     # small-lr tracking
    opt_p = torch.optim.Adam(Mp.parameters(), lr=1e-3)

    hist = {k: {"ACLR": [], "EVM": []} for k in ("frozen", "periodic", "online")}
    print(f"\n{'blk':>3} {'drift':>18} | {'frozen':>14} | {'periodic':>14} | {'online':>14}")
    print(f"{'':>3} {'A/phi/b3/ampm':>18} | {'ACLR   EVM':>14} | {'ACLR   EVM':>14} | {'ACLR   EVM':>14}")
    for blk in range(N_BLOCKS):
        pa_fn = make_pa_fn(pa, blk)
        A, phi, b3, ampm = drift_params(blk)

        mf = eval_drift(Mf, pa_fn, X_te, tg)
        mp = eval_drift(Mp, pa_fn, X_te, tg)
        mo = eval_drift(Mo, pa_fn, X_te, tg)
        for k, m in [("frozen", mf), ("periodic", mp), ("online", mo)]:
            hist[k]["ACLR"].append(m["ACLR"]); hist[k]["EVM"].append(m["EVM"])

        tag = "  <-- bias step" if blk == STEP_BLK else ""
        print(f"{blk:3d} {A:4.2f}/{phi:4.2f}/{b3:4.2f}/{ampm:4.2f} | "
              f"{mf['ACLR']:6.1f} {mf['EVM']:6.1f} | {mp['ACLR']:6.1f} {mp['EVM']:6.1f} | "
              f"{mo['ACLR']:6.1f} {mo['EVM']:6.1f}{tag}")

        # --- adapt for next block ---
        ila_update(Mo, pa_fn, x_full, opt_o, n_frames=80)          # online: continuous
        if blk % RETRAIN_EVERY == 0:
            ila_update(Mp, pa_fn, x_full, opt_p, n_frames=200)     # periodic: burst recal

    # --- summary ---
    print("\n            |    ACLR (mean/final/worst)   |    EVM (mean/final/worst)")
    for k in ("frozen", "periodic", "online"):
        a = np.array(hist[k]["ACLR"]); e = np.array(hist[k]["EVM"])
        print(f"  {k:9s} | {a.mean():7.1f}/{a[-1]:6.1f}/{a.max():6.1f}      "
              f"| {e.mean():7.1f}/{e[-1]:6.1f}/{e.max():6.1f}")
    fa, oa = np.array(hist["frozen"]["ACLR"]), np.array(hist["online"]["ACLR"])
    fe, oe = np.array(hist["frozen"]["EVM"]), np.array(hist["online"]["EVM"])
    print(f"\n  online vs frozen @ final block: ACLR {oa[-1]-fa[-1]:+.1f} dB, "
          f"EVM {oe[-1]-fe[-1]:+.1f} dB (negative = online better)")

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(N_BLOCKS)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for k, c in [("frozen", "tab:red"), ("periodic", "tab:orange"), ("online", "tab:green")]:
            ax[0].plot(t, hist[k]["ACLR"], marker=".", color=c, label=k)
            ax[1].plot(t, hist[k]["EVM"], marker=".", color=c, label=k)
        for a, ttl in zip(ax, ["ACLR (dB)", "EVM (dB)"]):
            a.axvline(STEP_BLK, ls="--", c="gray", lw=1, label="bias step")
            a.set_xlabel("time block (increasing PA drift)"); a.set_ylabel(ttl)
            a.grid(alpha=0.3); a.legend(fontsize=8)
        ax[0].set_title("PA drift tracking: frozen vs periodic vs online-ILA")
        fig.tight_layout()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig.savefig(os.path.join(RESULTS_DIR, "drift_tracking.png"), dpi=130)
        print("\n  saved plot -> results/drift_tracking.png")
    except Exception as e:
        print(f"\n  (plot skipped: {e})")


if __name__ == "__main__":
    main()
