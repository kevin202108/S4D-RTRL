#!/usr/bin/env python3
"""
drift_tracking.py -- Stage 2: PA drift tracking (the selling point).

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

Run:  uv run drift_tracking.py   (saves drift_tracking.png)
"""
import copy
import numpy as np
import torch

from online_adaptation import (make_pa, make_dpd, target_gain_of, segment,
                           DATASET, DEVICE, FS, BW, NSUB, NPERSEG)
from modules.data_collector import load_dataset
from utils.metrics import NMSE, EVM, ACLR

N_BLOCKS = 30
STEP_BLK = 15                    # bias step happens here
RETRAIN_EVERY = 6                # periodic host recal cadence (blocks)
rng = np.random.default_rng(0)


def drift_params(blk):
    """Return (A, phi, b3, ampm): small LINEAR gain/phase drift + growing
    NONLINEAR 3rd-order AM-AM (b3) and AM-PM (ampm) -- the nonlinear part is what
    creates spectral regrowth (ACLR) that the DPD must chase."""
    dl = blk / (N_BLOCKS - 1)            # 0 -> 1 ramp
    A = 1.0 - 0.05 * dl                  # up to 5% gain droop (linear, hits EVM)
    phi = 0.10 * dl                      # up to 0.10 rad phase drift (linear)
    b3 = 0.14 * dl                       # 3rd-order AM-AM growth (nonlinear -> ACLR)
    ampm = 0.18 * dl                     # 3rd-order AM-PM growth (nonlinear -> ACLR)
    if blk >= STEP_BLK:                  # mid-run bias step = nonlinearity jump
        A *= 0.97; phi += 0.05
        b3 += 0.05; ampm += 0.07
    return A, phi, b3, ampm


def make_pa_fn(pa, blk):
    """Forward-only drifted PA operator: (B,T,2) -> (B,T,2).
    drifted = g_lin * ( y + (b3 + j*ampm) * y|y|^2 )  -- complex 3rd-order term
    gives both AM-AM and AM-PM regrowth; |y|^2 normalized by nominal mean power."""
    A, phi, b3, ampm = drift_params(blk)
    g = complex(A * np.cos(phi), A * np.sin(phi))
    c3 = complex(b3, ampm)

    def pa_fn(xpd):
        with torch.no_grad():
            y = pa(xpd)
        yc = y[..., 0] + 1j * y[..., 1]
        p = (yc.abs() ** 2)
        p = p / (p.mean() + 1e-12)                    # normalize power -> stable coeff scale
        yc = g * (yc + c3 * yc * p)
        return torch.stack([yc.real, yc.imag], dim=-1)
    return pa_fn


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
    print(f"device = {DEVICE}  |  Stage 2: PA drift tracking")

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
        fig.tight_layout(); fig.savefig("drift_tracking.png", dpi=130)
        print("\n  saved plot -> drift_tracking.png")
    except Exception as e:
        print(f"\n  (plot skipped: {e})")


if __name__ == "__main__":
    main()
