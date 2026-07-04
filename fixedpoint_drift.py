#!/usr/bin/env python3
"""
fixedpoint_drift.py -- Stage 3 continuation: does the FIXED-POINT (split-W8) RFLO
learning path still TRACK PA DRIFT?  (= stage2's selling-point scenario replayed
on the hardware-faithful learning loop.)

Four arms, all starting from the same offline s4d_best, adaptation set {Abar,C_eff}
only (design C2), pure SGD + pole projection (C1/H1), ILA post-inverse (no PA grad):
    frozen    : never updates (shows the damage)
    float     : RFLO in float64 (the S1r-verified loop)
    w8        : split fixed point -- state/params @W16, traces(scaled)/lambda/grad @W8
    w8_decim  : same as w8 but adapts on 1/4 of the frames (K-decimation:
                adaptation-power knob; prices P_ADA reduction)
Drift model = drift_tracking.py verbatim (linear gain/phase ramp + growing 3rd-order
AM-AM/AM-PM + bias step), compressed to 12 blocks (step at 6).

Run:  uv run fixedpoint_drift.py     (CPU; ~1.5 h; saves fixedpoint_drift.png)
"""
import numpy as np
import torch

from online_adaptation import (make_pa, make_dpd, target_gain_of, segment,
                           DATASET, FS, BW, NSUB, NPERSEG)
from modules.data_collector import load_dataset
from utils.metrics import EVM, ACLR
from rtrl_rflo_full import extract, stream, rflo_grads, CD
from fixedpoint_sweep import make_q, RAD

N_BLOCKS = 12
STEP_BLK = 6
FL, N_FRAMES, LR = 500, 16, 0.2
ADAPT_LEN = 8000                                  # X_tr samples per block for ILA


def drift_params(blk):                            # drift_tracking.py, recompressed
    dl = blk / (N_BLOCKS - 1)
    A = 1.0 - 0.05 * dl
    phi = 0.10 * dl
    b3 = 0.14 * dl
    ampm = 0.18 * dl
    if blk >= STEP_BLK:
        A *= 0.97; phi += 0.05; b3 += 0.05; ampm += 0.07
    return A, phi, b3, ampm


def make_pa_fn(pa, blk):
    A, phi, b3, ampm = drift_params(blk)
    g = complex(A * np.cos(phi), A * np.sin(phi))
    c3 = complex(b3, ampm)

    def pa_fn(xpd):
        with torch.no_grad():
            y = pa(xpd)
        yc = y[..., 0] + 1j * y[..., 1]
        p = yc.abs() ** 2
        p = p / (p.mean() + 1e-12)
        yc = g * (yc + c3 * yc * p)
        return torch.stack([yc.real, yc.imag], dim=-1)
    return pa_fn


def eval_drift_ported(P, prm, pa_fn, X_np, tg):
    Xs = segment(X_np)[:1]                        # one spec-length segment
    z = torch.tensor(Xs[..., 0] + 1j * Xs[..., 1], dtype=CD)
    out = stream(P, *prm, z)
    xpd = torch.stack([out.real, out.imag], -1).to(torch.float32)
    yb = pa_fn(xpd).numpy().astype(np.float64)
    gt = tg * Xs.astype(np.float64)
    return dict(ACLR=float(np.mean(ACLR(yb, fs=FS, nperseg=NPERSEG,
                                        bw_main_ch=BW, n_sub_ch=NSUB))),
                EVM=float(EVM(yb, gt, sample_rate=FS, bw_main_ch=BW,
                              n_sub_ch=NSUB, nperseg=NPERSEG)))


PW_KEYS = ('Wm1', 'bm1', 'Wf1_1', 'bf1_1', 'Wf2_1', 'bf2_1',
           'Wm2', 'bm2', 'Wf1_2', 'bf1_2', 'Wf2_2', 'bf2_2')


def adapt_block(P, prm, pa_fn, x_seg, rng, n_frames, qfs=None, qpar=None,
                in_grads=False, pw_grads=False, opt_state=None, lr_rms=0.02):
    """One block of online ILA RFLO updates against the CURRENT drifted PA.
    Adaptation set = {Abar, C_eff} (traces) + output head Wo/bo/Ws/bs
    (STATELESS -> instantaneous exact grads; needed to track gain/phase drift,
    which the frozen linear passthrough would otherwise pin). P is the ARM'S OWN
    copy -- its head entries are updated in place."""
    A1, C1, A2, C2 = prm
    proj = lambda A: torch.where(A.abs() > RAD, A / A.abs() * RAD, A)
    zc = torch.tensor(x_seg[:, 0] + 1j * x_seg[:, 1], dtype=CD)[None]
    out = stream(P, A1, C1, A2, C2, zc)           # current predistorter output
    xpd = torch.stack([out.real, out.imag], -1).to(torch.float32)
    zpa = pa_fn(xpd)                              # observe drifted PA (forward only)
    z_in = torch.complex(zpa[0, :, 0], zpa[0, :, 1]).to(CD)[None]
    tgt = out.detach()
    qfs = qfs or {}
    def step(val, gr, gi, key):
        """B4 RMSprop-lite when opt_state given: per-param normalized step
        (one RMS register per parameter -- cheap hardware). Else pure SGD."""
        gc = torch.complex(gr, gi)
        if opt_state is None:
            return val - LR * gc
        v = opt_state.get(key, torch.zeros_like(gr))
        v = 0.99 * v + 0.01 * (gr * gr + gi * gi)
        opt_state[key] = v
        return val - lr_rms * gc / (v.sqrt() + 1e-8)

    for s in rng.integers(0, z_in.shape[1] - FL, size=n_frames):
        s = int(s)
        g, _ = rflo_grads(P, A1, C1, A2, C2, z_in[:, s:s + FL], tgt[:, s:s + FL],
                          head_grads=True, in_grads=in_grads, pw_grads=pw_grads,
                          **qfs)
        A1 = step(A1, g['Ab1r'], g['Ab1i'], 'Ab1')
        C1 = step(C1, g['Ce1r'], g['Ce1i'], 'Ce1')
        A2 = step(A2, g['Ab2r'], g['Ab2i'], 'Ab2')
        C2 = step(C2, g['Ce2r'], g['Ce2i'], 'Ce2')
        A1, A2 = proj(A1), proj(A2)
        hk = ('Wo', 'bo', 'Ws', 'bs') + (('Win', 'bin') if in_grads else ()) \
            + (PW_KEYS if pw_grads else ())
        for k in hk:
            P[k] = step(P[k], g[k + 'r'], g[k + 'i'], k)
            if qpar is not None:
                P[k] = qpar(P[k])
        if qpar is not None:
            A1, C1 = qpar(A1), qpar(C1); A2, C2 = qpar(A2), qpar(C2)
    return (A1, C1, A2, C2)


def main():
    torch.manual_seed(0)
    print("=== fixedpoint_drift: fixed-point RFLO under PA drift (12 blocks) ===")
    dpd = make_dpd(load_offline=True).cpu().eval()
    pa = make_pa().cpu().eval()
    P = extract(dpd)
    prm0 = (P['Ab1'].clone(), P['Ce1'].clone(), P['Ab2'].clone(), P['Ce2'].clone())

    X_tr, y_tr, *_r, X_te, y_te = load_dataset(dataset_name=DATASET)
    X_tr = np.asarray(X_tr, np.float32); X_te = np.asarray(X_te, np.float32)
    tg = target_gain_of(X_tr, np.asarray(y_tr, np.float32))
    x_seg = X_tr[:ADAPT_LEN]

    # fixed-point formats (ranges from the S3-1 probe; see fixedpoint_sweep.py)
    RANGES = dict(state=20.0, lam=2.4e-6, grad=1.5e-2, par=1.0)
    qs16, _ = make_q(16, RANGES['state']); qp16, _ = make_q(16, RANGES['par'])
    ql8, _ = make_q(8, RANGES['lam']);     qg8, _ = make_q(8, RANGES['grad'])

    def qtrace_scaled(w, Ab):
        sc = (1.0 - Ab.abs()).clamp_min(1e-4)
        q, _ = make_q(w, RANGES['state'])
        return lambda p: q(p * sc) / sc
    qt8 = (qtrace_scaled(8, prm0[0]), qtrace_scaled(8, prm0[2]))
    qfs_w8 = dict(qf_state=qs16, qf_trace=qt8, qf_lam=ql8, qf_grad=qg8)

    arms = {
        'frozen':   dict(prm=tuple(t.clone() for t in prm0), n=0,  qfs=None,   qpar=None,
                         ip=False, pw=False, rms=False),
        # reference from previous round: full traces set, SGD
        'f80_ip':   dict(prm=tuple(t.clone() for t in prm0), n=80, qfs=None,   qpar=None,
                         ip=True, pw=False, rms=False),
        # C5 relaxed: + pointwise mixing/FFN (stateless, instantaneous grads)
        'f80_full': dict(prm=tuple(t.clone() for t in prm0), n=80, qfs=None,   qpar=None,
                         ip=True, pw=True, rms=False),
        # B4: RMSprop-lite per-param step on top of the full set
        'f80_full_rms': dict(prm=tuple(t.clone() for t in prm0), n=80, qfs=None, qpar=None,
                             ip=True, pw=True, rms=True),
        # fixed-point version of the best config
        'w8_full_rms':  dict(prm=tuple(t.clone() for t in prm0), n=80, qfs=qfs_w8, qpar=qp16,
                             ip=True, pw=True, rms=True),
    }
    for a in arms.values():                       # per-arm P + optimizer state
        a['P'] = dict(P)
        a['opt'] = {} if a['rms'] else None
    rngs = {k: np.random.default_rng(0) for k in arms}
    hist = {k: dict(ACLR=[], EVM=[]) for k in arms}

    hdr = " | ".join(f"{k:>13}" for k in arms)
    print(f"\n{'blk':>3} {'A/phi/b3/ampm':>18} | {hdr}")
    for blk in range(N_BLOCKS):
        pa_fn = make_pa_fn(pa, blk)
        A, phi, b3, ampm = drift_params(blk)
        row = []
        for k, a in arms.items():
            m = eval_drift_ported(a['P'], a['prm'], pa_fn, X_te, tg)
            hist[k]['ACLR'].append(m['ACLR']); hist[k]['EVM'].append(m['EVM'])
            row.append(f"{m['ACLR']:6.1f} {m['EVM']:6.1f}")
        tag = "  <-- bias step" if blk == STEP_BLK else ""
        print(f"{blk:3d} {A:4.2f}/{phi:4.2f}/{b3:4.2f}/{ampm:4.2f} | "
              + " | ".join(row) + tag)
        for k, a in arms.items():
            if a['n'] > 0:
                a['prm'] = adapt_block(a['P'], a['prm'], pa_fn, x_seg, rngs[k],
                                       a['n'], qfs=a['qfs'], qpar=a['qpar'],
                                       in_grads=a['ip'], pw_grads=a['pw'],
                                       opt_state=a['opt'])

    print("\n            |  ACLR (mean/final/worst) |  EVM (mean/final/worst)")
    for k in arms:
        a = np.array(hist[k]['ACLR']); e = np.array(hist[k]['EVM'])
        print(f"  {k:9s} | {a.mean():6.1f}/{a[-1]:6.1f}/{a.max():6.1f}   "
              f"| {e.mean():6.1f}/{e[-1]:6.1f}/{e.max():6.1f}")
    dA = hist['w8_full_rms']['ACLR'][-1] - hist['f80_full_rms']['ACLR'][-1]
    dE = hist['w8_full_rms']['EVM'][-1] - hist['f80_full_rms']['EVM'][-1]
    fro = hist['w8_full_rms']['ACLR'][-1] - hist['frozen']['ACLR'][-1]
    pwg = hist['f80_full']['ACLR'][-1] - hist['f80_ip']['ACLR'][-1]
    rmg = hist['f80_full_rms']['ACLR'][-1] - hist['f80_full']['ACLR'][-1]
    print(f"\n  w8 vs float (full+rms) @ final: ACLR {dA:+.2f} dB, EVM {dE:+.2f} dB")
    print(f"  w8_full_rms vs frozen @ final: ACLR {fro:+.1f} dB")
    print(f"  C5 pointwise gain: {pwg:+.1f} dB;  B4 rms gain: {rmg:+.1f} dB")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(N_BLOCKS)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        cyc = ["tab:red", "tab:green", "tab:olive", "tab:blue", "tab:purple",
               "tab:brown", "tab:cyan"]
        colors = {k: cyc[i % len(cyc)] for i, k in enumerate(arms)}
        for k in arms:
            ax[0].plot(t, hist[k]['ACLR'], marker=".", color=colors[k], label=k)
            ax[1].plot(t, hist[k]['EVM'], marker=".", color=colors[k], label=k)
        for a, ttl in zip(ax, ["ACLR (dB)", "EVM (dB)"]):
            a.axvline(STEP_BLK, ls="--", c="gray", lw=1)
            a.set_xlabel("drift block"); a.set_ylabel(ttl)
            a.grid(alpha=0.3); a.legend(fontsize=8)
        ax[0].set_title("Fixed-point RFLO ({Abar,C_eff}-only) drift tracking")
        fig.tight_layout(); fig.savefig("fixedpoint_drift.png", dpi=130)
        print("  saved plot -> fixedpoint_drift.png")
    except Exception as e:
        print(f"  (plot skipped: {e})")

    ok = (abs(dA) < 1.0) and (hist['w8_full_rms']['ACLR'][-1] < hist['frozen']['ACLR'][-1] - 3.0)
    print("\n" + ("OK: split-W8 fixed-point RFLO tracks drift like float."
                  if ok else "PARTIAL: see numbers above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
