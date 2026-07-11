#!/usr/bin/env python3
"""
fixedpoint_drift.py -- does adapting Abar earn its eligibility-trace channels?

The decisive experiment.  Sweeps {memoryless, pole, zero} x {frozen, float, W8, W12}
and answers three coupled questions at once:

  * keep or drop the ONLINE UPDATE of Abar (the S4D poles),
  * settle `w_trace` (8 = Abar frozen by trace underflow; 12 = Abar alive),
  * decide whether the 32 `elig_trace_ch` channels are worth their area.

Why the drift KIND is the whole experiment
------------------------------------------
Abar spans the S4D poles, C_eff spans the zeros (drift_models.py derives this).
Inverting a PA that grew a pole needs a zero -> C_eff alone can do it.  Inverting a PA
that grew a ZERO (a notch) needs a pole -> only Abar can.  The historical `memoryless`
drift moves neither, so it was a rigged question whose answer could only be "Abar is
worthless".  `zero` is the arm that can actually see Abar; `pole` is the controlled
contrast at identical |H| tilt.

W8 vs W12 IS the Abar on/off switch: at W8 the scaled trace recursion rounds every
injection back to zero (max|p.sc| = 0.152 < LSB/2 = 0.25), so gAbar == 0 identically.

Three protocol invariants -- all learned by getting them wrong
-------------------------------------------------------------------------------
1.  The PA proxy and its offline DPD are ONE choice (`S4D_PA` env var).
2.  `lr` does not transfer across PA proxies.  Defaults below are the per-pair NMSE
    optima measured by wordlength_sweep.py (dgru: 0.2).
3.  Gradient block exponents are PER CLASS.  A shared exponent is calibrated by the
    largest class, and it destroys exactly the gAbar-vs-gCe separation measured here.
4.  `lr` does not transfer across ADAPTATION SETS either.  The sweep's per-pair optima
    were measured adapting {Abar, C_eff} only; this runner also adapts the stateless
    head, so the effective step is larger and the sweep's lr wrecks an already-converged
    DPD on the UNDRIFTED PA (float ACLR -50.79 -> -24.06).  `calibrate_lr` therefore picks
    the largest lr that leaves the offline optimum intact -- an adaptive loop must be a
    no-op at the optimum before it is allowed to track anything.  (The quantized arms hid
    this: make_q CLAMPS outlier gradients, so W8/W12 survived an lr that float did not.)

Metric warning: a pure `pole` drift is low-pass, so it SUPPRESSES out-of-band regrowth
and ACLR gets BETTER while EVM collapses.  The linear-memory arms (`pole`, `zero`) are
judged on NMSE/EVM only, never ACLR.  `drift_models.py` documents this.

Run:   uv run experiments/fixedpoint_drift.py
Smoke: S4D_FPDRIFT_SMOKE=1 uv run experiments/fixedpoint_drift.py
Env:   S4D_FPDRIFT_KINDS=memoryless,pole,zero  S4D_FPDRIFT_LR=<float>
"""
import os
import json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "results")

from online_adaptation import (make_pa, make_dpd, target_gain_of, segment,
                           DATASET, FS, BW, NSUB, NPERSEG, PA_TYPE)
from modules.data_collector import load_dataset
from utils.metrics import NMSE, EVM, ACLR
from rtrl_rflo_full import extract, stream, rflo_grads, CD
from fixedpoint_sweep import make_q, RAD, ADAPT_KEYS
import drift_models

SMOKE = os.environ.get("S4D_FPDRIFT_SMOKE", "") == "1"

N_BLOCKS = 3 if SMOKE else 12
STEP_BLK = 1 if SMOKE else 6
FL = 500
N_FRAMES = 6 if SMOKE else 80
ADAPT_LEN = 8000
W_TRACES = (8, 12)

LR_GRID = (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 1e-3, 5e-4)   # descending
LR_DAMAGE_TOL = 0.5      # dB of NMSE the loop may cost on the UNDRIFTED PA

# Abar-specific step multiplier.  |gAbar| ~ 1.7e-4 while |gCe| ~ 1.9e-3, so at a shared
# lr the poles crawl ~16x slower than the residues: over the whole run Abar can travel at
# most |dAbar| <= lr*|gAbar|*n_updates = 0.017, with every step perfectly aligned.  The
# offline poles occupy [0.794, 0.999] and the `zero` drift asks for a pole at radius 0.5,
# i.e. a move of ~0.29.  At LR_AB=1 the decisive arm therefore cannot test Abar at all --
# it reports "Abar is worthless" because Abar is pinned, not because it is useless.
# A per-class lr is nearly free in hardware (the SGD update already carries a per-class
# shift), so this is a legitimate knob -- unlike RMSprop-lite, which divides by a quantized
# RMS.  Set >1 to give the poles authority commensurate with their gradient scale.
LR_AB = float(os.environ.get("S4D_FPDRIFT_LR_AB", "1.0"))
LR_RMS = float(os.environ.get("S4D_FPDRIFT_LR_RMS", "0.02"))   # RMSprop has its own step size
# 'f_rms' / 'w8_rms' = RMSprop-lite on float / on 8-bit gradients.  w8_rms is the arm that
# shows why you cannot divide by the RMS of a quantized gradient.
ARMS = tuple(os.environ.get("S4D_FPDRIFT_ARMS", "frozen,float,w8,w12").split(","))

KINDS = tuple(os.environ.get("S4D_FPDRIFT_KINDS",
                             "memoryless,zero" if SMOKE else "memoryless,pole,zero").split(","))

# ACLR is not a valid discriminator for the linear-memory arms (see drift_models.py)
JUDGE = {k: ("NMSE", "EVM") if k in drift_models.MEMORY_KINDS else ("ACLR", "EVM")
         for k in drift_models.KINDS}


def eval_drift(P, prm, pa_fn, X_np, tg):
    """ACLR/EVM/NMSE of the cascade (DPD -> drifted PA) on one spec-length segment."""
    Xs = segment(X_np)[:1]
    z = torch.tensor(Xs[..., 0] + 1j * Xs[..., 1], dtype=CD)
    out = stream(P, *prm, z)
    xpd = torch.stack([out.real, out.imag], -1).to(torch.float32)
    yb = pa_fn(xpd).numpy().astype(np.float64)
    gt = tg * Xs.astype(np.float64)
    return dict(
        ACLR=float(np.mean(ACLR(yb, fs=FS, nperseg=NPERSEG, bw_main_ch=BW, n_sub_ch=NSUB))),
        EVM=float(EVM(yb, gt, sample_rate=FS, bw_main_ch=BW, n_sub_ch=NSUB, nperseg=NPERSEG)),
        NMSE=float(NMSE(yb, gt)))


def probe_per_class(P, prm, z, tgt, fl=FL, nfr=4):
    """Dynamic range of every quantized class, gradients PER CLASS."""
    mx = dict(state=0.0, trace=0.0, lam=0.0,
              par=float(max(prm[0].abs().max(), prm[2].abs().max(),
                            prm[1].abs().max(), prm[3].abs().max())))
    mx.update({f"grad_{k}": 0.0 for k in ADAPT_KEYS})

    def watch(key):
        def f(x):
            mx[key] = max(mx[key], float(x.abs().max()))
            return x
        return f

    T = z.shape[1]
    for s in range(0, min(nfr * fl, T - fl), fl):
        g, _ = rflo_grads(P, *prm, z[:, s:s + fl], tgt[:, s:s + fl],
                          head_grads=True, in_grads=True,
                          qf_state=watch('state'), qf_trace=watch('trace'),
                          qf_lam=watch('lam'))
        for k in ADAPT_KEYS:
            v = float(torch.complex(g[k + 'r'], g[k + 'i']).abs().max())
            mx[f"grad_{k}"] = max(mx[f"grad_{k}"], v)
    return mx


def make_qfs(w, mx, prm0):
    """Split fixed point: state/params @W16, trace @w, lambda/grad @W8 per-class."""
    qs16, _ = make_q(16, mx['state'])
    qp16, _ = make_q(16, mx['par'])
    ql8, _ = make_q(8, mx['lam'])

    # one exponent per gradient class, shared by its Re/Im halves
    qg = {}
    for k in ADAPT_KEYS:
        q, _ = make_q(8, mx[f"grad_{k}"])
        qg[k + 'r'] = q
        qg[k + 'i'] = q

    def qtrace(Ab):
        sc = (1.0 - Ab.abs()).clamp_min(1e-4)
        q, _ = make_q(w, mx['state'])
        return lambda p: q(p * sc) / sc

    qfs = dict(qf_state=qs16, qf_trace=(qtrace(prm0[0]), qtrace(prm0[2])),
               qf_lam=ql8, qf_grad=qg)
    return qfs, qp16


HEAD_KEYS = ('Wo', 'bo', 'Ws', 'bs', 'Win', 'bin')   # adapted head (pointwise stack stays frozen)


def adapt_block(P, prm, pa_fn, x_seg, rng, n_frames, lr, qfs=None, qpar=None,
                opt_state=None, lr_rms=LR_RMS):
    """One block of online ILA RFLO updates against the CURRENT drifted PA.
    Adaptation set = {Abar, C_eff} (traces) + {W_in,b_in} (trace) + head {Wo,bo,Ws,bs}
    (stateless -> instantaneous grads).  Pure SGD + pole projection.

    If `opt_state` is given, take an RMSprop-lite step instead (one RMS register per
    parameter -- cheap in hardware) and IGNORE lr.  Combined with `qfs`, this is the arm
    that shows why the low-precision learning path must contain NO DIVISION: the divisor
    is then the RMS of a QUANTIZED gradient, and small quantized gradients round toward
    zero, so the step blows up.  That is a statement about 8-bit gradients, so it can only
    be tested HERE -- the float-only ablation runner (drift_ablation.py::rms_b4) tests
    only the weaker "float RMSprop is pathological" half.
    """
    A1, C1, A2, C2 = prm
    proj = lambda A: torch.where(A.abs() > RAD, A / A.abs() * RAD, A)
    zc = torch.tensor(x_seg[:, 0] + 1j * x_seg[:, 1], dtype=CD)[None]
    out = stream(P, A1, C1, A2, C2, zc)
    xpd = torch.stack([out.real, out.imag], -1).to(torch.float32)
    zpa = pa_fn(xpd)                                  # observe drifted PA, forward only
    z_in = torch.complex(zpa[0, :, 0], zpa[0, :, 1]).to(CD)[None]
    tgt = out.detach()                                # ILA post-inverse target
    qfs = qfs or {}

    def step(val, gr, gi, key, scale=1.0):
        gc = torch.complex(gr, gi)
        if opt_state is None:
            return val - lr * scale * gc
        v = 0.99 * opt_state.get(key, torch.zeros_like(gr)) + 0.01 * (gr * gr + gi * gi)
        opt_state[key] = v
        return val - lr_rms * scale * gc / (v.sqrt() + 1e-8)

    for s in rng.integers(0, z_in.shape[1] - FL, size=n_frames):
        s = int(s)
        g, _ = rflo_grads(P, A1, C1, A2, C2, z_in[:, s:s + FL], tgt[:, s:s + FL],
                          head_grads=True, in_grads=True, **qfs)
        A1 = step(A1, g['Ab1r'], g['Ab1i'], 'Ab1', LR_AB)
        C1 = step(C1, g['Ce1r'], g['Ce1i'], 'Ce1')
        A2 = step(A2, g['Ab2r'], g['Ab2i'], 'Ab2', LR_AB)
        C2 = step(C2, g['Ce2r'], g['Ce2i'], 'Ce2')
        A1, A2 = proj(A1), proj(A2)
        for k in HEAD_KEYS:
            P[k] = step(P[k], g[k + 'r'], g[k + 'i'], k)
            if qpar is not None:
                P[k] = qpar(P[k])
        if qpar is not None:
            A1, C1 = qpar(A1), qpar(C1)
            A2, C2 = qpar(A2), qpar(C2)
    return (A1, C1, A2, C2)


def calibrate_lr(P0, prm0, pa, x_seg, X_te, tg, m_offline):
    """Largest lr that does not damage the offline optimum on the UNDRIFTED PA.

    Runs one float block of adaptation with the drift operator at blk=0 (the identity
    perturbation for every kind), starting from the converged offline DPD.  A correct
    online loop is a fixed point there; an lr that degrades NMSE by more than
    LR_DAMAGE_TOL is too large no matter how well it appears to 'track' later.
    Among the safe lrs we take the largest, since adaptation bandwidth must still keep
    up with the drift rate."""
    identity = drift_models.make_drift(pa, 0, N_BLOCKS, STEP_BLK, kind="memoryless")
    print(f"\ncalibrating lr (offline NMSE {m_offline['NMSE']:.2f} dB, "
          f"tolerance {LR_DAMAGE_TOL} dB):")
    chosen = None
    for lr in LR_GRID:
        Pc = dict(P0)                              # adapt_block rebinds into its own dict
        prm = adapt_block(Pc, tuple(t.clone() for t in prm0), identity, x_seg,
                          np.random.default_rng(0), N_FRAMES, lr)
        m = eval_drift(Pc, prm, identity, X_te, tg)
        dmg = m['NMSE'] - m_offline['NMSE']
        ok = dmg <= LR_DAMAGE_TOL
        print(f"  lr={lr:<7g} NMSE {m['NMSE']:7.2f}  ACLR {m['ACLR']:7.2f}  "
              f"damage {dmg:+6.2f} dB  {'OK' if ok else 'too large'}")
        if ok and chosen is None:
            chosen = lr
    if chosen is None:
        raise RuntimeError(f"no lr in {LR_GRID} leaves the offline optimum intact")
    print(f"  -> lr = {chosen}")
    return chosen


def gab_is_live(P, prm, z, tgt, qfs):
    """Sanity: is the Abar gradient actually nonzero under this trace wordlength?
    W8 must give exactly 0 (trace underflow); W12 must not.  If this assertion ever
    flips, the arms no longer mean what the table says they mean."""
    g, _ = rflo_grads(P, *prm, z[:, :FL], tgt[:, :FL], head_grads=True, in_grads=True, **qfs)
    return float(torch.complex(g['Ab1r'], g['Ab1i']).abs().max()), \
           float(torch.complex(g['Ab2r'], g['Ab2i']).abs().max())


def main():
    torch.manual_seed(0)
    print(f"=== Abar adaptation vs drift kind ===")
    print(f"PA={PA_TYPE}  blocks={N_BLOCKS} (step {STEP_BLK})  "
          f"frames/blk={N_FRAMES}  kinds={KINDS}{'  [SMOKE]' if SMOKE else ''}")

    dpd = make_dpd(load_offline=True).cpu().eval()
    pa = make_pa().cpu().eval()
    P0 = extract(dpd)
    prm0 = (P0['Ab1'].clone(), P0['Ce1'].clone(), P0['Ab2'].clone(), P0['Ce2'].clone())

    X_tr, y_tr, *_r, X_te, y_te = load_dataset(dataset_name=DATASET)
    X_tr = np.asarray(X_tr, np.float32); X_te = np.asarray(X_te, np.float32)
    tg = target_gain_of(X_tr, np.asarray(y_tr, np.float32))
    x_seg = X_tr[:ADAPT_LEN]

    # probe on the UNDRIFTED PA (blk=0 is the identity perturbation for every kind)
    zc = torch.tensor(x_seg[:, 0] + 1j * x_seg[:, 1], dtype=CD)[None]
    out0 = stream(P0, *prm0, zc)
    xpd0 = torch.stack([out0.real, out0.imag], -1).to(torch.float32)
    with torch.no_grad():
        zpa0 = pa(xpd0)
    z_probe = torch.complex(zpa0[0, :, 0], zpa0[0, :, 1]).to(CD)[None]
    mx = probe_per_class(P0, prm0, z_probe, out0.detach())
    print("\nprobe (per-class gradient maxima):")
    print("  " + "  ".join(f"{k}={mx['grad_' + k]:.2e}" for k in ADAPT_KEYS))

    qfs_by_w, qpar_by_w = {}, {}
    for w in W_TRACES:
        qfs_by_w[w], qpar_by_w[w] = make_qfs(w, mx, prm0)
        a1, a2 = gab_is_live(P0, prm0, z_probe, out0.detach(), qfs_by_w[w])
        live = "ALIVE" if max(a1, a2) > 0 else "dead (trace underflow)"
        print(f"  W{w}: |gAb1|max={a1:.3e}  |gAb2|max={a2:.3e}  -> Abar {live}")
    assert max(gab_is_live(P0, prm0, z_probe, out0.detach(), qfs_by_w[8])) == 0.0, \
        "W8 must freeze Abar by trace underflow; the W8-vs-W12 contrast is meaningless otherwise"

    # the offline optimum on the undrifted PA -- the fixed point the loop must respect
    identity = drift_models.make_drift(pa, 0, N_BLOCKS, STEP_BLK, kind="memoryless")
    m_off = eval_drift(P0, prm0, identity, X_te, tg)
    print(f"\noffline (undrifted): ACLR {m_off['ACLR']:.2f}  EVM {m_off['EVM']:.2f}  "
          f"NMSE {m_off['NMSE']:.2f}")

    env_lr = os.environ.get("S4D_FPDRIFT_LR")
    lr = float(env_lr) if env_lr else calibrate_lr(P0, prm0, pa, x_seg, X_te, tg, m_off)

    all_hist = {}
    for kind in KINDS:
        print(f"\n--- drift kind: {kind}   (judged on {'/'.join(JUDGE[kind])}) ---")
        pool = {'frozen': dict(n=0, qfs=None, qpar=None, rms=False),
                'float':  dict(n=N_FRAMES, qfs=None, qpar=None, rms=False),
                # RMSprop-lite on float vs on 8-bit gradients
                'f_rms':  dict(n=N_FRAMES, qfs=None, qpar=None, rms=True)}
        for w in W_TRACES:
            pool[f'w{w}'] = dict(n=N_FRAMES, qfs=qfs_by_w[w], qpar=qpar_by_w[w], rms=False)
        pool['w8_rms'] = dict(n=N_FRAMES, qfs=qfs_by_w[8], qpar=qpar_by_w[8], rms=True)
        arms = {k: pool[k] for k in ARMS}
        for a in arms.values():
            a['P'] = dict(P0)
            a['prm'] = tuple(t.clone() for t in prm0)
            a['opt'] = {} if a['rms'] else None      # RMS registers persist across blocks
        rngs = {k: np.random.default_rng(0) for k in arms}
        hist = {k: dict(ACLR=[], EVM=[], NMSE=[], dAb=[]) for k in arms}

        print(f"blk | " + " | ".join(f"{k + ' (' + '/'.join(JUDGE[kind]) + '/dAb)':>24}"
                                    for k in arms))
        for blk in range(N_BLOCKS):
            pa_fn = drift_models.make_drift(pa, blk, N_BLOCKS, STEP_BLK, kind=kind)
            for k, a in arms.items():
                if a['n']:
                    a['prm'] = adapt_block(a['P'], a['prm'], pa_fn, x_seg, rngs[k],
                                           a['n'], lr, qfs=a['qfs'], qpar=a['qpar'],
                                           opt_state=a['opt'])
            cells = []
            for k, a in arms.items():
                m = eval_drift(a['P'], a['prm'], pa_fn, X_te, tg)
                for f in ('ACLR', 'EVM', 'NMSE'):
                    hist[k][f].append(m[f])
                # how far have the poles actually travelled?  If this stays ~0 the arm
                # never tested Abar, whatever its metrics say.
                dab = max(float((a['prm'][0] - prm0[0]).abs().max()),
                          float((a['prm'][2] - prm0[2]).abs().max()))
                hist[k]['dAb'].append(dab)
                j0, j1 = JUDGE[kind]
                cells.append(f"{m[j0]:>8.2f}/{m[j1]:>8.2f}/{dab:.4f}")
            tag = "  <-- step" if blk == STEP_BLK else ""
            print(f"{blk:3d} | " + " | ".join(cells) + tag)

        all_hist[kind] = hist
        j0, j1 = JUDGE[kind]
        print(f"\n  summary ({j0}/{j1}; mean / final / worst)")
        for k in arms:
            for f in (j0, j1):
                v = np.array(hist[k][f])
                print(f"    {k:>7} {f:>5}: {v.mean():7.2f} / {v[-1]:7.2f} / {v.max():7.2f}")
        for k in arms:
            print(f"    {k:>7}  final max|dAbar| = {hist[k]['dAb'][-1]:.4f}")
        # the decisive contrast: does the live-Abar arm beat the frozen-Abar arm?
        if 'w12' in arms and 'w8' in arms:
            for f in (j0, j1):
                d = hist['w12'][f][-1] - hist['w8'][f][-1]
                verdict = "W12 (Abar alive) better" if d < 0 else "W8 (Abar frozen) better"
                print(f"    >>> {f}: W12 - W8 = {d:+.2f} dB   -> {verdict}")
            if hist['w12']['dAb'][-1] < 0.02:
                print("    !!! max|dAbar| < 0.02: the poles barely moved.  This arm did NOT "
                      "test Abar's value -- raise S4D_FPDRIFT_LR_AB before drawing a conclusion.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # tag by arm set too: a non-default arm list must NOT overwrite fixedpoint_drift_<pa>.json, which
    # is the canonical evidence behind Fig. fixedpoint_drift.png
    DEFAULT_ARMS = ('frozen', 'float', 'w8', 'w12')
    tag = f"{PA_TYPE}{'_smoke' if SMOKE else ''}" + (f"_lrab{LR_AB:g}" if LR_AB != 1.0 else "") \
        + ("" if tuple(ARMS) == DEFAULT_ARMS else "_" + "-".join(ARMS))
    out = os.path.join(RESULTS_DIR, f"fixedpoint_drift_{tag}.json")
    with open(out, "w") as f:
        json.dump(dict(pa=PA_TYPE, lr=lr, lr_ab=LR_AB, arms=list(ARMS), n_blocks=N_BLOCKS,
                       step_blk=STEP_BLK, n_frames=N_FRAMES, offline=m_off,
                       hist=all_hist), f, indent=1)
    print(f"\nsaved {out}")
    if not SMOKE and tuple(ARMS) == DEFAULT_ARMS:     # only the canonical run owns the figure
        plot_memoryless(out)


def plot_memoryless(json_path):
    """Paper figure: 8-bit vs float drift tracking (memoryless arm) -> results/fixedpoint_drift.png.

    This figure is the ONLY home of the float-vs-fixed-point comparison, so it must be built
    from THIS experiment's calibrated data.  The figure it replaces was produced by the old
    LR=0.2 fixedpoint_drift.py and showed 8-bit and float on the *same trajectory* -- an artifact of
    an unsafe lr (make_q clamps the outlier gradients and so protected the 8-bit arm while the
    float arm was damaged).  At a calibrated lr the 8-bit path tracks, but trails float ~1 dB.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.load(open(json_path))
    h = d['hist'].get('memoryless')
    if not h:
        print("  (figure skipped: no memoryless arm in this run)")
        return
    style = {'frozen': ("tab:red", "frozen (no adaptation)"),
             'float':  ("tab:blue", "float SGD"),
             'w12':    ("tab:green", "fixed-point, w_trace=12"),
             'w8':     ("tab:orange", "fixed-point, w_trace=8")}
    t = np.arange(d['n_blocks'])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, (c, lab) in style.items():
        if k not in h:
            continue
        ax[0].plot(t, h[k]['ACLR'], marker=".", color=c, label=lab)
        ax[1].plot(t, h[k]['EVM'], marker=".", color=c, label=lab)
    for a, ttl in zip(ax, ["ACLR (dB)", "EVM (dB)"]):
        a.axhline(d['offline'][ttl.split()[0]], ls=":", c="gray", lw=1)
        a.axvline(d['step_blk'], ls="--", c="gray", lw=1)
        a.set_xlabel("drift block"); a.set_ylabel(ttl)
        a.grid(alpha=0.3); a.legend(fontsize=8)
    pa_name = d.get('pa', PA_TYPE)
    ax[0].set_title(f"8-bit learning path vs float under PA drift "
                    f"(PA={pa_name}, lr={d['lr']:g})")
    ax[1].set_title("dotted = undrifted offline optimum; dashed = bias step", fontsize=8)
    fig.tight_layout()
    # name the figure by PA, so that running this for another proxy cannot silently
    # overwrite the figure a previous run produced.
    png = os.path.join(RESULTS_DIR, f"fixedpoint_drift_{pa_name}.png")
    fig.savefig(png, dpi=130)
    print(f"  saved results/fixedpoint_drift_{pa_name}.png")


if __name__ == "__main__":
    if os.environ.get("S4D_FPDRIFT_PLOT_ONLY"):     # rebuild the figure from committed evidence
        plot_memoryless(os.path.join(RESULTS_DIR, f"fixedpoint_drift_{PA_TYPE}.json"))
    else:
        main()
