#!/usr/bin/env python3
"""
drift_ablation.py -- ADAPTATION-SET ablation under PA drift, at a CALIBRATED lr.

Question: of the parameters we *could* adapt online, which ones actually earn their
hardware (their eligibility traces / gradient logic)?  Fix everything else -- float SGD,
pole projection, the `memoryless` drift schedule (12 blocks, bias step at block 6, the
historical stage-2 selling-point scenario) -- and vary ONLY the adaptation set:

    frozen    : none (shows the damage)
    rec_only  : {Abar, C_eff}                          -- traces only, no head
    rec_head  : {Abar, C_eff, Wo, bo, Ws, bs}          -- + stateless output head
    proposed  : {Abar, C_eff, Win, bin, Wo, bo, Ws, bs}-- + input-proj trace  (the design)
    full_ffn  : proposed + pointwise mixing/FFN         -- the pointwise stack unfrozen

Trace-wordlength / Abar-on-off is a DIFFERENT axis and lives in fixedpoint_drift.py
(fix the adaptation set = `proposed`, vary {frozen,float,W8,W12}); the two scripts share
`drift_models.make_drift(pa, blk, 12, 6, kind="memoryless")` bit-for-bit, so their `frozen`
and `proposed`/`float` arms are directly comparable.

Why calibrated lr (this file used to hardcode LR=0.2 -- the numbers it produced are RETRACTED)
--------------------------------------------------------------------------------------------
The adaptation set includes the stateless head, so the effective step is larger than the
per-pair sweep optimum measured on {Abar,C_eff} alone.  At LR=0.2 the loop DAMAGES the
converged offline DPD on the UNDRIFTED PA (block-0 NMSE -42.78 -> ~-20), and -- worse for an
ablation -- it damages the un-quantized float arm MORE than a make_q-clamped fixed-point arm,
which is how the old table reported "8-bit == float, same trajectory".  That parity was an
artifact of an unsafe lr, not a property of the 8-bit path.  `calibrate_lr` picks the largest
lr whose NMSE damage on the undrifted PA is <= LR_DAMAGE_TOL: an online loop must be a
fixed point at the optimum before it is allowed to track anything.

lr IS PER ARM (do not share one lr across the ablation).  It is scoped to the adaptation set:
a larger set is less stable, so a borrowed lr DAMAGES it (sharing proposed's lr=0.10 put
full_ffn 2.4 dB below the offline optimum at block 0); a smaller set tolerates a larger lr, so
a borrowed lr UNDERSTATES it -- and understating the rival arms would flatter our own design.
Each arm therefore gets its own knee (lr_sweep.py, S4D_SWEEP_ARM), and block 0 (the
identity perturbation) is asserted as a fixed point for every arm before any arm is compared.

Metric warning: `memoryless` is judged on ACLR/EVM.  For pole/zero drifts (not run here) use
NMSE/EVM -- a pole drift is low-pass and *improves* ACLR while EVM collapses (drift_models.py).

Run:   uv run experiments/drift_ablation.py   (CPU; ~35 min; saves results/drift_ablation_<pa>.png)
Smoke: S4D_FPDRIFT_SMOKE=1 uv run experiments/drift_ablation.py
Env:   S4D_DRIFT_ARMS=frozen,rec_only,...
       S4D_DRIFT_LR=<float>  |  S4D_DRIFT_LR=rec_only=0.3,rec_head=0.15,proposed=0.10,...
                                (arms omitted from the map are calibrated)
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
from fixedpoint_sweep import RAD
import drift_models

SMOKE = os.environ.get("S4D_FPDRIFT_SMOKE", "") == "1"

N_BLOCKS = 3 if SMOKE else 12
STEP_BLK = 1 if SMOKE else 6
FL = 500
N_FRAMES = 6 if SMOKE else 80
ADAPT_LEN = 8000
DRIFT_KIND = "memoryless"                        # the historical stage-2 schedule

LR_GRID = (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 1e-3, 5e-4)   # descending
LR_DAMAGE_TOL = 0.5      # dB of NMSE the loop may cost on the UNDRIFTED PA

HEAD_KEYS = ('Wo', 'bo', 'Ws', 'bs')
IN_KEYS = ('Win', 'bin')
PW_KEYS = ('Wm1', 'bm1', 'Wf1_1', 'bf1_1', 'Wf2_1', 'bf2_1',
           'Wm2', 'bm2', 'Wf1_2', 'bf1_2', 'Wf2_2', 'bf2_2')

# adaptation-set ablation: (adapt input-proj trace, adapt pointwise/FFN).  Head + recurrent
# are toggled explicitly per arm below.  rflo_grads always yields Abar/C_eff grads.
# `rms` = RMSprop-lite per-parameter normalized step (one RMS register per param -- cheap
# in hardware).  NOTE this runner is FLOAT-ONLY, so `rms_b4` tests only the weaker half of the
# case against it ("float RMSprop is pathological": it destroys the undrifted optimum in one
# block and never recovers).  The real finding is about 8-BIT gradients -- "you cannot divide
# by the RMS of a quantized gradient" -- and can ONLY be tested in the quantized runner:
#   S4D_FPDRIFT_ARMS=frozen,float,f_rms,w8_rms python experiments/fixedpoint_drift.py
# (w8_rms degenerates: EVM +76.5 dB, |dAbar| = 1.62 vs SGD's 0.0045.)
LR_RMS = 0.02            # RMSprop has its own step size; the SGD lr does not apply to it
ARM_SETS = {
    'frozen':   dict(n=0,        head=False, inp=False, pw=False, rms=False),
    'rec_only': dict(n=N_FRAMES, head=False, inp=False, pw=False, rms=False),
    'rec_head': dict(n=N_FRAMES, head=True,  inp=False, pw=False, rms=False),
    'proposed': dict(n=N_FRAMES, head=True,  inp=True,  pw=False, rms=False),
    'full_ffn': dict(n=N_FRAMES, head=True,  inp=True,  pw=True,  rms=False),
    'rms_b4':   dict(n=N_FRAMES, head=True,  inp=True,  pw=True,  rms=True),
}
ARMS = tuple(os.environ.get("S4D_DRIFT_ARMS",
             "frozen,rec_only,rec_head,proposed,full_ffn").split(","))


def eval_drift(P, prm, pa_fn, X_np, tg):
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


def adapt_block(P, prm, pa_fn, x_seg, rng, n_frames, lr, spec, opt_state=None):
    """One block of online ILA RFLO updates against the CURRENT drifted PA.  Pure SGD +
    pole projection.  `spec` toggles which parameters are in the adaptation set; the
    recurrent {Abar,C_eff} are always adapted (that is what a 'recurrent-only' arm means).
    If spec['rms'], take an RMSprop-lite step out of `opt_state` instead (lr is ignored)."""
    A1, C1, A2, C2 = prm
    proj = lambda A: torch.where(A.abs() > RAD, A / A.abs() * RAD, A)
    zc = torch.tensor(x_seg[:, 0] + 1j * x_seg[:, 1], dtype=CD)[None]
    out = stream(P, A1, C1, A2, C2, zc)
    xpd = torch.stack([out.real, out.imag], -1).to(torch.float32)
    zpa = pa_fn(xpd)                                  # observe drifted PA, forward only
    z_in = torch.complex(zpa[0, :, 0], zpa[0, :, 1]).to(CD)[None]
    tgt = out.detach()                                # ILA post-inverse target
    hk = (HEAD_KEYS if spec['head'] else ()) + (IN_KEYS if spec['inp'] else ()) \
        + (PW_KEYS if spec['pw'] else ())
    rms = spec.get('rms') and opt_state is not None

    def step(val, gr, gi, key):
        gc = torch.complex(gr, gi)
        if not rms:
            return val - lr * gc
        v = 0.99 * opt_state.get(key, torch.zeros_like(gr)) + 0.01 * (gr * gr + gi * gi)
        opt_state[key] = v
        return val - LR_RMS * gc / (v.sqrt() + 1e-8)   # <- the divide that 8-bit cannot do

    for s in rng.integers(0, z_in.shape[1] - FL, size=n_frames):
        s = int(s)
        g, _ = rflo_grads(P, A1, C1, A2, C2, z_in[:, s:s + FL], tgt[:, s:s + FL],
                          head_grads=spec['head'], in_grads=spec['inp'], pw_grads=spec['pw'])
        A1 = step(A1, g['Ab1r'], g['Ab1i'], 'Ab1')
        C1 = step(C1, g['Ce1r'], g['Ce1i'], 'Ce1')
        A2 = step(A2, g['Ab2r'], g['Ab2i'], 'Ab2')
        C2 = step(C2, g['Ce2r'], g['Ce2i'], 'Ce2')
        A1, A2 = proj(A1), proj(A2)
        for k in hk:
            P[k] = step(P[k], g[k + 'r'], g[k + 'i'], k)
    return (A1, C1, A2, C2)


def calibrate_lr(P0, prm0, pa, x_seg, X_te, tg, m_off, spec, name=""):
    """Largest lr whose damage to the offline optimum on the UNDRIFTED PA is <= LR_DAMAGE_TOL.

    PER ARM.  lr does NOT transfer between adaptation sets: a larger set is less stable, so an
    lr calibrated elsewhere can damage this arm, and a smaller set tolerates a larger lr, so a
    borrowed lr can UNDERSTATE it.  (The earlier version calibrated once on `proposed` and
    called it "the largest set, hence safe for every subset" -- wrong: `full_ffn` is `proposed`
    PLUS the pointwise keys, i.e. a SUPERset, and it duly came back damaged by 2.4 dB.)

    NOTE this is only the stability bound.  Tracking rises monotonically with lr up to it, so
    the bound is necessary, not sufficient -- the reported lr is the tracking-vs-damage knee
    from lr_sweep.py.
    """
    identity = drift_models.make_drift(pa, 0, N_BLOCKS, STEP_BLK, kind="memoryless")
    print(f"\ncalibrating lr for {name or 'arm'} (offline NMSE {m_off['NMSE']:.2f} dB, "
          f"tol {LR_DAMAGE_TOL} dB):")
    chosen = None
    for lr in LR_GRID:
        Pc = dict(P0)
        prm = adapt_block(Pc, tuple(t.clone() for t in prm0), identity, x_seg,
                          np.random.default_rng(0), N_FRAMES, lr, spec)
        m = eval_drift(Pc, prm, identity, X_te, tg)
        # judge on ALL THREE metrics: an lr can cost +0.6 dB ACLR while costing +4.4 dB EVM
        # (a real failure mode), and an ACLR- or NMSE-only gate waves it through
        dmgs = {k: m[k] - m_off[k] for k in ('ACLR', 'EVM', 'NMSE')}
        dmg = max(dmgs.values())
        ok = dmg <= LR_DAMAGE_TOL and not any(np.isnan(v) for v in m.values())
        print(f"  lr={lr:<7g} NMSE {m['NMSE']:7.2f}  ACLR {m['ACLR']:7.2f}  "
              f"damage A/E/N {dmgs['ACLR']:+.2f}/{dmgs['EVM']:+.2f}/{dmgs['NMSE']:+.2f} dB  "
              f"{'OK' if ok else 'too large (' + max(dmgs, key=dmgs.get) + ')'}")
        if ok and chosen is None:
            chosen = lr
    if chosen is None:
        raise RuntimeError(f"no lr in {LR_GRID} leaves the offline optimum intact")
    print(f"  -> lr = {chosen}")
    return chosen


def parse_lr_env(env_lr, arms):
    """S4D_DRIFT_LR is either a scalar (all arms) or a per-arm map 'rec_only=0.3,proposed=0.1'.
    Arms absent from the map are calibrated."""
    if '=' not in env_lr:
        return {k: float(env_lr) for k in arms}
    out = {}
    for part in env_lr.split(','):
        k, _, v = part.partition('=')
        k = k.strip()
        if k not in ARM_SETS:
            raise SystemExit(f"S4D_DRIFT_LR: unknown arm {k!r} (have {list(ARM_SETS)})")
        out[k] = float(v)
    return out


def main():
    torch.manual_seed(0)
    print("=== drift_ablation: adaptation-set ablation under PA drift (calibrated lr) ===")
    print(f"PA={PA_TYPE}  blocks={N_BLOCKS} (step {STEP_BLK})  frames/blk={N_FRAMES}  "
          f"arms={ARMS}{'  [SMOKE]' if SMOKE else ''}")
    dpd = make_dpd(load_offline=True).cpu().eval()
    pa = make_pa().cpu().eval()
    P0 = extract(dpd)
    prm0 = (P0['Ab1'].clone(), P0['Ce1'].clone(), P0['Ab2'].clone(), P0['Ce2'].clone())

    X_tr, y_tr, *_r, X_te, y_te = load_dataset(dataset_name=DATASET)
    X_tr = np.asarray(X_tr, np.float32); X_te = np.asarray(X_te, np.float32)
    tg = target_gain_of(X_tr, np.asarray(y_tr, np.float32))
    x_seg = X_tr[:ADAPT_LEN]

    identity = drift_models.make_drift(pa, 0, N_BLOCKS, STEP_BLK, kind="memoryless")
    m_off = eval_drift(P0, prm0, identity, X_te, tg)
    print(f"\noffline (undrifted): ACLR {m_off['ACLR']:.2f}  EVM {m_off['EVM']:.2f}  "
          f"NMSE {m_off['NMSE']:.2f}")

    env_lr = os.environ.get("S4D_DRIFT_LR")
    lrs = parse_lr_env(env_lr, ARMS) if env_lr else {}
    for k in ARMS:                       # every adapting SGD arm gets its OWN lr
        if ARM_SETS[k]['n'] and not ARM_SETS[k]['rms'] and k not in lrs:
            lrs[k] = calibrate_lr(P0, prm0, pa, x_seg, X_te, tg, m_off, ARM_SETS[k], k)
    lrs = {k: (LR_RMS if ARM_SETS[k]['rms'] else lrs.get(k, 0.0)) for k in ARMS}
    print("\nper-arm lr: " + "  ".join(f"{k}={lrs[k]:g}" + ("(rms)" if ARM_SETS[k]['rms'] else "")
                                      for k in ARMS if ARM_SETS[k]['n']))

    arms = {k: dict(ARM_SETS[k]) for k in ARMS}
    for a in arms.values():
        a['P'] = dict(P0)
        a['prm'] = tuple(t.clone() for t in prm0)
        a['opt'] = {} if a['rms'] else None       # RMSprop carries RMS registers across blocks
    rngs = {k: np.random.default_rng(0) for k in arms}
    hist = {k: dict(ACLR=[], EVM=[], NMSE=[]) for k in arms}

    print(f"\n{'blk':>3} | " + " | ".join(f"{k + ' (ACLR/EVM)':>20}" for k in arms))
    for blk in range(N_BLOCKS):
        pa_fn = drift_models.make_drift(pa, blk, N_BLOCKS, STEP_BLK, kind=DRIFT_KIND)
        for k, a in arms.items():
            if a['n']:
                a['prm'] = adapt_block(a['P'], a['prm'], pa_fn, x_seg, rngs[k],
                                       a['n'], lrs[k], a, opt_state=a['opt'])
        cells = []
        for k, a in arms.items():
            m = eval_drift(a['P'], a['prm'], pa_fn, X_te, tg)
            for f in ('ACLR', 'EVM', 'NMSE'):
                hist[k][f].append(m[f])
            cells.append(f"{m['ACLR']:8.2f}/{m['EVM']:8.2f}")
        tag = "  <-- step" if blk == STEP_BLK else ""
        print(f"{blk:3d} | " + " | ".join(cells) + tag)

    print("\n            |  ACLR (mean/final/worst)  |  EVM (mean/final/worst)   | NMSE fin")
    for k in arms:
        a = np.array(hist[k]['ACLR']); e = np.array(hist[k]['EVM']); n = np.array(hist[k]['NMSE'])
        print(f"  {k:9s} | {a.mean():7.2f}/{a[-1]:7.2f}/{a.max():7.2f}  "
              f"| {e.mean():7.2f}/{e[-1]:7.2f}/{e.max():7.2f}  | {n[-1]:7.2f}")
    if 'proposed' in arms and 'frozen' in arms:
        dA = hist['proposed']['ACLR'][-1] - hist['frozen']['ACLR'][-1]
        dE = hist['proposed']['EVM'][-1] - hist['frozen']['EVM'][-1]
        print(f"\n  proposed vs frozen @final: ACLR {dA:+.2f} dB, EVM {dE:+.2f} dB")
    if 'proposed' in arms and 'rec_head' in arms:
        print(f"  input-proj trace gain (proposed - rec_head): "
              f"ACLR {hist['proposed']['ACLR'][-1] - hist['rec_head']['ACLR'][-1]:+.2f} dB")
    if 'full_ffn' in arms and 'proposed' in arms:
        print(f"  FFN relax gain (full_ffn - proposed): "
              f"ACLR {hist['full_ffn']['ACLR'][-1] - hist['proposed']['ACLR'][-1]:+.2f} dB")

    # Block 0 is the IDENTITY perturbation: an arm that cannot hold the offline optimum on the
    # UNDRIFTED PA is running above its own stability bound, and its tracking number is not
    # comparable with the others.  Fail loudly -- this is how the lr=0.2 era faked its results.
    # (an `rms` arm is EXEMPT: RMSprop damaging the optimum IS the finding, not a bad lr)
    # judged on ALL THREE metrics -- an ACLR-only gate can pass an arm at +0.62 dB ACLR
    # while it was quietly costing +4.39 dB EVM, which would have flattered that arm
    bad = {}
    for k in arms:
        if not arms[k]['n'] or arms[k]['rms']:
            continue
        d = {f: hist[k][f][0] - m_off[f] for f in ('ACLR', 'EVM', 'NMSE')}
        if max(d.values()) > LR_DAMAGE_TOL or any(np.isnan(v) for v in d.values()):
            bad[k] = d
    if bad:
        print("\n  !! ARM(S) DAMAGE THE UNDRIFTED OPTIMUM AT THEIR lr -- ablation not comparable:")
        for k, d in bad.items():
            print(f"     {k} @ lr={lrs[k]:g}: block-0 damage "
                  f"ACLR {d['ACLR']:+.2f} / EVM {d['EVM']:+.2f} / NMSE {d['NMSE']:+.2f} dB "
                  f"(worst: {max(d, key=d.get)})")
    nan_arms = [k for k in arms if any(np.isnan(v) for v in hist[k]['ACLR'])]
    if nan_arms:
        print(f"\n  !! ARM(S) DIVERGED TO NaN: {nan_arms}  -- pole projection bounds |Abar| but "
              f"NOTHING bounds C_eff: pole projection is necessary, not sufficient")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # tag by arm set: a partial run (e.g. S4D_DRIFT_ARMS=frozen,rms_b4) must NOT overwrite the
    # canonical 5-arm ablation json/png that the paper's tab:ablation cites
    DEFAULT_ARMS = ('frozen', 'rec_only', 'rec_head', 'proposed', 'full_ffn')
    tag = f"{PA_TYPE}{'_smoke' if SMOKE else ''}" \
        + ("" if tuple(ARMS) == DEFAULT_ARMS else "_" + "-".join(ARMS))
    with open(os.path.join(RESULTS_DIR, f"drift_ablation_{tag}.json"), "w") as f:
        json.dump(dict(pa=PA_TYPE, lr=lrs, n_blocks=N_BLOCKS, step_blk=STEP_BLK,
                       n_frames=N_FRAMES, offline=m_off, arms=list(ARMS), hist=hist,
                       undrift_damage={k: hist[k]['ACLR'][0] - m_off['ACLR'] for k in arms}),
                  f, indent=1)
    print(f"\nsaved results/drift_ablation_{tag}.json")

    if not SMOKE:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            t = np.arange(N_BLOCKS)
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            cyc = ["tab:red", "tab:orange", "tab:olive", "tab:green", "tab:blue"]
            colors = {k: cyc[i % len(cyc)] for i, k in enumerate(arms)}
            for k in arms:
                ax[0].plot(t, hist[k]['ACLR'], marker=".", color=colors[k], label=k)
                ax[1].plot(t, hist[k]['EVM'], marker=".", color=colors[k], label=k)
            for a, ttl in zip(ax, ["ACLR (dB)", "EVM (dB)"]):
                a.axvline(STEP_BLK, ls="--", c="gray", lw=1)
                a.set_xlabel("drift block"); a.set_ylabel(ttl); a.grid(alpha=0.3); a.legend(fontsize=8)
            ax[0].set_title("Adaptation-set ablation (per-arm calibrated lr, memoryless)")
            ax[1].set_title("lr: " + ", ".join(f"{k}={lrs[k]:g}"
                                               for k in arms if arms[k]['n']), fontsize=7)
            fig.tight_layout()
            png = f"drift_ablation_{tag}.png"      # tag carries PA + arm set: never collide
            fig.savefig(os.path.join(RESULTS_DIR, png), dpi=130)
            print(f"  saved results/{png}")
        except Exception as e:
            print(f"  (plot skipped: {e})")


if __name__ == "__main__":
    main()
