#!/usr/bin/env python3
"""
lr_sweep.py -- DECIDE the online learning rate by measuring tracking, not by a rule.

fixedpoint_drift.calibrate_lr picks the *largest lr that does not damage the undrifted
optimum* (a stability bound).  That is necessary but not sufficient: under drift the
tracking error is lag (falls with lr) PLUS misadjustment / gradient-noise (rises with lr),
so the tracking-optimal lr can sit BELOW the stability bound.  Largest-safe != best-tracking.

This script settles it empirically.  Fixed adaptation set = `proposed`
({Abar, C_eff, W_in, W_out, W_skip} + head), float SGD, the same memoryless drift operator
as drift_ablation.py / fixedpoint_drift.py (12 blocks, bias step at block 6).  For each lr it
runs the WHOLE protocol from the offline optimum and reports:

    undrift : block-0 ACLR AFTER one block of adaptation on the UNDRIFTED PA (block 0 is the
              identity perturbation).  Must stay near the offline -50.79 dB -- an lr that
              moves this is damaging the optimum, disqualified no matter how it 'tracks'.
    mean    : mean ACLR/EVM over all blocks (overall tracking incl. the post-step transient)
    final   : block-11 ACLR/EVM (steady state, 5 blocks after the step -> recovered)

Pick the lr that maximizes tracking among the undamaging ones; feed it to drift_ablation.py
(S4D_DRIFT_LR) so the ablation is run at a MEASURED-best lr, not a heuristic one.

lr IS SCOPED TO THE ADAPTATION SET -- it does not transfer between arms.  A bigger adaptation
set is less stable, so the lr calibrated on `proposed` DAMAGES `full_ffn` (block-0 -48.4 vs
-50.8); and the smaller sets (`rec_only`, `rec_head`) tolerate a LARGER lr, so running them at
proposed's lr UNDERSTATES them -- which would flatter our own design.  An ablation is only fair
if every arm is given its OWN knee.  Sweep each arm (S4D_SWEEP_ARM), then run the ablation with
the per-arm lrs (S4D_DRIFT_LR="rec_only=..,rec_head=..,proposed=..,full_ffn=..").

Run:   uv run experiments/lr_sweep.py
Env:   S4D_PA=...  S4D_LR_GRID=0.2,0.1,0.05,0.02,0.01  S4D_SWEEP_ARM=proposed
"""
import os
import json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "results")

from online_adaptation import (make_pa, make_dpd, target_gain_of, DATASET, PA_TYPE)
from modules.data_collector import load_dataset
from rtrl_rflo_full import extract, CD
import drift_models
from drift_ablation import (adapt_block, eval_drift, ARM_SETS, LR_DAMAGE_TOL,
                          N_BLOCKS, STEP_BLK, N_FRAMES, ADAPT_LEN, DRIFT_KIND)

GRID = tuple(float(x) for x in
             os.environ.get("S4D_LR_GRID", "0.2,0.1,0.05,0.02,0.01").split(","))
ARM = os.environ.get("S4D_SWEEP_ARM", "proposed")
SPEC = ARM_SETS[ARM]


def run_one(P0, prm0, pa, x_seg, X_te, tg, lr):
    P = dict(P0)
    prm = tuple(t.clone() for t in prm0)
    rng = np.random.default_rng(0)
    hist = dict(ACLR=[], EVM=[], NMSE=[])
    for blk in range(N_BLOCKS):
        pa_fn = drift_models.make_drift(pa, blk, N_BLOCKS, STEP_BLK, kind=DRIFT_KIND)
        prm = adapt_block(P, prm, pa_fn, x_seg, rng, N_FRAMES, lr, SPEC)
        m = eval_drift(P, prm, pa_fn, X_te, tg)
        for f in ('ACLR', 'EVM', 'NMSE'):
            hist[f].append(m[f])
    return hist


def main():
    torch.manual_seed(0)
    print(f"=== lr sweep: DECIDE the online lr by tracking ({ARM} arm, memoryless) ===")
    print(f"PA={PA_TYPE}  arm={ARM} {SPEC}  blocks={N_BLOCKS} (step {STEP_BLK})  "
          f"frames/blk={N_FRAMES}  grid={GRID}")
    dpd = make_dpd(load_offline=True).cpu().eval()
    pa = make_pa().cpu().eval()
    P0 = extract(dpd)
    prm0 = (P0['Ab1'].clone(), P0['Ce1'].clone(), P0['Ab2'].clone(), P0['Ce2'].clone())

    X_tr, y_tr, *_r, X_te, y_te = load_dataset(dataset_name=DATASET)
    X_tr = np.asarray(X_tr, np.float32); X_te = np.asarray(X_te, np.float32)
    tg = target_gain_of(X_tr, np.asarray(y_tr, np.float32))
    x_seg = X_tr[:ADAPT_LEN]

    ident = drift_models.make_drift(pa, 0, N_BLOCKS, STEP_BLK, kind="memoryless")
    m_off = eval_drift(P0, prm0, ident, X_te, tg)
    print(f"\noffline (undrifted): ACLR {m_off['ACLR']:.2f}  EVM {m_off['EVM']:.2f}  "
          f"NMSE {m_off['NMSE']:.2f}\n")

    print(f"{'lr':>7} | {'undrift dmg A/E/N':>19} | {'mean A/E':>16} | {'final A/E':>16} "
          f"| {'worst A':>8}")
    out = {}
    for lr in GRID:
        h = run_one(P0, prm0, pa, x_seg, X_te, tg, lr)
        a = np.array(h['ACLR']); e = np.array(h['EVM']); n = np.array(h['NMSE'])
        # DAMAGE = how far block 0 (the identity perturbation, i.e. the UNDRIFTED PA) falls
        # below the offline optimum.  Judge on ALL THREE metrics, not ACLR alone: an lr
        # can cost +0.6 dB ACLR while costing +4.4 dB EVM, and an ACLR-only gate waves it
        # through.  Same tolerance as drift_ablation.LR_DAMAGE_TOL / fixedpoint_drift -- one
        # criterion across all three scripts, or the arms are not comparable.
        dmg = dict(ACLR=float(a[0] - m_off['ACLR']), EVM=float(e[0] - m_off['EVM']),
                   NMSE=float(n[0] - m_off['NMSE']))
        worst_dmg = max(dmg.values())
        nan = bool(np.isnan(a).any())
        out[lr] = dict(hist=h, undrift_ACLR=float(a[0]), undrift_damage=dmg,
                       worst_damage=worst_dmg, diverged=nan,
                       mean_ACLR=float(a.mean()), final_ACLR=float(a[-1]),
                       mean_EVM=float(e.mean()), final_EVM=float(e[-1]),
                       worst_ACLR=float(np.nanmax(a)) if not np.all(np.isnan(a)) else float('nan'))
        flag = ""
        if nan:
            flag = "  <-- DIVERGED (NaN)"
        elif worst_dmg > LR_DAMAGE_TOL:
            flag = f"  <-- DAMAGES optimum ({max(dmg, key=dmg.get)})"
        print(f"{lr:>7g} | {dmg['ACLR']:+5.2f}/{dmg['EVM']:+5.2f}/{dmg['NMSE']:+5.2f} "
              f"| {a.mean():>7.2f}/{e.mean():>7.2f} "
              f"| {a[-1]:>7.2f}/{e[-1]:>7.2f} | {np.nanmax(a) if not np.all(np.isnan(a)) else float('nan'):>8.2f}{flag}")

    # the knee: best tracking among lrs that leave the offline optimum a fixed point on EVERY
    # metric and do not diverge.  (Tracking is monotone in lr up to the bound, so this is a
    # Pareto choice, not a maximization.)
    safe = {lr: v for lr, v in out.items()
            if v['worst_damage'] <= LR_DAMAGE_TOL and not v['diverged']}
    if not safe:
        raise SystemExit(f"no lr in {GRID} leaves the offline optimum intact for arm {ARM!r}")
    best_mean = min(safe, key=lambda lr: safe[lr]['mean_ACLR'])
    best_final = min(safe, key=lambda lr: safe[lr]['final_ACLR'])
    print(f"\n  best-MEAN-ACLR  (undamaging): lr={best_mean}  "
          f"({safe[best_mean]['mean_ACLR']:.2f} dB)")
    print(f"  best-FINAL-ACLR (undamaging): lr={best_final}  "
          f"({safe[best_final]['final_ACLR']:.2f} dB)")
    print(f"  -> run the ablation with S4D_DRIFT_LR={ARM}={best_mean}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"{PA_TYPE}_{ARM}"
    with open(os.path.join(RESULTS_DIR, f"lr_sweep_{tag}.json"), "w") as f:
        json.dump(dict(pa=PA_TYPE, arm=ARM, spec=SPEC, offline=m_off, grid=list(GRID),
                       best_mean=best_mean, best_final=best_final,
                       results={str(k): v for k, v in out.items()}), f, indent=1)
    print(f"\nsaved results/lr_sweep_{tag}.json")


if __name__ == "__main__":
    main()
