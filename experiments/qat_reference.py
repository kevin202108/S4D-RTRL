#!/usr/bin/env python3
"""
qat_reference.py -- offline QAT reference for the DEPLOYED state wordlength.

Question (raised by QS4D, Siegel et al. 2026, who show QAT buys 3-10 bits of
state quantization on *classification* tasks): could offline QAT rescue the
W12/W10 deployed state datapath, where online adaptation left a ~0.8 dB
residual (fixedpoint_sweep, sweep 5)?

This script fields the strongest trainer we can build -- full-BPTT autograd
with a straight-through estimator (STE) through the deployed state quantizer,
Adam, float64 master weights, ALL parameters free (Abar, C_eff, W_in, the
pointwise stack, output/skip heads) -- i.e., none of the online-loop hardware
constraints. Start = the converged offline s4d_best; objective = the same ILA
post-inverse regression as the online loop; eval = deployed inference with the
same state quantizer. The BEST eval over training is reported (benefit of the
doubt to QAT).

If even this cannot close the state-quantization gap, the residual is run-time
state-rounding NOISE (power ~ LSB^2, independent of the weights), not a
calibration error -- and no amount of offline training can remove it.  Task
tolerance is what separates this from QS4D's conclusion: a 1% classification-
accuracy criterion absorbs that noise; a -50 dB ACLR regression sits right on it.

Run:  uv run experiments/qat_reference.py            (full: W12 + W10, 10 epochs each)
      uv run experiments/qat_reference.py --smoke    (quick wiring check: W12, 1 epoch)
"""
import sys
import numpy as np
import torch

from online_adaptation import make_pa, make_dpd, target_gain_of, DATASET
from modules.data_collector import load_dataset
from rtrl_rflo_full import extract, stream, loss_of, eval_ported, CD
from fixedpoint_sweep import make_q, RAD

# every complex tensor extract() produces -- the FULL parameter set is free
CPLX = ('Ab1', 'Ce1', 'Ab2', 'Ce2', 'Win', 'bin',
        'Wm1', 'bm1', 'Wf1_1', 'bf1_1', 'Wf2_1', 'bf2_1',
        'Wm2', 'bm2', 'Wf1_2', 'bf1_2', 'Wf2_2', 'bf2_2',
        'Wo', 'bo', 'Ws', 'bs')


def qat_train(P, z, tgt, pa, X_te, tg, qf_state, lr=1e-3, fl=500, epochs=10,
              eval_every=2, max_frames=None):
    """Offline QAT fine-tune. STE passes gradient through rounding AND clipping
    (standard QAT; clipping is never active at the calibrated range anyway).
    Returns (best_epoch, best_metrics) over the periodic deployed-path evals."""
    qste = lambda x: x + (qf_state(x) - x).detach()
    L = {}
    for k in CPLX:
        L[k + 'r'] = P[k].real.clone().requires_grad_(True)
        L[k + 'i'] = P[k].imag.clone().requires_grad_(True)
    cplx = lambda k: torch.complex(L[k + 'r'], L[k + 'i'])

    def assemble(detach=False):
        Pq = dict(P)
        for k in CPLX:
            Pq[k] = cplx(k).detach() if detach else cplx(k)
        return Pq

    opt = torch.optim.Adam(L.values(), lr=lr)
    rng = np.random.default_rng(0)
    starts = list(range(0, z.shape[1] - fl, fl))
    if max_frames is not None:
        starts = starts[:max_frames]
    proj = lambda A: torch.where(A.abs() > RAD, A / A.abs() * RAD, A)
    best = None
    for ep in range(1, epochs + 1):
        rng.shuffle(starts)
        losses = []
        for s in starts:
            Pq = assemble()
            out = stream(Pq, Pq['Ab1'], Pq['Ce1'], Pq['Ab2'], Pq['Ce2'],
                         z[:, s:s + fl], grad=True, qf=qste)
            loss = loss_of(out, tgt[:, s:s + fl])
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():                          # pole projection
                for k in ('Ab1', 'Ab2'):
                    A = proj(cplx(k))
                    L[k + 'r'].data.copy_(A.real)
                    L[k + 'i'].data.copy_(A.imag)
            losses.append(float(loss))
        if ep % eval_every == 0 or ep == epochs:
            Pq = assemble(detach=True)
            m = eval_ported(Pq, Pq['Ab1'], Pq['Ce1'], Pq['Ab2'], Pq['Ce2'],
                            pa, X_te, tg, qf=qf_state)
            if best is None or m['ACLR'] < best[1]['ACLR']:
                best = (ep, m)
            print(f"    ep {ep:2d}  train loss {np.mean(losses):.3e}  "
                  f"eval ACLR {m['ACLR']:7.2f}  EVM {m['EVM']:7.2f}")
    return best


def main():
    smoke = '--smoke' in sys.argv
    torch.manual_seed(0)
    print("=== qat_reference: offline QAT reference, deployed state wordlength ===")
    dpd = make_dpd(load_offline=True).cpu().eval()
    pa = make_pa().cpu().eval()
    P = extract(dpd)

    X_tr, y_tr, X_val, y_val, X_te, y_te = load_dataset(dataset_name=DATASET)
    X_tr = np.asarray(X_tr, np.float32)
    X_te = np.asarray(X_te, np.float32)
    tg = target_gain_of(X_tr, np.asarray(y_tr, np.float32))

    # same ILA post-inverse pair as rtrl_rflo_full / fixedpoint_sweep
    with torch.no_grad():
        seg = torch.tensor(X_tr[None, :12000]).float()
        xpd = dpd(seg)
        zpa = pa(xpd)
    z_in = torch.complex(zpa[0, :, 0], zpa[0, :, 1]).to(CD)[None]
    tgt = torch.complex(xpd[0, :, 0], xpd[0, :, 1]).to(CD)[None]

    # state-range probe on the OFFLINE params (deployment calibrates on the
    # shipped model; same 3000-sample window as fixedpoint_sweep's probe_ranges)
    _, S1p, S2p, _, _ = stream(P, P['Ab1'], P['Ce1'], P['Ab2'], P['Ce2'],
                               z_in[:, :3000], want_states=True)
    mx_state = float(max(S1p.abs().max(), S2p.abs().max()))
    print(f"state range probe: max|s| = {mx_state:.3e}")

    mfl = eval_ported(P, P['Ab1'], P['Ce1'], P['Ab2'], P['Ce2'], pa, X_te, tg)
    print(f"float offline eval: ACLR={mfl['ACLR']:.2f} EVM={mfl['EVM']:.2f} "
          f"NMSE={mfl['NMSE']:.2f} dB")
    q16, _ = make_q(16, mx_state)
    m16 = eval_ported(P, P['Ab1'], P['Ce1'], P['Ab2'], P['Ce2'], pa, X_te, tg,
                      qf=q16)
    print(f"W16 PTQ sanity:     ACLR={m16['ACLR']:.2f} "
          f"({m16['ACLR'] - mfl['ACLR']:+.2f} vs float, expect ~transparent)")

    sweeps = ((12,) if smoke else (12, 10))
    kw = dict(epochs=1, eval_every=1, max_frames=3) if smoke else {}
    residual = {}
    for w in sweeps:
        qsw, frac = make_q(w, mx_state)
        m_ptq = eval_ported(P, P['Ab1'], P['Ce1'], P['Ab2'], P['Ce2'],
                            pa, X_te, tg, qf=qsw)
        r_ptq = m_ptq['ACLR'] - mfl['ACLR']
        print(f"\n[W{w}] PTQ (no training): ACLR={m_ptq['ACLR']:.2f} "
              f"({r_ptq:+.2f} vs float)  -- QAT fine-tune (frac bits {frac}):")
        ep, m_qat = qat_train(P, z_in, tgt, pa, X_te, tg, qsw, **kw)
        r_qat = m_qat['ACLR'] - mfl['ACLR']
        residual[w] = (r_ptq, r_qat)
        print(f"  W{w} summary: PTQ {r_ptq:+.2f} dB -> QAT best {r_qat:+.2f} dB "
              f"(ep {ep}); QAT recovered {r_ptq - r_qat:+.2f} dB of the budget")

    if smoke:
        print("\nSMOKE OK (wiring only -- no verdict).")
        return 0

    r12_ptq, r12_qat = residual[12]
    r10_ptq, r10_qat = residual[10]
    print(f"\nverdict inputs: W12 residual after QAT {r12_qat:+.2f} dB "
          f"(the online loop left +1.37 dB in its own frame); "
          f"W10 after QAT {r10_qat:+.2f} dB")
    if r12_qat > 0.4:
        print("PASS: even unconstrained offline QAT (full BPTT + Adam, all params, "
              "STE) cannot close the W12 state-quantization gap -- the residual is "
              "run-time state-rounding noise, not calibration error.  No offline "
              "trainer can remove it; the deployed state datapath must carry the bits.")
        return 0
    print("FAIL: offline QAT closed the W12 gap below 0.4 dB.  The residual is then "
          "partly a calibration error after all -- online adaptation recovers the "
          "calibration component, and offline QAT can recover more.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
