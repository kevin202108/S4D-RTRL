#!/usr/bin/env python3
"""
fixedpoint_sweep.py -- Stage 3 step 1: fixed-point IN-THE-LOOP feasibility scan
for the RFLO online-learning path of the full s4d_best DPD.

Builds directly on rtrl_rflo_full.py (S1r: RFLO == BPTT verified in float64) and
quantizes the LEARNING datapath -- the part that is new hardware:
    * SSM state s (streaming forward)          [qf_state]
    * eligibility traces p                     [qf_trace]
    * error signals lambda                     [qf_lam]
    * gradients                                [qf_grad]
    * adapted params {Abar, C_eff} storage     [after each update + H1 projection]
The frozen pointwise stack stays float (that is inference hardware, already covered
by MP-DPD/DPD-NeuralEngine-style quantization; final eval also uses float inference
so this scan isolates the learning path).

Two rounding modes for the parameter update, because SGD steps (lr*g ~ 1e-5) are
far below the coefficient LSB at short wordlengths -- the classic fixed-point
learning cliff. 'nearest' shows the cliff; 'stoch' (stochastic rounding) is the
standard hardware fix (B5's wide accumulator is the alternative).

Protocol per (W, rounding): the S1r gate-C perturb-recover run (shared perturbed
start, {Abar,C_eff}-only RFLO SGD + pole projection 0.999, 6 epochs), final metrics
vs the float64 reference. Feasibility gate: some W <= 16 within 0.5 dB of float.

Run:  uv run fixedpoint_sweep.py
"""
import math
import numpy as np
import torch

from online_adaptation import make_pa, make_dpd, target_gain_of, DATASET
from modules.data_collector import load_dataset
from rtrl_rflo_full import (extract, stream, rflo_grads, eval_ported, CD)

RAD = 0.999


# --------------------------------------------------------------------------- #
#  Fixed-point simulation
# --------------------------------------------------------------------------- #
def make_q(w, maxabs, stochastic=False, margin=2.0):
    """Symmetric fixed-point quantizer: wordlength w, range calibrated to maxabs.
    Returns (q, frac_bits). Complex tensors quantize Re/Im independently."""
    # int bits may go NEGATIVE for small-valued classes (= per-class exponent /
    # block scaling in hardware); clamping at 0 zeroes out lambda at short W.
    intb = math.ceil(math.log2(maxabs * margin + 1e-30))
    frac = w - 1 - intb
    delta = 2.0 ** (-frac)
    lo, hi = -(2 ** (w - 1)), 2 ** (w - 1) - 1

    def qr(x):
        v = x / delta
        if stochastic:
            v = torch.floor(v + torch.rand_like(v))
        else:
            v = torch.round(v)
        return torch.clamp(v, lo, hi) * delta

    def q(x):
        if torch.is_complex(x):
            return torch.complex(qr(x.real), qr(x.imag))
        return qr(x)
    return q, frac


def probe_ranges(P, A1, C1, A2, C2, z, tgt, fl=500, nfr=6):
    """Float probe: record dynamic ranges of every learning-path tensor class."""
    mx = dict(state=0.0, trace=0.0, lam=0.0, grad=0.0,
              par=float(max(A1.abs().max(), A2.abs().max(),
                            C1.abs().max(), C2.abs().max())))
    T = z.shape[1]

    def watch(key):
        def f(x):
            v = float(x.abs().max()) if torch.is_complex(x) else float(x.abs().max())
            mx[key] = max(mx[key], v)
            return x
        return f
    for s in range(0, min(nfr * fl, T - fl), fl):
        g, _ = rflo_grads(P, A1, C1, A2, C2, z[:, s:s + fl], tgt[:, s:s + fl],
                          qf_state=watch('state'), qf_trace=watch('trace'),
                          qf_lam=watch('lam'), qf_grad=watch('grad'))
    return mx


# --------------------------------------------------------------------------- #
def run_adapt(P, A1, C1, A2, C2, z, tgt, pa, X_te, tg, lr=0.2, fl=500, epochs=6,
              qfs=None, qpar=None, accum=1, qacc=None, qf_eval=None):
    """accum>1 = B5 gradient accumulation: sum `accum` frame-gradients in a (wide,
    optionally quantized via qacc) accumulator, apply the MEAN every accum frames."""
    proj = lambda A: torch.where(A.abs() > RAD, A / A.abs() * RAD, A)
    if qpar is not None:
        A1, C1 = qpar(A1), qpar(C1); A2, C2 = qpar(A2), qpar(C2)
    rng = np.random.default_rng(0)
    starts = list(range(0, z.shape[1] - fl, fl))
    qfs = qfs or {}
    last = None
    acc = None; nacc = 0
    for ep in range(epochs):
        rng.shuffle(starts)
        for s in starts:
            g, l = rflo_grads(P, A1, C1, A2, C2, z[:, s:s + fl], tgt[:, s:s + fl],
                              **qfs)
            if acc is None:
                acc = {k: torch.zeros_like(v) for k, v in g.items()}
            for k in acc:
                acc[k] = acc[k] + g[k]
                if qacc is not None:
                    acc[k] = qacc(acc[k])
            nacc += 1
            if nacc >= accum:
                A1 = A1 - lr * torch.complex(acc['Ab1r'], acc['Ab1i']) / nacc
                C1 = C1 - lr * torch.complex(acc['Ce1r'], acc['Ce1i']) / nacc
                A2 = A2 - lr * torch.complex(acc['Ab2r'], acc['Ab2i']) / nacc
                C2 = C2 - lr * torch.complex(acc['Ce2r'], acc['Ce2i']) / nacc
                A1, A2 = proj(A1), proj(A2)
                if qpar is not None:
                    A1, C1 = qpar(A1), qpar(C1); A2, C2 = qpar(A2), qpar(C2)
                acc = {k: torch.zeros_like(v) for k, v in acc.items()}
                nacc = 0
            last = l
    m = eval_ported(P, A1, C1, A2, C2, pa, X_te, tg, qf=qf_eval)
    stable = bool((A1.abs().max() < 1) and (A2.abs().max() < 1))
    return m, stable, last


def main():
    torch.manual_seed(0)
    print("=== fixedpoint_sweep: quantized RFLO learning path on s4d_best ===")
    dpd = make_dpd(load_offline=True).cpu().eval()
    pa = make_pa().cpu().eval()
    P = extract(dpd)
    Ab1, Ce1, Ab2, Ce2 = P['Ab1'], P['Ce1'], P['Ab2'], P['Ce2']

    X_tr, y_tr, X_val, y_val, X_te, y_te = load_dataset(dataset_name=DATASET)
    X_tr = np.asarray(X_tr, np.float32); X_te = np.asarray(X_te, np.float32)
    tg = target_gain_of(X_tr, np.asarray(y_tr, np.float32))

    with torch.no_grad():
        seg = torch.tensor(X_tr[None, :12000]).float()
        xpd = dpd(seg); zpa = pa(xpd)
    z_in = torch.complex(zpa[0, :, 0], zpa[0, :, 1]).to(CD)[None]
    tgt = torch.complex(xpd[0, :, 0], xpd[0, :, 1]).to(CD)[None]

    # shared perturbed start (same recipe/seed as S1r gate C)
    rngp = torch.Generator().manual_seed(7)
    pert = lambda x, sc: x * (1 + sc * torch.randn(x.shape, generator=rngp, dtype=torch.float64)) \
        * torch.exp(1j * sc * torch.randn(x.shape, generator=rngp, dtype=torch.float64))
    proj = lambda A: torch.where(A.abs() > RAD, A / A.abs() * RAD, A)
    A1o, C1o = proj(pert(Ab1, 0.02)), pert(Ce1, 0.05)
    A2o, C2o = proj(pert(Ab2, 0.02)), pert(Ce2, 0.05)
    m0 = eval_ported(P, A1o, C1o, A2o, C2o, pa, X_te, tg)
    print(f"perturbed start: ACLR={m0['ACLR']:.2f} EVM={m0['EVM']:.2f} dB")

    # dynamic ranges -> fixed-point formats
    mx = probe_ranges(P, A1o.clone(), C1o.clone(), A2o.clone(), C2o.clone(), z_in, tgt)
    print("\n[probe] learning-path dynamic ranges (max|.|):")
    for k, v in mx.items():
        print(f"  {k:6s} {v:.3e}   (int bits needed ~{max(0, math.ceil(math.log2(v * 2 + 1e-30)))})")

    # float reference (same protocol)
    mF, stF, _ = run_adapt(P, A1o.clone(), C1o.clone(), A2o.clone(), C2o.clone(),
                           z_in, tgt, pa, X_te, tg)
    print(f"\n[float ref] adapted ACLR={mF['ACLR']:.2f} EVM={mF['EVM']:.2f} dB "
          f"(stable={stF})")

    print("\n[sweep] W | rounding | final ACLR / EVM (dB) | dACLR vs float | stable")
    results = {}
    for w in (16, 12, 10, 8):
        for mode in ('nearest', 'stoch'):
            st = (mode == 'stoch')
            qs, fs = make_q(w, mx['state']);  qt, ft = make_q(w, mx['trace'])
            ql, fL = make_q(w, mx['lam']);    qg, fg = make_q(w, mx['grad'])
            qp, fp = make_q(w, mx['par'], stochastic=st)   # rounding mode on UPDATE
            qfs = dict(qf_state=qs, qf_trace=qt, qf_lam=ql, qf_grad=qg)
            m, stable, _ = run_adapt(P, A1o.clone(), C1o.clone(), A2o.clone(),
                                     C2o.clone(), z_in, tgt, pa, X_te, tg,
                                     qfs=qfs, qpar=qp)
            d = m['ACLR'] - mF['ACLR']
            results[(w, mode)] = (m, d, stable)
            print(f"  W{w:2d} | {mode:7s} | {m['ACLR']:7.2f} / {m['EVM']:7.2f} | "
                  f"{d:+.2f} dB | {stable}   (frac bits: s{fs} p{ft} lam{fL} g{fg} par{fp})")

    # ---- sweep 2: per-channel SCALED traces (store p*(1-|Ab|), the RTL fix for
    # the trace dynamic-range bottleneck found above) ----
    print("\n[sweep 2] scaled eligibility traces (p_norm = p*(1-|Ab|), per channel)")

    def make_qtrace_scaled(w, Ab, stochastic=False):
        sc = (1.0 - Ab.abs()).clamp_min(1e-4)                # fixed per-channel scale
        # normalized trace range ~ state range -> calibrate on state probe
        q, frac = make_q(w, mx['state'], stochastic=stochastic)
        def qt(p):
            return q(p * sc) / sc
        return qt, frac
    for w in (12, 10, 8):
        for mode in ('nearest', 'stoch'):
            st = (mode == 'stoch')
            qs, fs = make_q(w, mx['state'])
            qt1, ft = make_qtrace_scaled(w, A1o, stochastic=False)
            qt2, _ = make_qtrace_scaled(w, A2o, stochastic=False)
            ql, fL = make_q(w, mx['lam']); qg, fg = make_q(w, mx['grad'])
            qp, fp = make_q(w, mx['par'], stochastic=st)
            qfs = dict(qf_state=qs, qf_trace=(qt1, qt2), qf_lam=ql, qf_grad=qg)
            m, stable, _ = run_adapt(P, A1o.clone(), C1o.clone(), A2o.clone(),
                                     C2o.clone(), z_in, tgt, pa, X_te, tg,
                                     qfs=qfs, qpar=qp)
            d = m['ACLR'] - mF['ACLR']
            results[(w, mode + '+scaled')] = (m, d, stable)
            print(f"  W{w:2d} | {mode:7s} | {m['ACLR']:7.2f} / {m['EVM']:7.2f} | "
                  f"{d:+.2f} dB | {stable}   (trace frac {ft} after scaling)")

    # ---- sweep 3: SPLIT wordlengths -- state/params stay W16 (inference-grade),
    # only the learning-signal tensors (scaled traces, lambda, grad) go short.
    # This prices the big SRAM (sensitivity memory) at low wordlength. ----
    print("\n[sweep 3] split: state/par @ W16, learning tensors @ W")
    qs16, _ = make_q(16, mx['state'])
    qp16, _ = make_q(16, mx['par'])
    for w in (12, 10, 8):
        qt1, ft = make_qtrace_scaled(w, A1o)
        qt2, _ = make_qtrace_scaled(w, A2o)
        ql, _ = make_q(w, mx['lam']); qg, _ = make_q(w, mx['grad'])
        qfs = dict(qf_state=qs16, qf_trace=(qt1, qt2), qf_lam=ql, qf_grad=qg)
        m, stable, _ = run_adapt(P, A1o.clone(), C1o.clone(), A2o.clone(),
                                 C2o.clone(), z_in, tgt, pa, X_te, tg,
                                 qfs=qfs, qpar=qp16)
        d = m['ACLR'] - mF['ACLR']
        results[(w, 'split16')] = (m, d, stable)
        print(f"  W{w:2d} learn | {m['ACLR']:7.2f} / {m['EVM']:7.2f} | "
              f"{d:+.2f} dB | {stable}")

    # ---- sweep 4: B5 gradient accumulation over K frames (wide W16 accumulator),
    # on top of the winning split config (state/par @W16, learning @W8) ----
    print("\n[sweep 4] B5 accumulate-K (W16 accumulator) on split W8 config")
    qacc, _ = make_q(16, mx['grad'] * 16, margin=2.0)        # wide: headroom for K sums
    qt1, _ = make_qtrace_scaled(8, A1o); qt2, _ = make_qtrace_scaled(8, A2o)
    ql8, _ = make_q(8, mx['lam']); qg8, _ = make_q(8, mx['grad'])
    qfs8 = dict(qf_state=qs16, qf_trace=(qt1, qt2), qf_lam=ql8, qf_grad=qg8)
    for K in (4, 16):
        m, stable, _ = run_adapt(P, A1o.clone(), C1o.clone(), A2o.clone(),
                                 C2o.clone(), z_in, tgt, pa, X_te, tg,
                                 qfs=qfs8, qpar=qp16, accum=K, qacc=qacc)
        d = m['ACLR'] - mF['ACLR']
        results[(K, 'accumK')] = (m, d, stable)
        print(f"  K={K:2d} | {m['ACLR']:7.2f} / {m['EVM']:7.2f} | {d:+.2f} dB | {stable}")

    # ---- sweep 5: DEPLOYED-datapath quantization ("online adaptation as QAT") ----
    # State quantized in BOTH the learning loop AND the deployed inference path.
    # Question: does continuous online adaptation absorb the datapath quantization
    # loss (making offline QAT unnecessary)?
    print("\n[sweep 5] deployed state quantization, learning @W8 split")
    for w in (16, 12, 10):
        qsw, _ = make_q(w, mx['state'])
        qt1, _ = make_qtrace_scaled(8, A1o); qt2, _ = make_qtrace_scaled(8, A2o)
        ql8, _ = make_q(8, mx['lam']); qg8, _ = make_q(8, mx['grad'])
        qfs = dict(qf_state=qsw, qf_trace=(qt1, qt2), qf_lam=ql8, qf_grad=qg8)
        # post-training reference: offline params, quantized inference, NO adaptation
        m_ptq = eval_ported(P, A1o, C1o, A2o, C2o, pa, X_te, tg, qf=qsw)
        m, stable, _ = run_adapt(P, A1o.clone(), C1o.clone(), A2o.clone(),
                                 C2o.clone(), z_in, tgt, pa, X_te, tg,
                                 qfs=qfs, qpar=qp16, qf_eval=qsw)
        d = m['ACLR'] - mF['ACLR']
        results[(w, 'deployed')] = (m, d, stable)
        print(f"  state W{w:2d} | no-adapt {m_ptq['ACLR']:7.2f}/{m_ptq['EVM']:7.2f} -> "
              f"adapted {m['ACLR']:7.2f}/{m['EVM']:7.2f} | vs float ref {d:+.2f} dB | {stable}")

    ok = any(abs(d) < 0.5 and stable for (m, d, stable) in results.values())
    best = min(((w, mode) for (w, mode) in results
                if abs(results[(w, mode)][1]) < 0.5 and results[(w, mode)][2]),
               default=None)
    print()
    if ok:
        print(f"OK: fixed-point RFLO learning path feasible; smallest passing config "
              f"within 0.5 dB of float: {best}")
        return 0
    print("FAIL: no wordlength <= 16 matched float within 0.5 dB -- needs wide "
          "accumulator (B5) / per-channel scaling before RTL.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
