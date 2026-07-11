#!/usr/bin/env python3
"""
wordlength_sweep.py -- PA proxy x {DLA, ILA} x eligibility-trace wordlength.

How much wordlength does the eligibility trace actually need?  The trace is the largest
SRAM in the learning datapath, so this is the number the hardware budget turns on.

Protocol (three invariants, each of which will silently fake a wordlength effect if
you get it wrong):

1.  **The PA proxy and its offline DPD are one choice.**  Driving a PA with a DPD
    trained against a *different* PA starts adaptation ~10 dB off that PA's own
    optimum, and the run then measures recovery from a bad initial condition rather
    than the wordlength.

2.  **The learning rate does not transfer across PA proxies.**  Gradient scale is a
    property of the PA, and make_q sets the LSB from the probed maximum -- so an lr
    borrowed from another proxy both overshoots AND destroys the gradient
    quantization.  We tune lr per (PA, mode) on that pair's own float arm, then hold
    it fixed across the wordlength axis, so wordlength is the only thing that moves
    within a pair.

3.  **Gradient block exponents are per class.**  A single exponent shared across
    {gAb, gCe} is calibrated by whichever class is largest, and the rest of the
    classes underflow wholesale.  The gap between classes exceeds 10x, so a shared
    exponent can cost gCe -- the parameter that does nearly all the ILA adaptation --
    over half of its components.

Rows: `W8/W12/W16` = trace wordlength at per-class gradient exponents.  `W16*` is the
control: W16 trace with a single global gradient exponent, i.e. what invariant 3 costs.

Absolute dB are NOT comparable across PA proxies (different DPD, different lr).
Read each pair against its own `offline` and `Float` rows; the wordlength claim is
the *contrast* dFloat, which is what the table reports.

Note `Stable` only checks |Abar| < 1.  It does NOT mean the arm is any good: a
diverged arm with positive EVM still reports Stable=True.  Read dNMSE.

Run:  uv run experiments/wordlength_sweep.py
"""
import sys
import os
import numpy as np
import torch
import math
import copy

import models as model
from online_adaptation import target_gain_of, DATASET, DEVICE, _ckpt
from modules.data_collector import load_dataset
from rtrl_rflo_full import extract, stream, feats_of, CD, head_t, pw, crelu, leaves_of, eval_ported
from fixedpoint_sweep import make_q

RAD = 0.999

# lr is selected per (PA, mode) on the float arm by best NMSE -- NMSE is what the
# adaptation loss actually minimizes.  ACLR must not be the selector: for the linear
# memory drifts it moves the wrong way (drift_models.py), and here it is nearly flat
# in lr while NMSE still separates the arms by several dB.
LR_GRID = (0.4, 0.2, 0.1, 0.05, 0.02, 0.01)

# Each PA proxy needs its OWN offline DPD (invariant 1).  Adding a proxy here means
# adding BOTH entries AND training the s4d_best DPD paired with it -- otherwise the
# sweep measures recovery from a bad initial condition, not the wordlength.  The DPD is
# s4d_best (H8, 2 layers, 1048 params) trained by OpenDPD `train_dpd` with the stock
# defaults (100 ep, lr 5e-4, bs 256, seed 0), so the PA proxy is the only axis that moves.
PA_FILE = {"dgru": "PA_S_0_M_DGRU_H_30_F_200_P_4424.pt"}
DPD_DIR = {"dgru": "PA_S_0_M_DGRU_H_30_F_200"}
DPD_FILE = "DPD_S_0_M_S4D_BEST_H_8_F_200_P_1048.pt"
PA_PROXIES = ("dgru",)

def make_pa_model(backbone_type="dgru"):
    ckpt = _ckpt(os.path.join(DATASET, "train_pa", PA_FILE[backbone_type]))
    pa = model.CoreModel(input_size=2, hidden_size=30, num_layers=1, backbone_type=backbone_type)
    pa.load_state_dict(torch.load(ckpt, map_location="cpu"))
    for p in pa.parameters():
        p.requires_grad = False
    return pa.to(DEVICE).eval()

def make_dpd_for(pa_type):
    """The offline s4d_best DPD trained end-to-end (DLA, float32) against `pa_type`."""
    ckpt = _ckpt(os.path.join(DATASET, "train_dpd", DPD_DIR[pa_type], DPD_FILE))
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"No offline DPD paired with PA '{pa_type}': {ckpt}\n"
            f"Train it first with OpenDPD:  main.py "
            f"--dataset_name {DATASET} --step train_dpd --PA_backbone {pa_type} "
            f"--PA_hidden_size 30 --DPD_backbone s4d_best --DPD_hidden_size 8 "
            f"--DPD_num_layers 2")
    dpd = model.CoreModel(input_size=2, hidden_size=8, num_layers=2, backbone_type="s4d_best")
    dpd.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return dpd.cpu().eval()

def rflo_grads_general(P, Ab1, Ce1, Ab2, Ce2, u_in, tgt, mode="ILA", pa_model=None,
                       qf_state=None, qf_trace=None, qf_lam=None, qf_grad=None):
    B, T = u_in.shape
    # 1) Stream pass
    out0, S1p, S2p, H0, feats = stream(P, Ab1, Ce1, Ab2, Ce2, u_in, want_states=True, qf=qf_state)
    
    # 2) Autograd leaf setup
    L = leaves_of({'Ab1': Ab1, 'Ce1': Ce1, 'Ab2': Ab2, 'Ce2': Ce2})
    Ab1l = torch.complex(L['Ab1r'], L['Ab1i']); Ce1l = torch.complex(L['Ce1r'], L['Ce1i'])
    Ab2l = torch.complex(L['Ab2r'], L['Ab2i']); Ce2l = torch.complex(L['Ce2r'], L['Ce2i'])
    h0b = H0.permute(0, 2, 1).reshape(B * T, -1)
    ftb = feats.permute(0, 2, 1).reshape(B * T, -1)
    s1p = S1p.reshape(B * T, *S1p.shape[2:])
    s2p = S2p.reshape(B * T, *S2p.shape[2:])
    
    gp = {k: P[k] for k in ('Wm1', 'bm1', 'Wf1_1', 'bf1_1', 'Wf2_1', 'bf2_1',
                            'Wm2', 'bm2', 'Wf1_2', 'bf1_2', 'Wf2_2', 'bf2_2')}
    
    s1 = Ab1l * s1p + h0b.unsqueeze(-1)
    s1r_n, s1i_n = s1.real, s1.imag
    s1u = torch.complex(s1r_n, s1i_n)
    y1 = (Ce1l * s1u).sum(-1)
    y1 = pw(crelu(y1), gp['Wm1'], gp['bm1'])
    h1 = h0b + y1 if P['residual'] else y1
    h1 = h1 + pw(crelu(pw(h1, gp['Wf1_1'], gp['bf1_1'])), gp['Wf2_1'], gp['bf2_1'])
    
    s2 = Ab2l * s2p + h1.unsqueeze(-1)
    s2r_n, s2i_n = s2.real, s2.imag
    s2u = torch.complex(s2r_n, s2i_n)
    y2 = (Ce2l * s2u).sum(-1)
    y2 = pw(crelu(y2), gp['Wm2'], gp['bm2'])
    h2 = h1 + y2 if P['residual'] else y2
    h2 = h2 + pw(crelu(pw(h2, gp['Wf1_2'], gp['bf1_2'])), gp['Wf2_2'], gp['bf2_2'])
    
    o = pw(h2, P['Wo'], P['bo'])[..., 0] + pw(ftb, P['Ws'], P['bs'])[..., 0]
    
    if mode == "ILA":
        tgtb = tgt.reshape(B * T)
        loss = ((o - tgtb).real ** 2 + (o - tgtb).imag ** 2).sum() / (B * T)
    else:
        # DLA
        o_complex = o.reshape(B, T)
        xpd = torch.stack([o_complex.real, o_complex.imag], dim=-1).to(torch.float32)
        z_pa = pa_model(xpd) # (B, T, 2)
        z_c = torch.complex(z_pa[..., 0], z_pa[..., 1]).to(CD)
        loss = ((z_c - tgt).real ** 2 + (z_c - tgt).imag ** 2).sum() / (B * T)
        
    wants = [s1r_n, s1i_n, s2r_n, s2i_n,
             L['Ce1r'], L['Ce1i'], L['Ce2r'], L['Ce2i']]
    lam = torch.autograd.grad(loss, wants)
    l1R, l1I, l2R, l2I, gC1r, gC1i, gC2r, gC2i = [g.detach() for g in lam[:8]]
    
    if qf_lam is not None:
        l1R, l1I, l2R, l2I = qf_lam(l1R), qf_lam(l1I), qf_lam(l2R), qf_lam(l2I)
    l1R = l1R.reshape(B, T, *l1R.shape[1:]); l1I = l1I.reshape(B, T, *l1I.shape[1:])
    l2R = l2R.reshape(B, T, *l2R.shape[1:]); l2I = l2I.reshape(B, T, *l2I.shape[1:])
    
    g = {'Ce1r': gC1r, 'Ce1i': gC1i, 'Ce2r': gC2r, 'Ce2i': gC2i}
    for li, (Ab, Sp, lR, lI, kr, ki) in enumerate((
            (Ab1, S1p, l1R, l1I, 'Ab1r', 'Ab1i'),
            (Ab2, S2p, l2R, l2I, 'Ab2r', 'Ab2i'))):
        qt = qf_trace[li] if isinstance(qf_trace, (tuple, list)) else qf_trace
        p = torch.zeros_like(Sp[:, 0])
        gr = torch.zeros_like(Ab.real); gi = torch.zeros_like(Ab.real)
        for t in range(T):
            p = Sp[:, t] + Ab * p
            if qt is not None:
                p = qt(p)
            gr = gr + (lR[:, t] * p.real + lI[:, t] * p.imag).sum(0)
            gi = gi + (-lR[:, t] * p.imag + lI[:, t] * p.real).sum(0)
        g[kr] = gr; g[ki] = gi
        
    if qf_grad is not None:
        # Per-class block exponents (invariant 3).  A single exponent shared by
        # {gAb, gCe} is calibrated by whichever class is largest, so the smaller
        # classes underflow wholesale -- including gCe, which does nearly all the
        # ILA adaptation.  Pass a dict to get one exponent per class; passing a
        # single callable reproduces the shared-exponent control (the `W16*` row).
        g = ({k: qf_grad[k](v) for k, v in g.items()} if isinstance(qf_grad, dict)
             else {k: qf_grad(v) for k, v in g.items()})
    return g, float(loss)


GRAD_KEYS = ('Ab1r', 'Ab1i', 'Ce1r', 'Ce1i', 'Ab2r', 'Ab2i', 'Ce2r', 'Ce2i')

def probe_ranges_general(P, A1, C1, A2, C2, u_in, tgt, mode="ILA", pa_model=None, fl=500, nfr=6):
    """Probe the dynamic range of every quantized class.  `grad` is probed PER CLASS
    (mx['grad_<key>']) as well as globally, so the caller can build per-class block
    exponents; mx['grad'] is retained only for reporting the global-exponent penalty."""
    mx = dict(state=0.0, trace=0.0, lam=0.0, grad=0.0,
              par=float(max(A1.abs().max(), A2.abs().max(),
                            C1.abs().max(), C2.abs().max())))
    mx.update({f"grad_{k}": 0.0 for k in GRAD_KEYS})
    T = u_in.shape[1]
    def watch(key):
        def f(x):
            mx[key] = max(mx[key], float(x.abs().max()))
            return x
        return f
    grad_watch = {k: watch(f"grad_{k}") for k in GRAD_KEYS}
    def watch_grad_all(k):
        def f(x):
            mx['grad'] = max(mx['grad'], float(x.abs().max()))
            return grad_watch[k](x)
        return f
    for s in range(0, min(nfr * fl, T - fl), fl):
        _, _ = rflo_grads_general(P, A1, C1, A2, C2, u_in[:, s:s + fl], tgt[:, s:s + fl],
                                  mode=mode, pa_model=pa_model,
                                  qf_state=watch('state'), qf_trace=watch('trace'),
                                  qf_lam=watch('lam'),
                                  qf_grad={k: watch_grad_all(k) for k in GRAD_KEYS})
    return mx

def run_adapt_general(P, A1, C1, A2, C2, u_in, tgt, pa_model, X_te, tg, mode="ILA",
                      lr=0.2, fl=500, epochs=6, qfs=None, qpar=None):
    proj = lambda A: torch.where(A.abs() > RAD, A / A.abs() * RAD, A)
    if qpar is not None:
        A1, C1 = qpar(A1), qpar(C1); A2, C2 = qpar(A2), qpar(C2)
    rng = np.random.default_rng(0)
    starts = list(range(0, u_in.shape[1] - fl, fl))
    qfs = qfs or {}
    
    for ep in range(epochs):
        rng.shuffle(starts)
        for s in starts:
            g, _ = rflo_grads_general(P, A1, C1, A2, C2, u_in[:, s:s + fl], tgt[:, s:s + fl],
                                      mode=mode, pa_model=pa_model, **qfs)
            
            A1 = A1 - lr * torch.complex(g['Ab1r'], g['Ab1i'])
            C1 = C1 - lr * torch.complex(g['Ce1r'], g['Ce1i'])
            A2 = A2 - lr * torch.complex(g['Ab2r'], g['Ab2i'])
            C2 = C2 - lr * torch.complex(g['Ce2r'], g['Ce2i'])
            A1, A2 = proj(A1), proj(A2)
            if qpar is not None:
                A1, C1 = qpar(A1), qpar(C1); A2, C2 = qpar(A2), qpar(C2)
                
    m = eval_ported(P, A1, C1, A2, C2, pa_model, X_te, tg)
    stable = bool((A1.abs().max() < 1) and (A2.abs().max() < 1))
    return m, stable

def main():
    torch.manual_seed(0)
    print("Loading datasets...")
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_dataset(dataset_name=DATASET)
    X_tr = np.asarray(X_tr, np.float32); X_te = np.asarray(X_te, np.float32)
    tg = target_gain_of(X_tr, np.asarray(y_tr, np.float32))
    
    seg = torch.tensor(X_tr[None, :12000]).float()
    x_in = torch.complex(seg[0, :, 0], seg[0, :, 1]).to(CD)[None]   # DLA input/target
    proj = lambda A: torch.where(A.abs() > RAD, A / A.abs() * RAD, A)

    print("\nStarting Multidimensional Sweep...")
    print("| PA Backbone | Mode | Trace W | lr | ACLR (dB) | EVM (dB) | NMSE (dB) | dACLR | dNMSE | Stable |")
    print("|---|---|---|---|---|---|---|---|---|---|")

    for pa_type in PA_PROXIES:
        pa_model = make_pa_model(pa_type)

        # The DPD must be the one trained against THIS PA, otherwise the sweep
        # measures recovery from a mismatched initial condition, not wordlength.
        dpd = make_dpd_for(pa_type)
        P = extract(dpd)
        Ab1, Ce1, Ab2, Ce2 = P['Ab1'], P['Ce1'], P['Ab2'], P['Ce2']

        with torch.no_grad():
            xpd = dpd(seg)
            z_pa = pa_model(xpd)
        tgt_ila = torch.complex(xpd[0, :, 0], xpd[0, :, 1]).to(CD)[None]   # ILA target
        z_in_ila = torch.complex(z_pa[0, :, 0], z_pa[0, :, 1]).to(CD)[None]

        # Unadapted offline baseline for this pair -- the reference every row below
        # must be read against (absolute dB are NOT comparable across PA proxies).
        m0 = eval_ported(P, Ab1, Ce1, Ab2, Ce2, pa_model, X_te, tg)
        print(f"| {pa_type:12s} | --   | offline | --    | {m0['ACLR']:9.2f} | {m0['EVM']:8.2f} | "
              f"{m0['NMSE']:9.2f} | --    | --    | -- |")

        # Perturb away from the offline optimum (same recipe/seed for every PA).
        rngp = torch.Generator().manual_seed(7)
        pert = lambda x, sc: x * (1 + sc * torch.randn(x.shape, generator=rngp, dtype=torch.float64)) \
            * torch.exp(1j * sc * torch.randn(x.shape, generator=rngp, dtype=torch.float64))
        A1o, C1o = proj(pert(Ab1, 0.02)), pert(Ce1, 0.05)
        A2o, C2o = proj(pert(Ab2, 0.02)), pert(Ce2, 0.05)

        for mode in ("DLA", "ILA"):
            # Set up inputs/targets according to mode
            if mode == "DLA":
                u_in = x_in
                tgt_adapt = x_in
            else:
                u_in = z_in_ila
                tgt_adapt = tgt_ila
                
            # Probe ranges for this specific configuration
            mx = probe_ranges_general(P, A1o.clone(), C1o.clone(), A2o.clone(), C2o.clone(),
                                      u_in, tgt_adapt, mode=mode, pa_model=pa_model)

            # Tune lr on THIS pair's float arm.  lr does not transfer across PA proxies:
            # gradient scale is a property of the PA, so an lr borrowed from another proxy
            # overshoots in float AND destroys the quantized arms.  Select on NMSE (the
            # quantity the adaptation loss minimizes); ACLR is nearly flat in lr here.
            best = None
            for lr in LR_GRID:
                mL, stL = run_adapt_general(P, A1o.clone(), C1o.clone(), A2o.clone(), C2o.clone(),
                                            u_in, tgt_adapt, pa_model, X_te, tg, mode=mode, lr=lr)
                if stL and (best is None or mL['NMSE'] < best[1]['NMSE']):
                    best = (lr, mL, stL)
            if best is None:
                raise RuntimeError(f"no stable lr in {LR_GRID} for {pa_type}/{mode}")
            lr, mF, stF = best
            print(f"| {pa_type:12s} | {mode:4s} | Float   | {lr:<5g} | {mF['ACLR']:9.2f} | {mF['EVM']:8.2f} | "
                  f"{mF['NMSE']:9.2f} | {0.0:5.2f} | {0.0:5.2f} | {stF} |")

            # Sweep Trace Wordlengths: W8, W12, W16
            qs16, _ = make_q(16, mx['state'])
            qp16, _ = make_q(16, mx['par'])
            ql8, _ = make_q(8, mx['lam'])
            qg8_perclass = {k: make_q(8, mx[f"grad_{k}"])[0] for k in GRAD_KEYS}
            qg8_global = make_q(8, mx['grad'])[0]          # the single-global-exponent control

            def make_qtrace_scaled(w, Ab):
                sc = (1.0 - Ab.abs()).clamp_min(1e-4)
                q, _ = make_q(w, mx['state'])
                return lambda p: q(p * sc) / sc

            def run(w, qg, tag):
                qt1 = make_qtrace_scaled(w, A1o)
                qt2 = make_qtrace_scaled(w, A2o)
                qfs = dict(qf_state=qs16, qf_trace=(qt1, qt2), qf_lam=ql8, qf_grad=qg)
                # same lr as this pair's float arm -> wordlength is the only variable
                m, stable = run_adapt_general(P, A1o.clone(), C1o.clone(), A2o.clone(), C2o.clone(),
                                              u_in, tgt_adapt, pa_model, X_te, tg, mode=mode,
                                              lr=lr, qfs=qfs, qpar=qp16)
                dA = m['ACLR'] - mF['ACLR']
                dN = m['NMSE'] - mF['NMSE']
                print(f"| {pa_type:12s} | {mode:4s} | {tag:<7s} | {lr:<5g} | {m['ACLR']:9.2f} | {m['EVM']:8.2f} | "
                      f"{m['NMSE']:9.2f} | {dA:+5.2f} | {dN:+5.2f} | {stable} |")

            for w in (8, 12, 16):
                run(w, qg8_perclass, f"W{w}")
            # one control arm: W16 trace with the old single global gradient exponent,
            # to price the per-class fix on this pair (W16 = the arm where gAb is actually alive)
            run(16, qg8_global, "W16*")

if __name__ == "__main__":
    main()
