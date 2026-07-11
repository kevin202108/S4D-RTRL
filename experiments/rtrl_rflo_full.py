#!/usr/bin/env python3
"""
rtrl_rflo_full.py -- Task: rerun the online loop with the REAL forward-mode RTRL
sensitivity (not reverse-mode autograd) on the FULL-CAPACITY s4d_best DPD.

Scope follows these design decisions, each chosen for the hardware:
  * Re/Im split-real gradient convention (matches the hardware datapath).
  * RFLO-style layer-local RTRL: exact within each diagonal SSM layer, keeps the
    SAME-timestep cross-layer path (pointwise, hardware computes it in-cycle),
    truncates only the delayed cross-layer paths (layer-1 params through the
    layer-2 state at later times).  Gate: cosine vs BPTT truth > 0.99.
  * Learn the DISCRETE (Abar, C_eff) directly -- no exp/log on the update path.
  * Adaptation set = {Abar, C_eff} of both layers; the pointwise/FFN stack is frozen.
  * Pole projection |Abar| <= 0.995 after each update.

Gates (each prints PASS/FAIL, script exits nonzero on failure):
  A  float64 port parity: custom streaming forward == S4D_DPD.forward (float32 tol)
  B  gradient: RFLO forward-mode vs full BPTT autograd truth on real ILA frames
       layer-2 blocks exact (machine precision), layer-1 blocks cosine > 0.99
  C  online ILA adaptation ({Abar,C_eff} only, pure SGD + pole projection):
       RFLO-updates run vs BPTT-updates run, same frames/lr -> final linearization
       metrics within 0.5 dB, poles stable throughout.

CPU + float64 on purpose: this is a correctness experiment, not a speed one.

Run:  uv run experiments/rtrl_rflo_full.py
"""
import copy
import json
import numpy as np
import torch

import models as model
from modules.data_collector import load_dataset
from utils.metrics import NMSE, EVM, ACLR
from online_adaptation import (make_pa, make_dpd, target_gain_of,
                           DATASET, FS, BW, NSUB, NPERSEG)

DEV = "cpu"     # note: models stay native float32; only the ported math is float64
CD = torch.complex128


# --------------------------------------------------------------------------- #
#  Extract s4d_best weights into plain complex128 tensors (pointwise frozen)
# --------------------------------------------------------------------------- #
def extract(dpd):
    bb = dpd.backbone
    P = {}
    P['mem_depth'] = bb.mem_depth; P['degrees'] = bb.degrees
    P['phase'] = bb.phase_feats; P['residual'] = bb.residual
    P['Win'] = bb.input_proj.weight.detach().to(CD)          # (H, C)
    P['bin'] = bb.input_proj.bias.detach().to(CD)            # (H,)
    for l, (layer, ffn) in enumerate(zip(bb.layers, bb.ffns), 1):
        Ab, Ce = layer.kernel.discrete()
        P[f'Ab{l}'] = Ab.detach().to(CD)                     # (H, N)
        P[f'Ce{l}'] = Ce.detach().to(CD)                     # (H, N)
        P[f'Wm{l}'] = layer.output_linear.weight.detach().to(CD)
        P[f'bm{l}'] = layer.output_linear.bias.detach().to(CD)
        P[f'Wf1_{l}'] = ffn.fc1.weight.detach().to(CD)
        P[f'bf1_{l}'] = ffn.fc1.bias.detach().to(CD)
        P[f'Wf2_{l}'] = ffn.fc2.weight.detach().to(CD)
        P[f'bf2_{l}'] = ffn.fc2.bias.detach().to(CD)
    P['Wo'] = bb.output_proj.weight.detach().to(CD)          # (1, H)
    P['bo'] = bb.output_proj.bias.detach().to(CD)
    P['Ws'] = bb.skip.weight.detach().to(CD)                 # (1, C)
    P['bs'] = bb.skip.bias.detach().to(CD)
    return P


def feats_of(z, P):
    """Replicate S4D_DPD._features: z (B,T) complex -> (B,C,T)."""
    amp2 = z.real ** 2 + z.imag ** 2
    cols = []
    for m in range(P['mem_depth']):
        zm = z if m == 0 else torch.roll(z, shifts=m, dims=-1)
        am2 = amp2 if m == 0 else torch.roll(amp2, shifts=m, dims=-1)
        if m > 0:
            zm = zm.clone(); zm[..., :m] = 0
            am2 = am2.clone(); am2[..., :m] = 0
        cols.append(zm)
        for p in P['degrees']:
            cols.append(zm * (am2 ** (p / 2.0)))
        if P['phase']:
            cols.append(zm / (am2.sqrt() + 1e-12))
    return torch.stack(cols, dim=1)


def crelu(z):
    return torch.complex(torch.relu(z.real), torch.relu(z.imag))


def pw(x, W, b):                                             # x (...,Cin) -> (...,Cout)
    return torch.einsum("...c,oc->...o", x, W) + b


def head_t(P, Ab1, Ce1, Ab2, Ce2, h0, ft, s1p, s2p, qf=None):
    """One (batched-over-time OK) step of the full pointwise chain given previous
    states as CONSTANTS. Returns (s1, s2, out). All ops are same-timestep.
    qf: optional state quantizer (fixed-point in-the-loop hook)."""
    s1 = Ab1 * s1p + h0.unsqueeze(-1)
    if qf is not None:
        s1 = qf(s1)
    y1 = (Ce1 * s1).sum(-1)
    y1 = pw(crelu(y1), P['Wm1'], P['bm1'])
    h1 = h0 + y1 if P['residual'] else y1
    h1 = h1 + pw(crelu(pw(h1, P['Wf1_1'], P['bf1_1'])), P['Wf2_1'], P['bf2_1'])
    s2 = Ab2 * s2p + h1.unsqueeze(-1)
    if qf is not None:
        s2 = qf(s2)
    y2 = (Ce2 * s2).sum(-1)
    y2 = pw(crelu(y2), P['Wm2'], P['bm2'])
    h2 = h1 + y2 if P['residual'] else y2
    h2 = h2 + pw(crelu(pw(h2, P['Wf1_2'], P['bf1_2'])), P['Wf2_2'], P['bf2_2'])
    o = pw(h2, P['Wo'], P['bo'])[..., 0] + pw(ft, P['Ws'], P['bs'])[..., 0]
    return s1, s2, o


def stream(P, Ab1, Ce1, Ab2, Ce2, z, want_states=False, grad=False, qf=None):
    """Recurrent forward over a frame. z (B,T) complex -> out (B,T) complex.
    grad=True keeps the full through-time graph (BPTT reference)."""
    B, T = z.shape
    H, N = Ab1.shape
    feats = feats_of(z, P)                                   # (B,C,T)
    H0 = torch.einsum("bct,oc->bot", feats, P['Win']) + P['bin'].view(1, -1, 1)
    s1 = torch.zeros(B, H, N, dtype=CD); s2 = torch.zeros(B, H, N, dtype=CD)
    outs, S1p, S2p = [], [], []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for t in range(T):
            if want_states:
                S1p.append(s1); S2p.append(s2)
            s1, s2, o = head_t(P, Ab1, Ce1, Ab2, Ce2,
                               H0[:, :, t], feats[:, :, t], s1, s2, qf=qf)
            outs.append(o)
    out = torch.stack(outs, dim=-1)                          # (B,T)
    if want_states:
        return out, torch.stack(S1p, 1), torch.stack(S2p, 1), H0, feats
    return out


def loss_of(out, tgt):                                       # MSE on Re/Im, OpenDPD-style
    return ((out - tgt).real ** 2 + (out - tgt).imag ** 2).mean()


# --------------------------------------------------------------------------- #
#  Gradients: BPTT truth (reverse over the full unrolled graph)  vs  RFLO
# --------------------------------------------------------------------------- #
def leaves_of(P):
    L = {}
    for k in ('Ab1', 'Ce1', 'Ab2', 'Ce2'):
        L[k + 'r'] = P[k].real.clone().requires_grad_(True)
        L[k + 'i'] = P[k].imag.clone().requires_grad_(True)
    return L


def bptt_grads(P, L, z, tgt):
    Ab1 = torch.complex(L['Ab1r'], L['Ab1i']); Ce1 = torch.complex(L['Ce1r'], L['Ce1i'])
    Ab2 = torch.complex(L['Ab2r'], L['Ab2i']); Ce2 = torch.complex(L['Ce2r'], L['Ce2i'])
    out = stream(P, Ab1, Ce1, Ab2, Ce2, z, grad=True)
    loss = loss_of(out, tgt)
    gs = torch.autograd.grad(loss, [L[k] for k in
         ('Ab1r', 'Ab1i', 'Ce1r', 'Ce1i', 'Ab2r', 'Ab2i', 'Ce2r', 'Ce2i')])
    return {k: g.detach() for k, g in zip(
        ('Ab1r', 'Ab1i', 'Ce1r', 'Ce1i', 'Ab2r', 'Ab2i', 'Ce2r', 'Ce2i'), gs)}, float(loss)


def rflo_grads(P, Ab1, Ce1, Ab2, Ce2, z, tgt,
               qf_state=None, qf_trace=None, qf_lam=None, qf_grad=None,
               head_grads=False, in_grads=False, pw_grads=False):
    """Forward-mode RFLO: eligibility traces p^l (exact within-layer RTRL), error
    signals lambda^l from the SAME-TIMESTEP pointwise chain (batched over t).
    Delayed cross-layer paths (theta^1 -> s2_{t+k}) are truncated -- that is the
    hardware-cheap approximation gate B quantifies.
    qf_*: optional Stage-3 fixed-point quantizers (state / eligibility trace /
    error signal / gradient). None = exact float (the verified gates)."""
    B, T = z.shape
    # 1) no-grad streaming pass storing previous states per t
    out0, S1p, S2p, H0, feats = stream(P, Ab1, Ce1, Ab2, Ce2, z, want_states=True,
                                       qf=qf_state)
    # 2) batched truncated graph: recompute step t with s_{t-1} as constants,
    #    Ab/Ce as real-pair leaves, and grab d L / d (Re,Im) of s1_t, s2_t.
    L = leaves_of({'Ab1': Ab1, 'Ce1': Ce1, 'Ab2': Ab2, 'Ce2': Ce2})
    Ab1l = torch.complex(L['Ab1r'], L['Ab1i']); Ce1l = torch.complex(L['Ce1r'], L['Ce1i'])
    Ab2l = torch.complex(L['Ab2r'], L['Ab2i']); Ce2l = torch.complex(L['Ce2r'], L['Ce2i'])
    h0b = H0.permute(0, 2, 1).reshape(B * T, -1)             # (B*T, H)
    ftb = feats.permute(0, 2, 1).reshape(B * T, -1)
    s1p = S1p.reshape(B * T, *S1p.shape[2:])
    s2p = S2p.reshape(B * T, *S2p.shape[2:])
    if in_grads:
        # input_proj Win/bin adapted. Split h0 to avoid double counting:
        # the s1-recurrence term uses the CONSTANT h0 (its Win-dependence is carried
        # by the eligibility trace below); the residual path uses the Win-leaf graph
        # version, so autograd gives exactly the non-s1 instantaneous part.
        IL = {}
        for k in ('Win', 'bin'):
            IL[k + 'r'] = P[k].real.clone().requires_grad_(True)
            IL[k + 'i'] = P[k].imag.clone().requires_grad_(True)
        Winl = torch.complex(IL['Winr'], IL['Wini'])
        binl = torch.complex(IL['binr'], IL['bini'])
        h0g = pw(ftb, Winl, binl)                            # (B*T, H) Win-leaf
    else:
        h0g = h0b
    PW_KEYS = ('Wm1', 'bm1', 'Wf1_1', 'bf1_1', 'Wf2_1', 'bf2_1',
               'Wm2', 'bm2', 'Wf1_2', 'bf1_2', 'Wf2_2', 'bf2_2')
    if pw_grads:
        # pointwise mixing/FFN params adapted (the relaxed set). They are STATELESS
        # (per-timestep), so grads are instantaneous -- same-timestep exact,
        # delayed cross-layer paths truncated (same RFLO caveat as C_eff^1).
        PL = {}
        for k in PW_KEYS:
            PL[k + 'r'] = P[k].real.clone().requires_grad_(True)
            PL[k + 'i'] = P[k].imag.clone().requires_grad_(True)
        gp = {k: torch.complex(PL[k + 'r'], PL[k + 'i']) for k in PW_KEYS}
    else:
        gp = {k: P[k] for k in PW_KEYS}
    s1 = Ab1l * s1p + h0b.unsqueeze(-1)
    s1r_n, s1i_n = s1.real, s1.imag                          # intermediate nodes
    s1u = torch.complex(s1r_n, s1i_n)
    y1 = (Ce1l * s1u).sum(-1)
    y1 = pw(crelu(y1), gp['Wm1'], gp['bm1'])
    h1 = h0g + y1 if P['residual'] else y1
    h1 = h1 + pw(crelu(pw(h1, gp['Wf1_1'], gp['bf1_1'])), gp['Wf2_1'], gp['bf2_1'])
    s2 = Ab2l * s2p + h1.unsqueeze(-1)
    s2r_n, s2i_n = s2.real, s2.imag
    s2u = torch.complex(s2r_n, s2i_n)
    y2 = (Ce2l * s2u).sum(-1)
    y2 = pw(crelu(y2), gp['Wm2'], gp['bm2'])
    h2 = h1 + y2 if P['residual'] else y2
    h2 = h2 + pw(crelu(pw(h2, gp['Wf1_2'], gp['bf1_2'])), gp['Wf2_2'], gp['bf2_2'])
    if head_grads:
        # output-side head params: STATELESS -> gradients are instantaneous & exact
        HL = {}
        for k in ('Wo', 'bo', 'Ws', 'bs'):
            HL[k + 'r'] = P[k].real.clone().requires_grad_(True)
            HL[k + 'i'] = P[k].imag.clone().requires_grad_(True)
        Wo = torch.complex(HL['Wor'], HL['Woi']); bo = torch.complex(HL['bor'], HL['boi'])
        Ws = torch.complex(HL['Wsr'], HL['Wsi']); bs = torch.complex(HL['bsr'], HL['bsi'])
        o = pw(h2, Wo, bo)[..., 0] + pw(ftb, Ws, bs)[..., 0]
    else:
        o = pw(h2, P['Wo'], P['bo'])[..., 0] + pw(ftb, P['Ws'], P['bs'])[..., 0]
    tgtb = tgt.reshape(B * T)
    loss = ((o - tgtb).real ** 2 + (o - tgtb).imag ** 2).sum() / (B * T)  # == frame MSE
    wants = [s1r_n, s1i_n, s2r_n, s2i_n,
             L['Ce1r'], L['Ce1i'], L['Ce2r'], L['Ce2i']]
    hkeys = []
    if head_grads:
        hkeys += ['Wor', 'Woi', 'bor', 'boi', 'Wsr', 'Wsi', 'bsr', 'bsi']
        wants += [HL[k] for k in hkeys[-8:]]
    if in_grads:
        hkeys += ['Winr', 'Wini', 'binr', 'bini']
        wants += [IL[k] for k in hkeys[-4:]]
    if pw_grads:
        pkeys = [k + s for k in PW_KEYS for s in ('r', 'i')]
        hkeys += pkeys
        wants += [PL[k] for k in pkeys]
    lam = torch.autograd.grad(loss, wants)
    l1R, l1I, l2R, l2I, gC1r, gC1i, gC2r, gC2i = [g.detach() for g in lam[:8]]
    hg = {k: g.detach() for k, g in zip(hkeys, lam[8:])}
    if qf_lam is not None:
        l1R, l1I, l2R, l2I = qf_lam(l1R), qf_lam(l1I), qf_lam(l2R), qf_lam(l2I)
    l1R = l1R.reshape(B, T, *l1R.shape[1:]); l1I = l1I.reshape(B, T, *l1I.shape[1:])
    l2R = l2R.reshape(B, T, *l2R.shape[1:]); l2I = l2I.reshape(B, T, *l2I.shape[1:])
    # 3) eligibility scan + Re/Im combine (split-real convention)
    g = {'Ce1r': gC1r, 'Ce1i': gC1i, 'Ce2r': gC2r, 'Ce2i': gC2i}
    for li, (Ab, Sp, lR, lI, kr, ki) in enumerate((
            (Ab1, S1p, l1R, l1I, 'Ab1r', 'Ab1i'),
            (Ab2, S2p, l2R, l2I, 'Ab2r', 'Ab2i'))):
        qt = qf_trace[li] if isinstance(qf_trace, (tuple, list)) else qf_trace
        p = torch.zeros_like(Sp[:, 0])
        gr = torch.zeros_like(Ab.real); gi = torch.zeros_like(Ab.real)
        for t in range(T):
            p = Sp[:, t] + Ab * p                            # p_t = s_{t-1} + Ab p_{t-1}
            if qt is not None:
                p = qt(p)
            gr = gr + (lR[:, t] * p.real + lI[:, t] * p.imag).sum(0)
            gi = gi + (-lR[:, t] * p.imag + lI[:, t] * p.real).sum(0)
        g[kr] = gr; g[ki] = gi
    if in_grads:
        # eligibility traces for input_proj (through the s1 recurrence):
        #   q_t = Ab1 (.) q_{t-1} + feats_t     (per output channel h, per state n)
        H, N = Ab1.shape
        Cin = feats.shape[1]
        fB = feats.permute(0, 2, 1)                          # (B,T,C)
        q = torch.zeros(B, H, N, Cin, dtype=Ab1.dtype)
        qb = torch.zeros(B, H, N, dtype=Ab1.dtype)
        gwr = torch.zeros(H, Cin); gwi = torch.zeros(H, Cin)
        gbr = torch.zeros(H); gbi = torch.zeros(H)
        for t in range(T):
            q = Ab1[None, :, :, None] * q + fB[:, t][:, None, None, :]
            qb = Ab1[None] * qb + 1.0
            lR, lI = l1R[:, t], l1I[:, t]                    # (B,H,N)
            gwr += (torch.einsum('bhn,bhnc->hc', lR, q.real)
                    + torch.einsum('bhn,bhnc->hc', lI, q.imag))
            gwi += (-torch.einsum('bhn,bhnc->hc', lR, q.imag)
                    + torch.einsum('bhn,bhnc->hc', lI, q.real))
            gbr += (lR * qb.real + lI * qb.imag).sum((0, 2))
            gbi += (-lR * qb.imag + lI * qb.real).sum((0, 2))
        hg['Winr'] = hg['Winr'] + gwr; hg['Wini'] = hg['Wini'] + gwi
        hg['binr'] = hg['binr'] + gbr; hg['bini'] = hg['bini'] + gbi
    g.update(hg)
    if qf_grad is not None:
        # `qf_grad` may be a dict keyed by gradient class -> PER-CLASS block exponents,
        # matching the hardware datapath.  A single shared exponent is calibrated by
        # whichever class is largest, so the small classes underflow wholesale: the spread
        # between classes exceeds 10x, and the class that loses is gCe -- the one doing
        # nearly all the ILA adaptation.  A plain callable is still accepted, and
        # reproduces that single-exponent behaviour as a control.
        g = ({k: qf_grad[k](v) for k, v in g.items()} if isinstance(qf_grad, dict)
             else {k: qf_grad(v) for k, v in g.items()})
    return g, float(loss)


def cosine(a, b):
    a = a.flatten(); b = b.flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


# --------------------------------------------------------------------------- #
#  Metrics with the ported forward (adapted {Ab,Ce} live outside the nn.Module)
# --------------------------------------------------------------------------- #
def eval_ported(P, Ab1, Ce1, Ab2, Ce2, pa, X_np, tg, nseg=2, qf=None):
    """OpenDPD-faithful eval on the first `nseg` test segments (CPU, keeps it quick).
    qf: optional state quantizer on the DEPLOYED inference path (QAT-style eval)."""
    n = X_np.shape[0] // NPERSEG
    Xs = X_np[:min(nseg, n) * NPERSEG].reshape(-1, NPERSEG, 2).astype(np.float64)
    z = torch.tensor(Xs[..., 0] + 1j * Xs[..., 1], dtype=CD)
    out = stream(P, Ab1, Ce1, Ab2, Ce2, z, qf=qf)            # (B,T) complex
    xpd = torch.stack([out.real, out.imag], -1).to(torch.float32)
    with torch.no_grad():
        yb = pa(xpd).numpy().astype(np.float64)
    gt = tg * Xs
    aclr = np.mean(ACLR(yb, fs=FS, nperseg=NPERSEG, bw_main_ch=BW, n_sub_ch=NSUB))
    return dict(NMSE=float(NMSE(yb, gt)),
                EVM=float(EVM(yb, gt, sample_rate=FS, bw_main_ch=BW,
                              n_sub_ch=NSUB, nperseg=NPERSEG)),
                ACLR=float(aclr))


# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(0)
    print("=== rtrl_rflo_full: forward-mode RFLO on full s4d_best (CPU/float64) ===")
    dpd = make_dpd(load_offline=True).cpu().eval()
    pa = make_pa().cpu().eval()
    P = extract(dpd)
    Ab1, Ce1 = P['Ab1'].clone(), P['Ce1'].clone()
    Ab2, Ce2 = P['Ab2'].clone(), P['Ce2'].clone()
    H, N = Ab1.shape
    print(f"model: H={H} N={N} layers=2 mem={P['mem_depth']} phase={P['phase']}  "
          f"max|Ab|={max(Ab1.abs().max(), Ab2.abs().max()):.4f}")

    X_tr, y_tr, X_val, y_val, X_te, y_te = load_dataset(dataset_name=DATASET)
    X_tr = np.asarray(X_tr, np.float32); X_te = np.asarray(X_te, np.float32)
    tg = target_gain_of(X_tr, np.asarray(y_tr, np.float32))

    # ---------------- Gate A: float64 port parity ----------------
    zs = torch.tensor(X_tr[:2000, 0] + 1j * X_tr[:2000, 1], dtype=CD)[None]
    with torch.no_grad():
        ref = dpd(torch.tensor(X_tr[None, :2000]).float())
    ref_c = torch.complex(ref[..., 0], ref[..., 1]).to(CD)
    got = stream(P, Ab1, Ce1, Ab2, Ce2, zs)
    errA = float((got - ref_c).abs().max())
    okA = errA < 1e-3                                         # float32 model tolerance
    print(f"\n[Gate A] port parity max|diff| = {errA:.2e}  -> {'PASS' if okA else 'FAIL'}")

    # ---------------- ILA data for gates B/C: xpd, z (all detached) --------------
    with torch.no_grad():
        seg = torch.tensor(X_tr[None, :12000]).float()
        xpd = dpd(seg)                                        # predistorter output
        zpa = pa(xpd)                                         # observed PA output
    z_in = torch.complex(zpa[0, :, 0], zpa[0, :, 1]).to(CD)[None]     # post-inv INPUT
    tgt = torch.complex(xpd[0, :, 0], xpd[0, :, 1]).to(CD)[None]      # post-inv TARGET

    # ---------------- Gate B: RFLO vs BPTT truth on ILA frames ----------------
    fl = 512
    okB = True
    print(f"\n[Gate B] RFLO vs BPTT truth (fl={fl}, ILA post-inverse objective)")
    for s in (0, 4000, 8000):
        zf, tf = z_in[:, s:s + fl], tgt[:, s:s + fl]
        L = leaves_of({'Ab1': Ab1, 'Ce1': Ce1, 'Ab2': Ab2, 'Ce2': Ce2})
        gT, lT = bptt_grads(P, L, zf, tf)
        gF, lF = rflo_grads(P, Ab1, Ce1, Ab2, Ce2, zf, tf)
        cs = {k: cosine(gF[k], gT[k]) for k in gT}
        e2 = {k: float((gF[k] - gT[k]).abs().max()) for k in ('Ab2r', 'Ab2i', 'Ce2r', 'Ce2i')}
        c1 = min(cs['Ab1r'], cs['Ab1i'], cs['Ce1r'], cs['Ce1i'])
        c2 = min(cs['Ab2r'], cs['Ab2i'], cs['Ce2r'], cs['Ce2i'])
        exact2 = max(e2.values())
        cg = cosine(torch.cat([gF[k].flatten() for k in sorted(gT)]),
                    torch.cat([gT[k].flatten() for k in sorted(gT)]))
        print(f"  frame@{s:5d}  loss={lT:.3e}  GLOBAL cos={cg:.6f}  "
              f"layer1 min-cos={c1:.6f}  layer2 min-cos={c2:.9f}  "
              f"layer2 max|diff|={exact2:.2e}")
        okB &= (cg > 0.99) and (c1 > 0.95) and (exact2 < 1e-9)
    print(f"  -> {'PASS' if okB else 'FAIL'} "
          f"(gate: global cos>0.99, per-block floor 0.95, layer2 exact)")

    # ------- Gate B2: input_proj grads (C2-3: trace + residual-direct) vs BPTT ----
    zf, tf = z_in[:, :fl], tgt[:, :fl]
    Wr = P['Win'].real.clone().requires_grad_(True)
    Wi = P['Win'].imag.clone().requires_grad_(True)
    br = P['bin'].real.clone().requires_grad_(True)
    bi = P['bin'].imag.clone().requires_grad_(True)
    P2 = dict(P); P2['Win'] = torch.complex(Wr, Wi); P2['bin'] = torch.complex(br, bi)
    outw = stream(P2, Ab1, Ce1, Ab2, Ce2, zf, grad=True)
    tw = torch.autograd.grad(loss_of(outw, tf), [Wr, Wi, br, bi])
    gI, _ = rflo_grads(P, Ab1, Ce1, Ab2, Ce2, zf, tf, in_grads=True)
    csI = {k: cosine(gI[k], t) for k, t in zip(('Winr', 'Wini', 'binr', 'bini'), tw)}
    okB2 = min(csI.values()) > 0.95
    print(f"\n[Gate B2] input_proj RFLO vs BPTT: "
          + "  ".join(f"{k}={v:.4f}" for k, v in csI.items())
          + f"  -> {'PASS' if okB2 else 'FAIL'} (floor 0.95)")
    okB &= okB2

    # ------- Gate B3: pointwise (C5-relaxed) grads, spot check Wm1/Wf1_2 ----------
    Mr = P['Wm1'].real.clone().requires_grad_(True)
    Mi = P['Wm1'].imag.clone().requires_grad_(True)
    Fr = P['Wf1_2'].real.clone().requires_grad_(True)
    Fi = P['Wf1_2'].imag.clone().requires_grad_(True)
    P3 = dict(P); P3['Wm1'] = torch.complex(Mr, Mi); P3['Wf1_2'] = torch.complex(Fr, Fi)
    outp = stream(P3, Ab1, Ce1, Ab2, Ce2, zf, grad=True)
    tp = torch.autograd.grad(loss_of(outp, tf), [Mr, Mi, Fr, Fi])
    gP, _ = rflo_grads(P, Ab1, Ce1, Ab2, Ce2, zf, tf, pw_grads=True)
    csP = {k: cosine(gP[k], t) for k, t in zip(('Wm1r', 'Wm1i', 'Wf1_2r', 'Wf1_2i'), tp)}
    okB3 = min(csP.values()) > 0.95
    print(f"[Gate B3] pointwise RFLO vs BPTT:  "
          + "  ".join(f"{k}={v:.4f}" for k, v in csP.items())
          + f"  -> {'PASS' if okB3 else 'FAIL'} (floor 0.95)")
    okB &= okB3

    # ---------------- Gate C: online ILA adaptation, RFLO vs BPTT updates --------
    print("\n[Gate C] online ILA adaptation ({Ab,Ce} only, pure SGD + pole proj)")
    base = eval_ported(P, Ab1, Ce1, Ab2, Ce2, pa, X_te, tg)
    print(f"  start (offline s4d_best): ACLR={base['ACLR']:.2f} EVM={base['EVM']:.2f} "
          f"NMSE={base['NMSE']:.2f} dB")
    # perturb poles/readouts (mimic drift-induced mismatch), then adapt back online
    rngp = torch.Generator().manual_seed(7)
    pert = lambda x, sc: x * (1 + sc * torch.randn(x.shape, generator=rngp, dtype=torch.float64)) \
        * torch.exp(1j * sc * torch.randn(x.shape, generator=rngp, dtype=torch.float64))
    runs = {}
    RAD = 0.999                                               # > offline max|Ab|=0.9988
    proj = lambda A: torch.where(A.abs() > RAD, A / A.abs() * RAD, A)
    # ONE perturbation draw -> both runs start from identical params (fair A/B)
    A1o, C1o = proj(pert(Ab1, 0.02)), pert(Ce1, 0.05)
    A2o, C2o = proj(pert(Ab2, 0.02)), pert(Ce2, 0.05)
    m0 = eval_ported(P, A1o, C1o, A2o, C2o, pa, X_te, tg)
    print(f"  perturbed start (shared): ACLR={m0['ACLR']:.2f} EVM={m0['EVM']:.2f} dB")
    for tag in ('rflo', 'bptt'):
        A1, C1, A2, C2 = A1o.clone(), C1o.clone(), A2o.clone(), C2o.clone()
        lr, fl2 = 0.2, 500
        rng = np.random.default_rng(0)
        starts = list(range(0, z_in.shape[1] - fl2, fl2))
        hist = []
        for ep in range(6):
            rng.shuffle(starts)
            for s in starts:
                zf, tf = z_in[:, s:s + fl2], tgt[:, s:s + fl2]
                if tag == 'rflo':
                    g, l = rflo_grads(P, A1, C1, A2, C2, zf, tf)
                else:
                    L = leaves_of({'Ab1': A1, 'Ce1': C1, 'Ab2': A2, 'Ce2': C2})
                    g, l = bptt_grads(P, L, zf, tf)
                A1 = A1 - lr * torch.complex(g['Ab1r'], g['Ab1i'])
                C1 = C1 - lr * torch.complex(g['Ce1r'], g['Ce1i'])
                A2 = A2 - lr * torch.complex(g['Ab2r'], g['Ab2i'])
                C2 = C2 - lr * torch.complex(g['Ce2r'], g['Ce2i'])
                A1 = proj(A1); A2 = proj(A2)                  # pole projection
                hist.append(l)
            print(f"    [{tag}] epoch {ep}  post-inv loss={np.mean(hist[-len(starts):]):.3e}")
        mf = eval_ported(P, A1, C1, A2, C2, pa, X_te, tg)
        stable = bool((A1.abs().max() < 1) and (A2.abs().max() < 1))
        runs[tag] = dict(start=m0, final=mf, stable=stable)
        print(f"    [{tag}] perturbed {m0['ACLR']:.2f}/{m0['EVM']:.2f} -> "
              f"adapted {mf['ACLR']:.2f}/{mf['EVM']:.2f} dB  (poles stable={stable})")
    dA = abs(runs['rflo']['final']['ACLR'] - runs['bptt']['final']['ACLR'])
    dE = abs(runs['rflo']['final']['EVM'] - runs['bptt']['final']['EVM'])
    hurt = base['ACLR'] - runs['rflo']['start']['ACLR']       # perturbation damage (dB)
    recovered = (runs['rflo']['final']['ACLR'] < runs['rflo']['start']['ACLR'] - 1.0) \
        or (hurt > -2.0)                                      # tiny damage -> nothing to recover
    okC = (dA < 0.5) and (dE < 0.5) and runs['rflo']['stable'] and recovered
    print(f"  -> {'PASS' if okC else 'FAIL'} (RFLO vs BPTT final gap: "
          f"ACLR {dA:.2f} dB, EVM {dE:.2f} dB, gate <0.5 dB; perturb damage "
          f"{-hurt:.1f} dB, recovered={recovered})")

    print()
    if okA and okB and okC:
        print("OK: forward-mode RFLO (O(N)/step, no BPTT buffers) reproduces the "
              "autograd online loop on the full s4d_best.")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
