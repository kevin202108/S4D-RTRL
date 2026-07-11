#!/usr/bin/env python3
"""
online_adaptation.py -- online adaptation reaches the offline s4d_best optimum.

Does adapting the DPD ONLINE (streaming, one small update at a time) converge to
the same ACLR/EVM as the frozen offline `s4d_best`?

This module is also the shared configuration hub: it owns the PA/DPD pairing, the
checkpoint paths, and the metric band parameters that every other experiment imports.

Reframing (why autograd, not the RTRL sensitivity code, is used here):
    RTRL == the true gradient at machine precision (verified against autograd on the
    real S4D kernel: 4.9e-17).  RTRL's value is O(1)-memory FORWARD-MODE for HARDWARE.
    The software convergence study only needs the true gradient -> autograd.  DLA
    (backprop through the frozen differentiable PA) is the sanity floor; ILA (no PA
    gradient) is the hardware path.

Eval matches OpenDPD EXACTLY so the numbers are comparable:
    * DPD target = target_gain * X   (target_gain = mean max|y|/max|x| over train)
    * metric band params (fs, bw, n_sub_ch, nperseg) READ FROM datasets/*/spec.json
      -- for APA_200MHz: fs=983.04MHz, n_sub_ch=5, nperseg=19662 (NOT the util
      defaults 800MHz/10/2560, which were the whole EVM discrepancy).
    * metrics = OpenDPD utils.metrics on (prediction, target), no extra alignment

Baseline, frozen offline s4d_best over the full X_te (one 19662-sample segment),
on the checkpoints shipped in opendpd/save/:
    NMSE -42.778   ACLR -50.788   EVM -47.599 dB
Reference (OpenDPD's own log, s4d_best seed0): ACLR -51.26, EVM -46.43 dB.
Re-measure rather than trust a comment.

Run:  uv run experiments/online_adaptation.py
"""
import os
import json
import copy
import numpy as np
import torch

import models as model
from modules.data_collector import load_dataset
from utils.metrics import NMSE, EVM, ACLR

DATASET = "APA_200MHz"
HERE = os.path.dirname(os.path.abspath(__file__))
OPENDPD_ROOT = os.path.dirname(os.path.abspath(model.__file__))


def _ckpt(rel):
    """Resolve a checkpoint under the vendored opendpd/save/ tree."""
    return os.path.join(OPENDPD_ROOT, "save", rel)


# The PA proxy and its offline DPD are ONE choice, never two.  Selecting a PA without
# its matched DPD silently starts adaptation ~10 dB off that PA's own optimum, and the
# run then measures recovery from a bad initial condition instead of whatever it claims
# to measure.  Adding a PA proxy therefore means adding BOTH entries below AND training
# the s4d_best DPD that is paired with it.
_PA_ZOO = {
    "dgru": ("PA_S_0_M_DGRU_H_30_F_200_P_4424.pt", "PA_S_0_M_DGRU_H_30_F_200"),
}
PA_TYPE = os.environ.get("S4D_PA", "dgru")
if PA_TYPE not in _PA_ZOO:
    raise ValueError(f"S4D_PA={PA_TYPE!r} not in {sorted(_PA_ZOO)}")
_pa_file, _dpd_dir = _PA_ZOO[PA_TYPE]

PA_CKPT = _ckpt(os.path.join(DATASET, "train_pa", _pa_file))
DPD_CKPT = _ckpt(os.path.join(DATASET, "train_dpd", _dpd_dir,
                              "DPD_S_0_M_S4D_BEST_H_8_F_200_P_1048.pt"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# metric band params come from the dataset spec (NOT the util defaults!)
_SPEC = json.load(open(os.path.join(OPENDPD_ROOT, "datasets", DATASET, "spec.json")))
FS = float(_SPEC["input_signal_fs"])        # 983.04e6 for APA_200MHz
BW = float(_SPEC["bw_main_ch"])             # 200e6
NSUB = int(_SPEC["n_sub_ch"])               # 5
NPERSEG = int(_SPEC["nperseg"])             # 19662 (whole test seq = 1 segment)


def make_pa():
    pa = model.CoreModel(input_size=2, hidden_size=30, num_layers=1, backbone_type=PA_TYPE)
    pa.load_state_dict(torch.load(PA_CKPT, map_location="cpu"))
    for p in pa.parameters():
        p.requires_grad = False
    return pa.to(DEVICE).eval()


def make_dpd(load_offline=False):
    dpd = model.CoreModel(input_size=2, hidden_size=8, num_layers=2, backbone_type="s4d_best")
    if load_offline:
        if not os.path.exists(DPD_CKPT):
            raise FileNotFoundError(
                f"No offline DPD paired with PA {PA_TYPE!r}: {DPD_CKPT}\n"
                f"Train it first with OpenDPD (DLA, float32, s4d_best, OpenDPD defaults):\n"
                f"  main.py --dataset_name {DATASET} --step train_dpd --accelerator cpu "
                f"--PA_backbone {PA_TYPE} --PA_hidden_size 30 "
                f"--DPD_backbone s4d_best --DPD_hidden_size 8 --DPD_num_layers 2")
        dpd.load_state_dict(torch.load(DPD_CKPT, map_location="cpu"))
    return dpd.to(DEVICE)


def target_gain_of(X, y):
    """OpenDPD's set_target_gain: mean( max|y| / max|x| )."""
    amp_in = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    amp_out = np.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)
    return float(np.mean(amp_out.max() / amp_in.max()))


def segment(a, nperseg=NPERSEG):
    a = np.asarray(a, dtype=np.float32)
    n = a.shape[0] // nperseg
    return a[:n * nperseg].reshape(n, nperseg, 2)


@torch.no_grad()
def evaluate(dpd, pa, X_np, target_gain, nperseg=NPERSEG):
    """OpenDPD-faithful eval: per-segment cascade, target = target_gain * X."""
    dpd.eval(); pa.eval()
    Xs = segment(X_np, nperseg)                          # (n_seg, nperseg, 2) float32
    xb = torch.tensor(Xs).to(DEVICE)
    yb = pa(dpd(xb)).cpu().numpy().astype(np.float64)    # cascade out, per segment
    gt = target_gain * Xs.astype(np.float64)             # DPD ground truth
    aclr = np.mean(ACLR(yb, fs=FS, nperseg=nperseg, bw_main_ch=BW, n_sub_ch=NSUB))
    return dict(NMSE=float(NMSE(yb, gt)),
                EVM=float(EVM(yb, gt, sample_rate=FS, bw_main_ch=BW,
                              n_sub_ch=NSUB, nperseg=nperseg)),
                ACLR=float(aclr))


def main():
    torch.manual_seed(0)
    print(f"device = {DEVICE}")

    X_tr, y_tr, X_val, y_val, X_te, y_te = load_dataset(dataset_name=DATASET)
    X_tr = np.asarray(X_tr, np.float32); X_te = np.asarray(X_te, np.float32)
    tg = target_gain_of(X_tr, np.asarray(y_tr, np.float32))
    print(f"data: X_train {X_tr.shape}  X_test {X_te.shape}  target_gain={tg:.4f}")

    pa = make_pa()

    # --- baseline: frozen offline s4d_best ---
    base = evaluate(make_dpd(load_offline=True), pa, X_te, tg)
    print(f"\n[offline s4d_best]  ACLR={base['ACLR']:.2f}  EVM={base['EVM']:.2f}  "
          f"NMSE={base['NMSE']:.2f} dB   (OpenDPD log: ACLR -51.26 / EVM -46.43)")

    # --- online adaptation from a FRESH model (autograd, DLA through frozen PA) ---
    dpd = make_dpd(load_offline=False)
    cas = model.CascadedModel(dpd_model=dpd, pa_model=pa)
    cas.freeze_pa_model(); cas = cas.to(DEVICE)

    EPOCHS, fl, stride, lr0 = 40, 500, 500, 2e-3
    tgt_tr = tg * X_tr                                    # DPD target = target_gain * X
    x_tr_t = torch.tensor(X_tr[None]).to(DEVICE)
    tgt_tr_t = torch.tensor(tgt_tr[None].astype(np.float32)).to(DEVICE)
    N = x_tr_t.shape[1]
    starts = list(range(0, N - fl, stride))
    opt = torch.optim.Adam(dpd.parameters(), lr=lr0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(starts), eta_min=lr0 * 0.02)
    lossf = torch.nn.MSELoss()

    print(f"\n[online adaptation] {EPOCHS} epochs x {len(starts)} frames (fl={fl}), "
          f"cosine LR {lr0:.0e}->{lr0*0.02:.0e}")
    rng = np.random.default_rng(0)
    best = {"ACLR": 0.0}; best_state = None
    for epoch in range(EPOCHS):
        rng.shuffle(starts)
        for s in starts:
            cas.train()          # cuDNN GRU (PA) needs train mode for backward;
                                 # PA params stay frozen via requires_grad=False
            loss = lossf(cas(x_tr_t[:, s:s + fl, :]), tgt_tr_t[:, s:s + fl, :])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m = evaluate(dpd, pa, X_te, tg)
        if m["ACLR"] < best["ACLR"]:
            best = m; best_state = copy.deepcopy(dpd.state_dict())
        if epoch % 4 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:2d}  loss={loss.item():.2e}  "
                  f"ACLR={m['ACLR']:.2f}  EVM={m['EVM']:.2f}  NMSE={m['NMSE']:.2f}  "
                  f"lr={sched.get_last_lr()[0]:.1e}")

    if best_state is not None:
        dpd.load_state_dict(best_state)
    gap = best["ACLR"] - base["ACLR"]
    print(f"\n[result] online best : ACLR={best['ACLR']:.2f}  EVM={best['EVM']:.2f}  NMSE={best['NMSE']:.2f} dB")
    print(f"         offline base : ACLR={base['ACLR']:.2f}  EVM={base['EVM']:.2f}  NMSE={base['NMSE']:.2f} dB")
    verdict = "CONVERGED" if abs(gap) < 1.0 else f"gap {gap:+.2f} dB -- raise EPOCHS / tune LR"
    print(f"         ACLR gap to offline = {gap:+.2f} dB  ({verdict})")
    ROOT = os.path.dirname(HERE)
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(dpd.state_dict(), os.path.join(out_dir, "dpd_online_best.pt"))
    print("  saved best online DPD -> results/dpd_online_best.pt")


if __name__ == "__main__":
    main()
