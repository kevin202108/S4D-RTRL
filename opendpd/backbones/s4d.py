"""
S4D backbone for OpenDPD  (complex diagonal State-Space Model + per-step FFN).

Author: Kai-Chun Fan.  License: Apache-2.0.  Part of S4D-RTRL, an extension to
OpenDPD (https://github.com/lab-emi/OpenDPD, Apache-2.0).

Self-contained drop-in following OpenDPD's backbone contract:
    forward(x, h_0) :  x = [B, T, 2] (I/Q)  ->  out = [B, T, 2] (I/Q)

Design (the "M3" recipe from the Complex-S4D-DPD param study):
  * input feature = only the instantaneous envelope basis  [z, z*|z|^2]
    (NO hand-crafted high-order powers; mem_depth=1 -> the SSM carries the memory)
  * 2 layers of:  residual complex S4D (linear memory) + complex FFN (nonlinearity)
  * native-complex throughout; output projected to a single complex tap -> (I,Q)

Key finding it embodies: the SSM handles memory cheaply, a small FFN replaces the
polynomial nonlinearity, so ~4k params match a 28k baseline. Map model size via
`hidden_size` (=H); state size N defaults to H//2.

Recommended OpenDPD args to reproduce M3:
    --DPD_hidden_size 16 --DPD_num_layers 2     (or PA_* for PA modeling)
"""
import math
from typing import Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Complex building blocks
# --------------------------------------------------------------------------- #
class ComplexPointwise(nn.Module):
    """Complex 1x1 conv (per-time-step FC). [B, C_in, T] -> [B, C_out, T]."""
    def __init__(self, in_ch: int, out_ch: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, dtype=torch.cfloat) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_ch, dtype=torch.cfloat)) if bias else None

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.copy_(torch.randn_like(self.weight) * 0.05)
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.einsum("bct,oc->bot", x, self.weight)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1)
        return y


class ModReLU(nn.Module):
    """y = ReLU(|z| + b) * z / |z|  -- gates magnitude, preserves phase."""
    def __init__(self, num_features: int, eps: float = 1e-12, init_bias: float = -0.05):
        super().__init__()
        self.init_bias = float(init_bias)
        self.b = nn.Parameter(torch.full((num_features,), float(init_bias)))
        self.eps = eps

    def reset_parameters(self):
        with torch.no_grad():
            self.b.fill_(self.init_bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r = z.abs()
        gate = torch.relu(r + self.b.view(1, -1, 1))
        return gate * z / (r + self.eps)


class ComplexSplitReLU(nn.Module):
    """ReLU on real and imag separately. Unlike ModReLU this is NOT phase-preserving,
    so it can realise amplitude-dependent phase rotation (AM-PM) -- needed for DPD EVM."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(torch.relu(z.real), torch.relu(z.imag))


def make_activation(name: str, num_features: int) -> nn.Module:
    name = (name or "crelu").lower()
    if name == "modrelu":
        return ModReLU(num_features)
    if name in ("crelu", "splitrelu"):
        return ComplexSplitReLU()
    if name == "none":
        return nn.Identity()
    raise ValueError(f"unknown activation '{name}'")


class ComplexFFN(nn.Module):
    """Per-time-step complex MLP: H -> mult*H -> H. The activation is the learned nonlinearity."""
    def __init__(self, h: int, mult: float = 2.0, activation: str = "crelu"):
        super().__init__()
        hidden = max(1, int(round(mult * h)))
        self.fc1 = ComplexPointwise(h, hidden)
        self.act = make_activation(activation, hidden)
        self.fc2 = ComplexPointwise(hidden, h)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(z)))

    def reset_parameters(self):
        self.fc1.reset_parameters()
        if hasattr(self.act, "reset_parameters"):
            self.act.reset_parameters()
        self.fc2.reset_parameters()


# --------------------------------------------------------------------------- #
#  Complex S4D kernel + block (FFT-conv form, trained in parallel over time)
# --------------------------------------------------------------------------- #
class S4DKernel(nn.Module):
    def __init__(self, d_model: int, n_ssm: int, dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        H, N = int(d_model), int(n_ssm)
        self._H, self._N = H, N
        self.dt_min, self.dt_max = float(dt_min), float(dt_max)
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        C = torch.randn(H, N, dtype=torch.cfloat) * 0.05
        self.C = nn.Parameter(torch.view_as_real(C))           # (H, N, 2)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(H, N)))
        w = torch.linspace(-math.pi, math.pi, steps=N)
        self.A_imag = nn.Parameter(w.unsqueeze(0).repeat(H, 1))  # (H, N)

    def reset_parameters(self):
        """Re-draw all kernel parameters from the current torch RNG state, replicating
        the constructor's init exactly (same draw order/shapes/dtypes), so that
        `torch.manual_seed(s); kernel.reset_parameters()` == fresh construction under seed s."""
        H, N = self._H, self._N
        with torch.no_grad():
            log_dt = torch.rand(H, device=self.log_dt.device) \
                * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
            self.log_dt.copy_(log_dt)
            C = torch.randn(H, N, dtype=torch.cfloat, device=self.C.device) * 0.05
            self.C.copy_(torch.view_as_real(C))
            self.log_A_real.fill_(math.log(0.5))
            w = torch.linspace(-math.pi, math.pi, steps=N, device=self.A_imag.device)
            self.A_imag.copy_(w.unsqueeze(0).repeat(H, 1))

    def forward(self, L: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt)                             # (H,)
        C = torch.view_as_complex(self.C)                      # (H, N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag     # (H, N)
        dtA = A * dt.unsqueeze(-1)                             # (H, N)
        t = torch.arange(L, device=dtA.device, dtype=dtA.real.dtype)
        C_eff = C * torch.expm1(dtA) / A
        K = torch.einsum("hn,hnl->hl", C_eff, torch.exp(dtA.unsqueeze(-1) * t))
        return K                                               # (H, L) complex

    def discrete(self):
        """Discrete-time SSM params for the recurrent form: state s_t = Abar*s_{t-1}+u_t,
        y_t = sum_n C_eff*s_t.  Same math as the conv kernel (K[t]=sum_n C_eff*Abar^t)."""
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dtA = A * dt.unsqueeze(-1)
        Abar = torch.exp(dtA)                                  # (H, N) complex
        C_eff = C * torch.expm1(dtA) / A                       # (H, N) complex
        return Abar, C_eff


class S4D(nn.Module):
    """One complex S4D layer: FFT-conv + complex D skip + activation + channel mixing."""
    def __init__(self, d_model: int, n_ssm: int, activation: str = "crelu", use_d: bool = True):
        super().__init__()
        self.h = int(d_model)
        self.use_d = bool(use_d)
        if self.use_d:
            self.D = nn.Parameter(torch.randn(self.h, dtype=torch.cfloat) * 0.05)
        self.kernel = S4DKernel(self.h, n_ssm)
        self.activation = make_activation(activation, self.h)
        self.output_linear = ComplexPointwise(self.h, self.h)

    def reset_parameters(self):
        if self.use_d:
            with torch.no_grad():
                self.D.copy_(torch.randn_like(self.D) * 0.05)
        self.kernel.reset_parameters()
        if hasattr(self.activation, "reset_parameters"):
            self.activation.reset_parameters()
        self.output_linear.reset_parameters()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, H, L) complex
        B, H, L = u.shape
        k = self.kernel(L)                                     # (H, L)
        nfft = 2 * L
        k_f = torch.fft.fft(k, n=nfft)
        u_f = torch.fft.fft(u, n=nfft)
        y = torch.fft.ifft(u_f * k_f.unsqueeze(0), n=nfft)[..., :L]
        if self.use_d:
            y = y + u * self.D.view(1, -1, 1)
        y = self.activation(y)
        y = self.output_linear(y)
        return y

    @torch.no_grad()
    def forward_recurrent(self, u: torch.Tensor) -> torch.Tensor:
        """Streaming/inference form: sample-by-sample diagonal recurrence.
        Mathematically identical to forward() (same params), O(1) state per step,
        no FFT -- the form used for real-time DPD on hardware.
            s_t = Abar (x) s_{t-1} + u_t ;  y_t = sum_n C_eff (x) s_t
        """
        B, H, L = u.shape
        Abar, C_eff = self.kernel.discrete()                   # (H, N) complex
        s = torch.zeros(B, H, Abar.shape[1], dtype=u.dtype, device=u.device)
        ys = []
        for t in range(L):
            s = Abar.unsqueeze(0) * s + u[:, :, t].unsqueeze(-1)   # (B, H, N)
            ys.append((C_eff.unsqueeze(0) * s).sum(-1))           # (B, H)
        y = torch.stack(ys, dim=-1)                               # (B, H, L)
        if self.use_d:
            y = y + u * self.D.view(1, -1, 1)
        y = self.activation(y)
        y = self.output_linear(y)
        return y


# --------------------------------------------------------------------------- #
#  OpenDPD backbone
# --------------------------------------------------------------------------- #
class S4D_DPD(nn.Module):
    def __init__(self, hidden_size: int, output_size: int = 2, num_layers: int = 2,
                 d_state: Optional[int] = None, degrees=(2,), mem_depth: int = 1,
                 ffn_mult: float = 2.0, residual: bool = True, lin_skip: bool = True,
                 activation: str = "crelu", use_d: bool = True, phase_feats: bool = False,
                 bias: bool = True, **kwargs):
        super().__init__()
        H = int(hidden_size)
        N = int(d_state) if d_state else max(2, H // 2)
        self.degrees = tuple(int(d) for d in degrees)
        self.mem_depth = int(mem_depth)
        self.residual = bool(residual)
        self.phase_feats = bool(phase_feats)
        per_tap = 1 + len(self.degrees) + (1 if self.phase_feats else 0)
        in_ch = self.mem_depth * per_tap                       # complex channels

        self.input_proj = ComplexPointwise(in_ch, H)
        self.layers = nn.ModuleList(S4D(H, N, activation=activation, use_d=use_d) for _ in range(num_layers))
        self.ffns = nn.ModuleList(ComplexFFN(H, ffn_mult, activation=activation) for _ in range(num_layers))
        self.output_proj = ComplexPointwise(H, 1)
        # End-to-end linear passthrough: a clean linear map from the raw envelope
        # features straight to the output, OUTSIDE every nonlinearity. Lets the net
        # represent identity trivially -> preserves in-band gain/phase (low EVM).
        # This is the DPD analogue of DGRU concatenating its input into fc_out, and
        # is distinct from the S4D `D` term (which lives inside the block, before the
        # activation / mixing / FFN, so it is not an end-to-end linear anchor).
        self.skip = ComplexPointwise(in_ch, 1) if lin_skip else None

        # Anchor the linear passthrough near identity (unity on the raw-z channel) so
        # the net starts close to "DPD = identity". The SSM/FFN branch keeps its normal
        # init so its gradients stay alive (do NOT zero output_proj -- that freezes it).
        if self.skip is not None:
            with torch.no_grad():
                self.skip.weight.zero_()
                self.skip.weight[0, 0] = 1.0
                if self.skip.bias is not None:
                    self.skip.bias.zero_()

    def _features(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, T) complex -> (B, C, T) complex envelope-polynomial features
        amp2 = (z.real ** 2 + z.imag ** 2)                     # (B, T) real
        cols = []
        for m in range(self.mem_depth):
            zm = z if m == 0 else torch.roll(z, shifts=m, dims=-1)
            am2 = amp2 if m == 0 else torch.roll(amp2, shifts=m, dims=-1)
            if m > 0:
                zm = zm.clone(); zm[..., :m] = 0
                am2 = am2.clone(); am2[..., :m] = 0
            cols.append(zm)
            for p in self.degrees:                             # even p -> amp2**(p/2)
                cols.append(zm * (am2 ** (p / 2.0)))
            if self.phase_feats:                               # unit phasor ~ (cos + j sin)
                cols.append(zm / (am2.sqrt() + 1e-12))
        return torch.stack(cols, dim=1)                        # (B, C, T)

    def forward(self, x: torch.Tensor, h_0=None) -> torch.Tensor:
        # x: (B, T, 2) real -> out: (B, T, 2) real
        z = torch.complex(x[..., 0], x[..., 1])                # (B, T)
        feats = self._features(z)                              # (B, C, T)
        h = self.input_proj(feats)                             # (B, H, T)
        for layer, ffn in zip(self.layers, self.ffns):
            y = layer(h)
            h = h + y if self.residual else y
            h = h + ffn(h)
        out = self.output_proj(h)                              # (B, 1, T) complex
        if self.skip is not None:
            out = out + self.skip(feats)                       # linear passthrough
        out = out[:, 0, :]                                     # (B, T) complex
        return torch.stack([out.real, out.imag], dim=-1)       # (B, T, 2)

    @torch.no_grad()
    def forward_recurrent(self, x: torch.Tensor, h_0=None) -> torch.Tensor:
        """Streaming inference: identical output to forward(), but each S4D layer runs
        its O(1) diagonal recurrence instead of FFT-conv. All other layers are pointwise."""
        z = torch.complex(x[..., 0], x[..., 1])
        feats = self._features(z)
        h = self.input_proj(feats)
        for layer, ffn in zip(self.layers, self.ffns):
            y = layer.forward_recurrent(h)
            h = h + y if self.residual else y
            h = h + ffn(h)
        out = self.output_proj(h)
        if self.skip is not None:
            out = out + self.skip(feats)
        out = out[:, 0, :]
        return torch.stack([out.real, out.imag], dim=-1)

    def reset_parameters(self):
        """Fully re-initialise all parameters from the current torch RNG state.
        Mirrors the constructor's init (same draw order), so multi-seed runs via
        `torch.manual_seed(s); net.reset_parameters()` get genuinely different,
        per-seed-reproducible initialisations (J3c caveat #2 fix)."""
        self.input_proj.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        for ffn in self.ffns:
            ffn.reset_parameters()
        self.output_proj.reset_parameters()
        if self.skip is not None:
            # re-anchor the linear passthrough at identity (as in the constructor)
            with torch.no_grad():
                self.skip.weight.zero_()
                self.skip.weight[0, 0] = 1.0
                if self.skip.bias is not None:
                    self.skip.bias.zero_()
