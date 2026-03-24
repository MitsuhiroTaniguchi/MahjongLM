from __future__ import annotations

import importlib.util
import math
import sys
import types
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to create spec for {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_namespace(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


@lru_cache(maxsize=1)
def load_official_mamba3_siso_deps():
    try:
        from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
        from mamba_ssm.ops.triton.angle_cumsum import angle_dt
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

        return RMSNormGated, angle_dt, mamba3_siso_combined
    except Exception:
        root = Path(__file__).resolve().parents[2] / "external" / "mamba" / "mamba_ssm"
        if not root.is_dir():
            raise ModuleNotFoundError(
                "Official mamba_ssm sources were not found under external/mamba. "
                "Clone https://github.com/state-spaces/mamba first."
            )

        _ensure_namespace("mamba_ssm", root)
        _ensure_namespace("mamba_ssm.ops", root / "ops")
        _ensure_namespace("mamba_ssm.ops.triton", root / "ops" / "triton")
        _ensure_namespace("mamba_ssm.ops.triton.mamba3", root / "ops" / "triton" / "mamba3")

        layernorm_mod = _load_module(
            "mamba_ssm.ops.triton.layernorm_gated",
            root / "ops" / "triton" / "layernorm_gated.py",
        )
        angle_mod = _load_module(
            "mamba_ssm.ops.triton.angle_cumsum",
            root / "ops" / "triton" / "angle_cumsum.py",
        )
        siso_mod = _load_module(
            "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined",
            root / "ops" / "triton" / "mamba3" / "mamba3_siso_combined.py",
        )
        return layernorm_mod.RMSNorm, angle_mod.angle_dt, siso_mod.mamba3_siso_combined


class OfficialMamba3SISO(nn.Module):
    """Training-time SISO Mamba-3 block adapted from the official implementation.

    This keeps the official forward-path structure and defaults, but intentionally omits
    inference-cache / decoding-only paths that rely on extra kernels not needed for
    causal LM training/eval in this codebase.
    """

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        a_floor: float = 1e-4,
        is_outproj_norm: bool = False,
        chunk_size: int = 64,
        layer_idx: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        RMSNormGated, _, _ = load_official_mamba3_siso_deps()
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.a_floor = a_floor
        self.is_outproj_norm = is_outproj_norm
        self.mimo_rank = 1

        self.d_inner = int(self.expand * self.d_model)
        if self.d_inner % self.headdim != 0:
            raise ValueError("Mamba3 d_inner must be divisible by headdim")
        self.nheads = self.d_inner // self.headdim
        self.num_bc_heads = ngroups

        if rope_fraction not in {0.5, 1.0}:
            raise ValueError("Mamba3 rope_fraction must be 0.5 or 1.0")
        self.rotary_dim_divisor = int(2 / rope_fraction)
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        if self.num_rope_angles <= 0:
            raise ValueError("Mamba3 rope_fraction produced zero rope angles")

        d_in_proj = (
            2 * self.d_inner
            + 2 * self.d_state * self.num_bc_heads
            + 3 * self.nheads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)

        _dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)
        self.dt_bias._no_weight_decay = True

        self.B_bias = nn.Parameter(
            1 + torch.zeros((self.nheads, 1, self.d_state), dtype=torch.float32, device=device),
            requires_grad=True,
        )
        self.C_bias = nn.Parameter(
            1 + torch.zeros((self.nheads, 1, self.d_state), dtype=torch.float32, device=device),
            requires_grad=True,
        )

        self.B_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.is_outproj_norm:
            self.norm = RMSNormGated(
                self.d_inner,
                eps=1e-5,
                norm_before_gate=True,
                group_size=self.headdim,
                **factory_kwargs,
            )

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        _, angle_dt, mamba3_siso_combined = load_official_mamba3_siso_deps()
        batch, seqlen, _ = u.shape

        zxBCdtAtrap = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.num_bc_heads,
                self.d_state * self.num_bc_heads,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=1, g=self.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=1, g=self.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")

        a = -F.softplus(dd_A.to(torch.float32))
        a = torch.clamp(a, max=-self.a_floor)
        dt = F.softplus(dd_dt + self.dt_bias)
        adt = a * dt
        dt = rearrange(dt, "b l n -> b n l")
        adt = rearrange(adt, "b l n -> b n l")

        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1)
        angles = angle_dt(angles, dt.transpose(-1, -2))

        B = self.B_norm(B)
        C = self.C_norm(C)

        y = mamba3_siso_combined(
            Q=C.squeeze(2),
            K=B.squeeze(2),
            V=x,
            ADT=adt,
            DT=dt,
            Trap=trap,
            Q_bias=self.C_bias.squeeze(1),
            K_bias=self.B_bias.squeeze(1),
            Angles=angles,
            D=self.D,
            Z=z if not self.is_outproj_norm else None,
            chunk_size=self.chunk_size,
            Input_States=None,
            return_final_states=False,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        if self.is_outproj_norm:
            z = rearrange(z, "b l h p -> b l (h p)")
            y = self.norm(y, z)
        return self.out_proj(y.to(x.dtype))
