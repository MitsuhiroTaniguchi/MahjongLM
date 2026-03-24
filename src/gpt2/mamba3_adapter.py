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


@lru_cache(maxsize=1)
def load_official_mamba3_mimo_deps():
    try:
        from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
        from mamba_ssm.ops.triton.angle_cumsum import angle_dt
        from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo

        return RMSNormGated, angle_dt, mamba3_mimo
    except Exception as exc:
        root = Path(__file__).resolve().parents[2] / "external" / "mamba" / "mamba_ssm"
        if not root.is_dir():
            raise ModuleNotFoundError(
                "Official mamba_ssm sources were not found under external/mamba. "
                "Clone https://github.com/state-spaces/mamba first."
            ) from exc

        _ensure_namespace("mamba_ssm", root)
        _ensure_namespace("mamba_ssm.ops", root / "ops")
        _ensure_namespace("mamba_ssm.ops.triton", root / "ops" / "triton")
        _ensure_namespace("mamba_ssm.ops.triton.mamba3", root / "ops" / "triton" / "mamba3")
        _ensure_namespace("mamba_ssm.ops.tilelang", root / "ops" / "tilelang")
        _ensure_namespace("mamba_ssm.ops.tilelang.mamba3", root / "ops" / "tilelang" / "mamba3")

        layernorm_mod = _load_module(
            "mamba_ssm.ops.triton.layernorm_gated",
            root / "ops" / "triton" / "layernorm_gated.py",
        )
        angle_mod = _load_module(
            "mamba_ssm.ops.triton.angle_cumsum",
            root / "ops" / "triton" / "angle_cumsum.py",
        )
        # These imports require tilelang and a working C++/CUDA build toolchain.
        mimo_mod = _load_module(
            "mamba_ssm.ops.tilelang.mamba3.mamba3_mimo",
            root / "ops" / "tilelang" / "mamba3" / "mamba3_mimo.py",
        )
        return layernorm_mod.RMSNorm, angle_mod.angle_dt, mimo_mod.mamba3_mimo


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
        is_mimo: bool = False,
        mimo_rank: int = 4,
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
        if is_mimo:
            RMSNormGated, _, _ = load_official_mamba3_mimo_deps()
        else:
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
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1

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
            + 2 * self.d_state * self.num_bc_heads * self.mimo_rank
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
            1 + torch.zeros((self.nheads, self.mimo_rank, self.d_state), dtype=torch.float32, device=device),
            requires_grad=True,
        )
        self.C_bias = nn.Parameter(
            1 + torch.zeros((self.nheads, self.mimo_rank, self.d_state), dtype=torch.float32, device=device),
            requires_grad=True,
        )

        self.B_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.is_mimo:
            mimo_x_init_weights = torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank
            mimo_z_init_weights = torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device)
            mimo_o_init_weights = torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank
            self.mimo_x = nn.Parameter(mimo_x_init_weights, requires_grad=True)
            self.mimo_z = nn.Parameter(mimo_z_init_weights, requires_grad=True)
            self.mimo_o = nn.Parameter(mimo_o_init_weights, requires_grad=True)

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
        if self.is_mimo:
            _, angle_dt, mamba3_mimo = load_official_mamba3_mimo_deps()
        else:
            _, angle_dt, mamba3_siso_combined = load_official_mamba3_siso_deps()

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
        B = rearrange(B, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
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

        if self.is_mimo:
            y = mamba3_mimo(
                Q=C,
                K=B,
                V=x,
                ADT=adt,
                DT=dt,
                Trap=trap,
                Q_bias=self.C_bias,
                K_bias=self.B_bias,
                MIMO_V=self.mimo_x,
                MIMO_Z=self.mimo_z,
                MIMO_Out=self.mimo_o if not self.is_outproj_norm else None,
                Angles=angles,
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                rotary_dim_divisor=self.rotary_dim_divisor,
                dtype=x.dtype,
                return_state=False,
            )
            if self.is_outproj_norm:
                z = torch.einsum("blhp,hrp->blrhp", z.float(), self.mimo_z)
                z = rearrange(z, "b l r h p -> b l r (h p)")
                y = rearrange(y, "b l r h p -> b l r (h p)").float()
                y = self.norm(y, z)
                y = rearrange(y, "b l r (h p) -> b l r h p", p=self.headdim)
                y = torch.einsum("blrhp,hrp->blhp", y, self.mimo_o)
            y = rearrange(y, "b l h p -> b l (h p)")
        else:
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
