from __future__ import annotations

import torch

YOU_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)

_POLAR_EXPRESS_RAW_COEFFICIENTS = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)
_POLAR_EXPRESS_SAFETY_FACTOR = 1.05
POLAR_EXPRESS_COEFFICIENTS = tuple(
    (
        a / _POLAR_EXPRESS_SAFETY_FACTOR,
        b / _POLAR_EXPRESS_SAFETY_FACTOR**3,
        c / _POLAR_EXPRESS_SAFETY_FACTOR**5,
    )
    for (a, b, c) in _POLAR_EXPRESS_RAW_COEFFICIENTS
)


def _normalize_rows(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    return tensor / (tensor.norm(dim=-1, keepdim=True) + eps)


def _normalize_cols(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    return tensor / (tensor.norm(dim=-2, keepdim=True) + eps)


def apply_muonplus_normalization(update: torch.Tensor, *, enabled: bool, eps: float) -> torch.Tensor:
    if not enabled:
        return update
    return _normalize_cols(_normalize_rows(update, eps), eps)


def _standard_newton_schulz_zeropower(grad: torch.Tensor, steps: int) -> torch.Tensor:
    if grad.ndim < 2:
        raise ValueError("Muon orthogonalization requires tensors with ndim >= 2")
    a, b, c = (3.4445, -4.7750, 2.0315)
    update = grad.bfloat16()
    transposed = False
    if update.size(-2) > update.size(-1):
        update = update.mT
        transposed = True
    update = update / (update.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = update @ update.mT
        poly = b * gram + c * gram @ gram
        update = a * update + poly @ update
    if transposed:
        update = update.mT
    return update


def _gram_newton_schulz_zeropower(
    grad: torch.Tensor,
    *,
    coefficients: tuple[tuple[float, float, float], ...],
    reset_iterations: tuple[int, ...],
    eps: float,
) -> torch.Tensor:
    if grad.ndim < 2:
        raise ValueError("Muon orthogonalization requires tensors with ndim >= 2")
    if not coefficients:
        raise ValueError("Gram Newton-Schulz requires at least one coefficient triple")

    original_dtype = grad.dtype
    update = grad.to(torch.float32)
    transposed = False
    if update.size(-2) > update.size(-1):
        update = update.mT
        transposed = True

    update = update / (update.norm(dim=(-2, -1), keepdim=True) + eps)
    iter_dtype = torch.float16 if update.is_cuda else update.dtype
    update = update.to(iter_dtype)

    if update.size(-2) != update.size(-1):
        gram = update @ update.mT
        identity = torch.eye(gram.size(-1), device=gram.device, dtype=update.dtype)
        transport = None
        reset_set = set(reset_iterations)

        for idx, (a, b, c) in enumerate(coefficients):
            if idx in reset_set and idx != 0:
                update = transport @ update
                gram = update @ update.mT
                transport = None

            z_term = (
                torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
                if gram.ndim == 3
                else torch.addmm(gram * b, gram, gram, beta=1.0, alpha=c)
            )
            if transport is None:
                transport = z_term + a * identity
            else:
                transport = torch.addmm(a * transport, transport, z_term, beta=1.0, alpha=1.0)

            if idx < len(coefficients) - 1 and idx + 1 not in reset_set:
                gram_z = torch.addmm(a * gram, gram, z_term, beta=1.0, alpha=1.0)
                gram = torch.addmm(a * gram_z, z_term, gram_z, beta=1.0, alpha=1.0)

        update = transport @ update
    else:
        for a, b, c in coefficients:
            gram = update @ update.mT
            poly = torch.addmm(b * gram, gram, gram, beta=1.0, alpha=c)
            update = torch.addmm(a * update, update, poly, beta=1.0, alpha=1.0)

    if transposed:
        update = update.mT
    return update.to(original_dtype)


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int) -> torch.Tensor:
    """Approximate orthogonalization used by Muon.

    For the common 5-step Muon setting, use Gram Newton-Schulz with the
    Polar Express coefficients and a single restart after iteration 2.
    This is mathematically equivalent to iterating on the polar factor while
    reducing work from the full rectangular matrix to its Gram matrix.
    For other step counts, fall back to the original Newton-Schulz-5 update.
    """
    if steps <= 0:
        raise ValueError("Muon orthogonalization steps must be positive")
    if steps <= len(POLAR_EXPRESS_COEFFICIENTS):
        return _gram_newton_schulz_zeropower(
            grad,
            coefficients=POLAR_EXPRESS_COEFFICIENTS[:steps],
            reset_iterations=(2,) if steps > 2 else (),
            eps=1e-7,
        )
    return _standard_newton_schulz_zeropower(grad, steps=steps)


def muon_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    beta: float,
    ns_steps: int,
    nesterov: bool,
    use_muon_plus: bool,
    norm_eps: float,
) -> torch.Tensor:
    momentum_buffer.lerp_(grad, 1.0 - beta)
    update = grad.lerp(momentum_buffer, beta) if nesterov else momentum_buffer
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update = apply_muonplus_normalization(update, enabled=use_muon_plus, eps=norm_eps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    exp_avg.lerp_(grad, 1.0 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1.0 - betas[1])
    exp_avg_hat = exp_avg / (1.0 - betas[0] ** step)
    exp_avg_sq_hat = exp_avg_sq / (1.0 - betas[1] ** step)
    return exp_avg_hat / (exp_avg_sq_hat.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Single-device Muon optimizer with AdamW-style auxiliary groups."""

    def __init__(self, param_groups: list[dict]):
        normalized_groups: list[dict] = []
        for group in param_groups:
            use_muon = bool(group["use_muon"])
            params = list(group["params"])
            if not params:
                continue
            normalized: dict = {"params": params, "use_muon": use_muon}
            if use_muon:
                normalized["lr"] = group.get("lr", 0.02)
                normalized["momentum"] = group.get("momentum", 0.95)
                normalized["weight_decay"] = group.get("weight_decay", 0.0)
                normalized["nesterov"] = group.get("nesterov", True)
                normalized["ns_steps"] = group.get("ns_steps", 5)
                normalized["use_muon_plus"] = group.get("use_muon_plus", False)
                normalized["norm_eps"] = group.get("norm_eps", 1e-7)
            else:
                normalized["lr"] = group.get("lr", 3e-4)
                normalized["betas"] = group.get("betas", (0.9, 0.95))
                normalized["eps"] = group.get("eps", 1e-10)
                normalized["weight_decay"] = group.get("weight_decay", 0.0)
            normalized_groups.append(normalized)
        super().__init__(normalized_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    update = muon_update(
                        parameter.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                        use_muon_plus=group["use_muon_plus"],
                        norm_eps=group["norm_eps"],
                    )
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(update.reshape(parameter.shape), alpha=-group["lr"])
            else:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        parameter.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        step=state["step"],
                        betas=group["betas"],
                        eps=group["eps"],
                    )
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
        return loss
