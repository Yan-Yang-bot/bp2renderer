# tools.py
import torch
from dataclasses import dataclass
from typing import Union


@dataclass
class GSParams:
    Du: Union[torch.Tensor, float] = 0.16
    Dv: Union[torch.Tensor, float] = 0.08
    F:  Union[torch.Tensor, float] = 0.035
    k:  Union[torch.Tensor, float] = 0.065


def laplacian_periodic(x: torch.Tensor) -> torch.Tensor:
    return (
        torch.roll(x,  1, dims=-1) + torch.roll(x, -1, dims=-1) +
        torch.roll(x,  1, dims=-2) + torch.roll(x, -1, dims=-2) -
        4.0 * x
    )


@torch.no_grad()
def init_state_batch(
    B: int, H: int, W: int, device="cuda",
    noise_u=0.02, noise_v=0.02,
    patch_size=10
):
    u = torch.ones(B, 1, H, W, device=device)
    v = torch.zeros(B, 1, H, W, device=device)

    u += noise_u * torch.randn_like(u)
    v += noise_v * torch.randn_like(v)

    hs = H // 2
    ws = W // 2
    ps = patch_size
    u[..., hs-ps:hs+ps, ws-ps:ws+ps] = 0.50
    v[..., hs-ps:hs+ps, ws-ps:ws+ps] = 0.25

    u = u.clamp(0.0, 1.5)
    v = v.clamp(0.0, 1.0)
    return u, v


def _gray_scott_step(u: torch.Tensor, v: torch.Tensor, p: GSParams, dt: float):
    Lu = laplacian_periodic(u)
    Lv = laplacian_periodic(v)

    uvv = u * v * v
    du = p.Du * Lu - uvv + p.F * (1.0 - u)
    dv = p.Dv * Lv + uvv - (p.F + p.k) * v

    u_next = u + dt * du
    v_next = v + dt * dv
    return u_next, v_next


def n_steps(u: torch.Tensor, v: torch.Tensor, p: GSParams, dt: float, n: int):
    for _ in range(n):
        u2, v2 = _gray_scott_step(u, v, p, dt)

    v_max, v_min = v2.amax(), v2.amin()
    if v_max > 1.0 or v_min < 0.0:
        print("Overflow:", v_min, v_max)

    return u2, v2


def debug_freq(it: int, print_more: int = 0) -> bool:
    p = it % 30 == 0 or \
        it > 125 and it % 25 == 0 or \
        it > 200 and it % 10 == 0 or \
        it > 300
    return it != 0 and ((p or it % print_more == 0) if print_more else p)
