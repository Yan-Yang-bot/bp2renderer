# generate_gray_scott_batch.py
import torch
from dataclasses import dataclass

@dataclass
class GSParams:
    Du: float = 0.16
    Dv: float = 0.08
    F:  float = 0.035
    k:  float = 0.065

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

def gray_scott_step(u: torch.Tensor, v: torch.Tensor, p: GSParams, dt: float):
    Lu = laplacian_periodic(u)
    Lv = laplacian_periodic(v)

    uvv = u * v * v
    du = p.Du * Lu - uvv + p.F * (1.0 - u)
    dv = p.Dv * Lv + uvv - (p.F + p.k) * v

    u_next = u + dt * du
    v_next = v + dt * dv
    return u_next, v_next

@torch.no_grad()
def generate_gray_scott_target_batch(
    B: int,
    H: int = 128,
    W: int = 128,
    dt: float = 1.0,
    params: GSParams = GSParams(),
    device: str = "cuda",
    max_steps: int = 400000,
    tol: float = 1e-3,
    return_uv: bool = False,
):
    """
    Generate B target patterns with fixed params, each sample stops independently.
    Stopping criterion: per-sample max(|Δu|,|Δv|) < tol
    """
    u, v = init_state_batch(B, H, W, device=device)

    converged = torch.zeros(B, device=device, dtype=torch.bool)

    for _ in range(max_steps):
        u_next, v_next = gray_scott_step(u, v, params, dt)

        # per-sample max abs delta
        du = (u_next - u).abs().amax(dim=(-3, -2, -1))  # [B]
        dv = (v_next - v).abs().amax(dim=(-3, -2, -1))  # [B]
        delta = torch.maximum(du, dv)                    # [B]

        newly = delta < tol
        converged = converged | newly

        # freeze converged samples: only update those not converged
        mask = (~converged).view(B, 1, 1, 1).to(u.dtype)
        u = u + mask * (u_next - u)
        v = v + mask * (v_next - v)

        if torch.all(converged):
            break

    if return_uv:
        return u, v
    return v  # [B,1,H,W]

# Example usage:
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device checking done.')
    params = GSParams(Du=0.2267, Dv=0.1087, F=0.0409, k=0.0498)
    print('parameters generated.')
    patterns = []

    print('Generating...')
    for i in range(4):
        v = generate_gray_scott_target_batch(
            B=1, H=128, W=128,
            dt=1.0,
            params=params,
            device=device
        )
        # v: [1, 1, 128, 128]
        patterns.append(v[0, 0].detach().cpu())
        print(str(i+1))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    for ax, img in zip(axes.flatten(), patterns):
        ax.imshow(img, cmap='viridis')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

