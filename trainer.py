# trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from generate_batch import (
    GSParams,
    gray_scott_step,
    init_state_batch,
    generate_gray_scott_target_batch,
)
from radial_power_spectrum_loss import radial_ps_loss


class LearnableGSParams(nn.Module):
    """
    Learnable parameters with constraints:
      Du, Dv > 0 via softplus
      F, k in (0, max_Fk) via sigmoid
    """
    def __init__(self, max_Fk: float = 0.1, init_logD: float = -2.0):
        super().__init__()
        self.log_Du = nn.Parameter(torch.tensor(init_logD))
        self.log_Dv = nn.Parameter(torch.tensor(init_logD))
        self.raw_F  = nn.Parameter(torch.tensor(0.0))
        self.raw_k  = nn.Parameter(torch.tensor(0.0))
        self.max_Fk = float(max_Fk)

    def as_gsparams(self) -> GSParams:
        Du = F.softplus(self.log_Du)          # >0
        Dv = F.softplus(self.log_Dv)          # >0
        Fv = self.max_Fk * torch.sigmoid(self.raw_F)  # (0, max_Fk)
        kv = self.max_Fk * torch.sigmoid(self.raw_k)  # (0, max_Fk)
        return GSParams(Du=float(Du.detach().cpu()),
                        Dv=float(Dv.detach().cpu()),
                        F=float(Fv.detach().cpu()),
                        k=float(kv.detach().cpu()))

    def forward_params_tensor(self):
        # return tensors (for differentiable step)
        Du = F.softplus(self.log_Du)
        Dv = F.softplus(self.log_Dv)
        Fv = self.max_Fk * torch.sigmoid(self.raw_F)
        kv = self.max_Fk * torch.sigmoid(self.raw_k)
        return Du, Dv, Fv, kv


def gray_scott_step_learnable(u, v, Du, Dv, Fv, kv, dt: float):
    # identical dynamics, but takes tensors for params
    from generate_batch import laplacian_periodic
    Lu = laplacian_periodic(u)
    Lv = laplacian_periodic(v)

    uvv = u * v * v
    du = Du * Lu - uvv + Fv * (1.0 - u)
    dv = Dv * Lv + uvv - (Fv + kv) * v

    u_next = u + dt * du
    v_next = v + dt * dv
    return u_next, v_next


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1) Make / load targets (no_grad)
    B_targets = 16
    H = W = 128
    true_params = GSParams(Du=0.16, Dv=0.08, F=0.035, k=0.065)

    v_targets = generate_gray_scott_target_batch(
        B=B_targets, H=H, W=W, dt=1.0,
        params=true_params, device=device,
        max_steps=40000, tol=1e-3, return_uv=False
    ).detach()  # [16,1,H,W]

    # --- 2) Learnable params + optimizer
    model = LearnableGSParams(max_Fk=0.1, init_logD=-2.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    # --- 3) Training hyperparams
    train_steps = 2000
    batch_sim = 4          # how many sims per iter
    unroll_K = 1000         # truncated backprop steps
    dt = 1.0
    alpha = 0.0           # pixel L2 weight
    grad_clip = 1.5

    # Optional: pick targets as batch too
    for it in range(1, train_steps + 1):
        opt.zero_grad(set_to_none=True)

        # sample a batch of targets (distribution matching)
        idx = torch.randint(0, B_targets, (batch_sim,), device=device)
        v_tgt = v_targets[idx]  # [B,1,H,W]

        # random init every iter (matches "real world unknown init")
        u, v = init_state_batch(batch_sim, H, W, device=device)

        # differentiable forward (fixed K steps)
        Du, Dv, Fv, kv = model.forward_params_tensor()
        for _ in range(unroll_K):
            u, v = gray_scott_step_learnable(u, v, Du, Dv, Fv, kv, dt)

        v_pred = v  # [B,1,H,W]

        loss = radial_ps_loss(v_pred, v_tgt) + alpha * ((v_pred - v_tgt) ** 2).mean()
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        opt.step()

        if it % 50 == 0 or it == 1:
            with torch.no_grad():
                Du_, Dv_, F_, k_ = model.forward_params_tensor()
                print(
                    f"it={it:4d} loss={loss.item():.6f} "
                    f"Du={Du_.item():.4f} Dv={Dv_.item():.4f} F={F_.item():.4f} k={k_.item():.4f}"
                )

    print("Done.")
    print("Learned params (approx):", model.as_gsparams())


if __name__ == "__main__":
    main()

