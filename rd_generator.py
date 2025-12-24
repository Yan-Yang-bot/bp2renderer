# rd_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from tqdm import trange

from tools import GSParams, gray_scott_step


class RDGenerator(nn.Module):
    def __init__(self, max_Fk: float = 0.1, init_logD: float = -2.0):
        super().__init__()
        self.log_Du = nn.Parameter(torch.tensor(init_logD))
        self.log_Dv = nn.Parameter(torch.tensor(init_logD))
        self.raw_F  = nn.Parameter(torch.tensor(0.0))
        self.raw_k  = nn.Parameter(torch.tensor(0.0))
        self.max_Fk = float(max_Fk)

    def params_tensor(self) -> GSParams:
        Du = F.softplus(self.log_Du)                  # >0
        Dv = F.softplus(self.log_Dv)                  # >0
        Fv = self.max_Fk * torch.sigmoid(self.raw_F)  # (0, max_Fk)
        kv = self.max_Fk * torch.sigmoid(self.raw_k)  # (0, max_Fk)
        return GSParams(Du=Du, Dv=Dv, F=Fv, k=kv)

    @staticmethod
    def _per_sample_converged(u_next, v_next, u, v, tol: float) -> torch.Tensor:
        """
        returns: [B] bool
        """
        du = (u_next - u).abs().amax(dim=(-3, -2, -1))
        dv = (v_next - v).abs().amax(dim=(-3, -2, -1))
        delta = torch.maximum(du, dv)
        return delta < tol

    @staticmethod
    def _simulate_until_converged(
        u: torch.Tensor,
        v: torch.Tensor,
        p: GSParams,
        dt: float,
        tol: float,
        max_steps: int,
        disable_progress_bar: bool = False,
    ):
        """
        公用 simulate 核心：按样本判断收敛 + freeze。
        既可用于 target 生成（no_grad 环境），也可用于训练 warmup（no_grad 环境）。

        u,v: [B,1,H,W]
        p: GSParams (fields can be Tensor or float)
        """
        B = u.shape[0]
        converged = torch.zeros(B, device=u.device, dtype=torch.bool)
        active_idx = torch.arange(B, device=u.device)  # indices of not-yet-converged

        steps_taken = 0
        for t in trange(max_steps,
                        desc=f"Simulating to steady state th={tol}",
                        leave=True,
                        disable=disable_progress_bar):
            if active_idx.numel() == 0:
                steps_taken = t
                break

            # slice active samples only
            u_a = u.index_select(0, active_idx)
            v_a = v.index_select(0, active_idx)
            u_next_a, v_next_a = gray_scott_step(u_a, v_a, p, dt)

            newly_a = RDGenerator._per_sample_converged(u_next_a, v_next_a, u_a, v_a, tol)
            # write back updated states for active samples
            u[active_idx] = u_next_a
            v[active_idx] = v_next_a
            # mark converged (global)
            converged[active_idx] |= newly_a

            # shrink active set
            still_active_mask = ~converged[active_idx]
            active_idx = active_idx[still_active_mask]

        else:
            steps_taken = t + 1

        return u, v, converged, steps_taken

    # -----------------------------
    # 1) 作为成员函数的 target 生成
    # -----------------------------
    @staticmethod
    @torch.no_grad()
    def generate_gray_scott_target_batch(
        u: torch.Tensor,
        v: torch.Tensor,
        dt: float = 1.0,
        params: GSParams = GSParams(),
        device: Union[str, torch.device] = "cuda",
        max_steps: int = 400000,
        tol: float = 1e-3,
        return_uv: bool = False,
    ):
        """
        你给的 generate_gray_scott_target_batch：每样本独立停。
        放到 RDGenerator 里，方便统一接口。
        """
        dev = torch.device(device)
        u = u.to(device)
        v = v.to(device)
        u, v, _, nstep = RDGenerator._simulate_until_converged(
            u=u, v=v, p=params, dt=dt, tol=tol, max_steps=max_steps
        )
        print(f"Generation finished - maximum {nstep} steps.")

        if return_uv:
            return u, v
        return v  # [B,1,H,W]

    # ----------------------------------------
    # 2) TBPTT：warmup 到稳态 + 最后 K 步带图
    # ----------------------------------------
    def simulate_to_steady_trunc_bptt(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        batch_sim: int,
        dt: float = 1.0,
        K: int = 3000,
        tol: float = 2e-4,
        max_steps: int = 40000,
        device: Union[str, torch.device] = "cuda",
        return_steps_taken: bool = True,
    ):
        """
        输入 u,v，先 no_grad 跑到稳态（每样本独立停+freeze），
        再 detach 状态，最后 K 步带图，反传到参数。
        """
        dev = torch.device(device)
        u = u.to(dev)
        v = v.to(dev)

        # Phase 1: run-to-steady without graph
        p = self.params_tensor()
        with torch.no_grad():
            u, v, _, steps_taken = self._simulate_until_converged(
                u=u, v=v, p=p, dt=dt, tol=tol, max_steps=max_steps, disable_progress_bar=True
            )

        # Phase 2: truncated BPTT window with graph
        u = u.detach()
        v = v.detach()

        p = self.params_tensor()  # same underlying Parameters, now used in grad mode

        #### debug ####
        #print("Phase2 u.requires_grad:", u.requires_grad, "v.requires_grad:", v.requires_grad)
        #print("Phase2 p types:", type(p.Du), type(p.F))
        #if isinstance(p.Du, torch.Tensor):
        #    print("Phase2 p.Du.requires_grad:", p.Du.requires_grad)
        #if isinstance(p.F, torch.Tensor):
        #    print("Phase2 p.F.requires_grad:", p.F.requires_grad)
        #### end ####
        for _ in range(K):
            u, v = gray_scott_step(u, v, p, dt)

        if return_steps_taken:
            return u, v, steps_taken
        return u, v

