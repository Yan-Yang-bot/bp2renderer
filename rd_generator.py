# rd_generator.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional
from tqdm import trange

from tools import GSParams, n_steps, py_float64


class RDGenerator(nn.Module):
    def __init__(self, params: Optional[GSParams] = None, max_Fk: float = 0.1, init_logD: float = -2.0):
        super().__init__()
        if params is not None:
            init_logDu = math.log(math.exp(py_float64(params.Du)) - 1)
            init_logDv = math.log(math.exp(py_float64(params.Dv)) - 1)
            raw_F = math.log(py_float64(params.F) / (0.1 - py_float64(params.F)))
            raw_k = math.log(py_float64(params.k) / (0.1 - py_float64(params.k)))
            # print(init_logDu, init_logDv, raw_F, raw_k)
        else:
            init_logDu = init_logDv = init_logD
            raw_F = raw_k = 0.0
        self.log_Du = nn.Parameter(torch.tensor(init_logDu))
        self.log_Dv = nn.Parameter(torch.tensor(init_logDv))
        self.raw_F  = nn.Parameter(torch.tensor(raw_F))
        self.raw_k  = nn.Parameter(torch.tensor(raw_k))
        self.max_Fk = float(max_Fk)
        self.backup = {n: p.detach().clone() for n, p in self.named_parameters() if p.requires_grad}

    def backup_params(self):  # always overwrite
        self.backup = {n: p.detach().clone() for n, p in self.named_parameters() if p.requires_grad}

    def restore_params(self):
        with torch.no_grad():
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.copy_(self.backup[n])

    def params_tensor(self) -> GSParams:
        Du = F.softplus(self.log_Du)                  # >0
        Dv = F.softplus(self.log_Dv)                  # >0
        Fv = self.max_Fk * torch.sigmoid(self.raw_F)  # (0, max_Fk)
        kv = self.max_Fk * torch.sigmoid(self.raw_k)  # (0, max_Fk)
        return GSParams(Du=Du, Dv=Dv, F=Fv, k=kv)

    @staticmethod
    @torch.no_grad()
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
        shared simulation core: each sample converges and freezes separately.
        used in target generation (no grad) and training warmup (no grad).

        u,v: [B,1,H,W]
        p: GSParams (fields can be Tensor or float)
        """
        B = u.shape[0]
        converged = torch.zeros(B, device=u.device, dtype=torch.bool, requires_grad=False)
        active_idx = torch.arange(B, device=u.device, requires_grad=False)  # indices of not-yet-converged

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
            u_next_a, v_next_a, overflow = n_steps(u_a, v_a, p, dt, n=1)

            if overflow:
                steps_taken = t + 1
                break

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

    @staticmethod
    def simulate_constant_steps(
        u: torch.Tensor,
        v: torch.Tensor,
        p: GSParams,
        num_steps: int,
        dt: float = 1.0,
        disable_progress_bar: bool = False,
    ):
        """
        u,v: [B,1,H,W]
        p: GSParams (fields can be Tensor or float)
        return: u,v,overflow
        """
        overflow = False
        for _ in trange(num_steps,
                        desc=f"Simulating for {num_steps} steps",
                        leave=True,
                        disable=disable_progress_bar):

            u, v, of = n_steps(u, v, p, dt, n=1)
            overflow = overflow or of

        return u, v, overflow

    # -----------------------------
    # 1) Target Generation
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
        dev = torch.device(device)
        u = u.to(dev)
        v = v.to(dev)
        u, v, converged, nstep = RDGenerator._simulate_until_converged(
            u=u, v=v, p=params, dt=dt, tol=tol, max_steps=max_steps
        )
        if not torch.all(converged):
            raise ValueError(f"Your parameters {params} led to NaN/Inf "
                             f"or physically non-meaningful values (outside [0,1]).")
        print(f"Generation finished - maximum {nstep} steps.")

        if return_uv:
            return u, v
        return v  # [B,1,H,W]

    # ----------------------------------------
    # 2) TBPTT：warmup until close to stable status + store gradient for the subsequent K steps.
    # ----------------------------------------
    def simulate_to_steady_trunc_bptt(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        dt: float = 1.0,
        K: int = 3000,
        tol: float = 2e-4,
        max_steps: int = 40000,
        device: Union[str, torch.device] = "cuda",
        return_steps_taken: bool = True,
        disable_progress_bar: bool =True
    ):
        """
        Input u,v，run with no_grad until almost converge (each sample's convergence judged independently)
        Then detach and run K steps with grad that later backpropagate to parameters.
        """
        dev = torch.device(device)
        u = u.to(dev)
        v = v.to(dev)

        # Phase 1: run-to-steady without graph
        p = self.params_tensor()
        # with torch.no_grad():
        #     u, v, converged, steps_taken = self._simulate_until_converged(
        #         u=u, v=v, p=p, dt=dt, tol=tol, max_steps=max_steps, disable_progress_bar=True
        #     )
        # TODO: Truncated BPTT is temporarily disabled, using full BPTT now.
        #  Enable it (above commented code) when necessary.
        u, v, converged, steps_taken = self._simulate_until_converged(
            u=u, v=v, p=p, dt=dt, tol=tol, max_steps=max_steps, disable_progress_bar=disable_progress_bar
        )

        overflow_early = not torch.all(converged)

        # Phase 2: truncated BPTT window with graph
        # TODO: Truncated BPTT is temporarily disabled, using full BPTT now.
        #  Enable it (below detach & extra steps code, and increase tol passed into here) when necessary.
        # u = u.detach()
        # v = v.detach()
        # u, v, overflow_late = n_steps(u, v, p, dt, K)

        overflow = overflow_early #or overflow_late

        if return_steps_taken:
            return u, v, overflow, steps_taken
        return u, v, overflow

