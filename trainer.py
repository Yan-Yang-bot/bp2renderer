# trainer.py
import os
import copy
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools import GSParams, init_state_batch, debug_freq
from power_spectrum_2d import power_spectrum_2d
from rd_generator import RDGenerator


def scale_lr(opt, factor):
    for g in opt.param_groups:
        g["lr"] *= factor

def restore_lr(opt, lrs):
    for i, g in enumerate(opt.param_groups):
        g["lr"] = lrs[i]

class Trainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gen = RDGenerator().to(self.device)
        self.set_default_hyperparams()
    
    def set_default_hyperparams(self):
        # training hyperparams
        self.train_steps = 2000
        self.alpha = 0.0
        self.grad_clip = None
        self.batch_sim = 8
        self.init_lr = 1e-2
        self.weight_decay = 5e-4
        # simulation hyperparams
        self.dt = 1.0
        self.unroll_K = 1000
        self.tol = 1e-3
        self.max_steps = 40000
        # adaptive-lr hyperparams
        self.max_tries = 8
        self.shrink = 0.1
        # target generation hyperparams
        self.B_targets = 512
        self.H = 128
        self.W = 128
        self.true_params = GSParams(Du=0.16, Dv=0.08, F=0.035, k=0.065)

    def make_targets(self):
        param_tag = (
            f"Du{self.true_params.Du}_"
            f"Dv{self.true_params.Dv}_"
            f"F{self.true_params.F}_"
            f"k{self.true_params.k}_"
            f"HW{self.H}x{self.W}_"
            f"B{self.B_targets}"
        )
        target_pt_path = f"targets_GS_{param_tag}.pt"
        
        if os.path.exists(target_pt_path):
            print(f"Loading cached target patterns from {target_pt_path}")
            self.v_targets = torch.load(target_pt_path, map_location=self.device)
        
        else:
            print(f"Generating target patterns using: {self.true_params}.")
            u_t0, v_t0 = init_state_batch(self.B_targets, self.H, self.W, device=self.device)
            print(f"Initial status obtained for a batch of {B_targets}, iterating...")
            self.v_targets = RDGenerator.generate_gray_scott_target_batch(
                u=u_t0, v=v_t0, dt=self.dt,
                params=self.true_params, device=self.device,
                max_steps=40000, tol=1e-4, return_uv=False
            ).detach()  # [B_targets,1,H,W]
            torch.save(self.v_targets.cpu(), target_pt_path)
            print(f"Target patterns generated and saved to {target_pt_path}")

    def train(self):
        # TODO: wrap the adaptive-lr opt in a class.
        opt = torch.optim.SGD(self.gen.parameters(), lr=self.init_lr, momentum=0.0, weight_decay=self.weight_decay)

        print("\33[31mStart training...\33[0m")

        lrs0 = lrs_trial = [g["lr"] for g in opt.param_groups]
        loss = None
        for it in trange(self.train_steps+1, desc="Training iters", leave=True):
            try:
                with torch.no_grad():
                    # These things are used in both the trial forward(s) and the real forward following it/them.
                    u_t, v_t = init_state_batch(self.batch_sim, self.H, self.W, device=self.device)
                    idx_t = torch.randint(0, self.B_targets, (self.batch_sim,), device=self.device)
                    v_t_tgt = self.v_targets[idx_t]
                    _, _, ps_t_tgt = power_spectrum_2d(v_t_tgt, log=True)

                # To make the trial and real forwards use same elements, we put backprop with its tiral forwards before real forward. This necessitates the skipping of the first backprop as there is no loss at this time.
                if loss is not None:
                    # ---------- safe step ----------
                    lrs_trial = lrs0
                    # 1) backup things before repeated trials (because they require param resets)
                    self.gen.backup_params()
                    # 2) get the base gradients (for later linear search of lr)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    #### debug ####
                    mx = 0.0
                    mx_name = None
                    for n,p in self.gen.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            v = p.grad.detach().abs().max().item()
                            if v > mx:
                                mx = v; mx_name = n
                    print(f"[it {it}] max|grad|={mx:.3e} at {mx_name}, lr={opt.param_groups[0]['lr']:.3e}, lr*maxgrad={opt.param_groups[0]['lr']*mx:.3e}")
                    #### end ####
                    opt_state0 = copy.deepcopy(opt.state_dict())

                    print("\33[31mlr trials:\33[0m")
                    for i in range(self.max_tries):
                        print(f"Trial {i+1}...")
                        self.gen.restore_params()
                        opt.load_state_dict(copy.deepcopy(opt_state0))
                        restore_lr(opt, lrs_trial)
                        #### debug ####
                        with torch.no_grad():
                            _, v0, _ = self.gen.simulate_to_steady_trunc_bptt(u=u_t, v=v_t,
                                batch_sim=self.batch_sim,
                                dt=self.dt,
                                K=self.unroll_K,
                                tol=self.tol,
                                max_steps=self.max_steps,
                                device=self.device,
                                return_steps_taken=True)
                            ok0 = torch.isfinite(v0).all().item()
                        print("no-step finite?", ok0)
                        print([g["lr"] for g in opt.param_groups])
                        #### end ####
                        opt.step()
                        with torch.no_grad():
                            _, v_t_pred, t_steps_taken = self.gen.simulate_to_steady_trunc_bptt(
                                u=u_t, v=v_t,
                                batch_sim=self.batch_sim,
                                dt=self.dt,
                                K=self.unroll_K,
                                tol=self.tol,
                                max_steps=self.max_steps,
                                device=self.device,
                                return_steps_taken=True
                            )
                            _, _, ps_t_pred = power_spectrum_2d(v_t_pred, log=True)
                            loss_check = F.mse_loss(ps_t_pred, ps_t_tgt)

                        ok = torch.isfinite(v_t_pred).all().item() and \
                            torch.isfinite(ps_t_pred).all().item() and \
                                torch.isfinite(loss_check).item()
                        # TODO: Maybe also check 0.0 v_pred std, etc., too.
                        print(f"{t_steps_taken}(maximum {self.max_steps}) + {self.unroll_K} steps until converge")
                        print(ok, [g["lr"] for g in opt.param_groups])
                        if ok: break
                        else:
                            print(f"Tried {i+1}.", end=" ")
                            scale_lr(opt, self.shrink)
                            lrs_trial = [g["lr"] for g in opt.param_groups]
                            print(lrs_trial, end=" ")
                            

                    else:
                        raise ValueError("All lrs failed with NaN.")

                    print("\n\33[31mEnd lr trials.\33[0m")
                #### end of trials ####

                u, v = u_t, v_t
        
                # differentiable forward (fixed K steps)
                u_pred, v_pred, steps_taken = self.gen.simulate_to_steady_trunc_bptt(
                    u=u, v=v,
                    batch_sim=self.batch_sim,
                    dt=self.dt,
                    K=self.unroll_K,
                    tol=self.tol,
                    max_steps=self.max_steps,
                    device=self.device,
                    return_steps_taken=True,
                )

                ps_tgt = ps_t_tgt # [B,H,W]
                E, ps_mean, ps_pred = power_spectrum_2d(v_pred, log=True)   # [B,H,W]

                loss = F.mse_loss(ps_pred, ps_tgt)
                #loss += alpha * ((v_pred - v_tgt) ** 2).mean()
                #loss_E = lam_E * (
                #    torch.relu(E_min - E) +
                #    torch.relu(E - E_max)
                #)
                #loss_total = loss + loss_E


                #### debug ####
                if debug_freq(it, print_more=1):

                    pred_std = v_pred.std(dim=(-2, -1))
                    print(f"\n\n\033[32mPrediction stats: {pred_std.min().item():.04e}, {pred_std.median().item():.04e}, {pred_std.max().item():.04e}")
                    print(f"Power spectrum: {E.item():.4e}, w/ log: {ps_mean:.4e}\033[0m\n")

                    g_v = torch.autograd.grad(loss, v_pred, retain_graph=True, allow_unused=True)[0]
                    print("dL/dv_pred:", "None\n" if g_v is None else f"({g_v.abs().mean().item():.4e}, {g_v.abs().max().item():.4e})\n")

                    print("\033[36mBefore backprop (last grads)\033[0m", f"grad log_Du: {self.gen.log_Du.grad.abs().mean().item():.4e}",
                        f"grad log_Dv: {self.gen.log_Dv.grad.abs().mean().item():.4e}",
                        f"grad raw_F : {self.gen.raw_F.grad.abs().mean().item():.4e}",
                        f"grad raw_k : {self.gen.raw_k.grad.abs().mean().item():.4e}")
                    with torch.no_grad():
                        p = self.gen.params_tensor()
                        print(
                            f"it={it:4d} loss={loss.item():.6f}",   # loss_total={loss_total.item():.6f}",
                            f"Du={p.Du.item():.4f} Dv={p.Dv.item():.4f} F={p.F.item():.4f} k={p.k.item():.4f}"
                        )
                #### end ####

            except KeyboardInterrupt:
                done = False
                break
            
        else:
            done = True
        with torch.no_grad():
            p = self.gen.params_tensor()
            print("\33[31mDone.\nLearned params (tensor):\33[0m" if done else f"\33[31mStep {it} aborted.\nShowing params at step {it-1}:\33[0m")
            print("Du=", float(p.Du.cpu()),
                "Dv=", float(p.Dv.cpu()),
                "F=", float(p.F.cpu()),
                "k=", float(p.k.cpu()))
            if not done:
                print(f"To restore training, use parameters log_Du={self.gen.log_Du.item()}, log_Dv={self.gen.log_Dv.item()}, raw_F={self.gen.raw_F.item()}, raw_k={self.gen.raw_k.item()}.")

    def test(self):
        pass

    def save_model(self, path: str):
        pass

    def load_model(self, path: str):
        pass


if __name__ == "__main__":
    trainer = Trainer()
    trainer.make_targets()
    trainer.train()

