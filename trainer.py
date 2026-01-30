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


def load_lr(opt, lrs):
    for i, g in enumerate(opt.param_groups):
        g["lr"] = lrs[i]


class Trainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gen = RDGenerator().to(self.device)
        self.set_default_hyperparams()
        self.loss = None
        self.opt_state0 = None
        # TODO: wrap the adaptive-lr opt in a class.
        self.opt = torch.optim.SGD(self.gen.parameters(), lr=self.init_lr, momentum=0.0, weight_decay=self.weight_decay)
        self.lrs0 = [g["lr"] for g in self.opt.param_groups]
        self.lrs_trial = self.lrs0
        self.v_targets = self.make_targets()
    
    def set_default_hyperparams(self):
        # training hyperparams
        self.train_steps = 2000
        self.alpha = 0.0
        self.grad_clip = None
        self.batch_sim = 8
        self.init_lr = 1.2e-2
        self.weight_decay = 5e-4
        # simulation hyperparams
        self.dt = 1.0
        self.unroll_K = 1000
        self.tol = 5e-5# float('inf') #1e-3
        self.max_steps = 40000
        # adaptive-lr hyperparams
        self.max_tries = 4
        self.shrink = 0.1
        self.jump = 1.2
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
            v_targets = torch.load(target_pt_path, map_location=self.device)
        
        else:
            print(f"Generating target patterns using: {self.true_params}.")
            u_t0, v_t0 = init_state_batch(self.B_targets, self.H, self.W, device=self.device)
            print(f"Initial status obtained for a batch of {self.B_targets}, iterating...")
            v_targets = RDGenerator.generate_gray_scott_target_batch(
                u=u_t0, v=v_t0, dt=self.dt,
                params=self.true_params, device=self.device,
                max_steps=40000, tol=1e-4, return_uv=False
            ).detach()  # [B_targets,1,H,W]
            torch.save(v_targets.cpu(), target_pt_path)
            print(f"Target patterns generated and saved to {target_pt_path}")

        return v_targets

    def step_and_trial(self, i, shrink=True):
        """
        Restore the starting point of all trials of this iteration,
        shrink or amplify learning rate,
        step the optimizer to get the new parameters,
        then trial forward (no grad) to see
        if current learning rate is viable (producing v without NaN/Inf/^[0,1]).
        - i (Integer): index of the current lr value, logging purpose only
        - shrink (Boolean): if this round shrink the lr (when False, amplify lr using `self.jump` instead)
        - Return (Boolean): if current learning rate can be used in real forward.
        """
        print(f"Trying learning rate No.{i+1} {self.lrs_trial}...")
        self.gen.restore_params()
        self.opt.load_state_dict(copy.deepcopy(self.opt_state0))
        load_lr(self.opt, self.lrs_trial)

        self.opt.step()
        print("Parameters updated:")
        p = self.gen.params_tensor()
        print(f"Du={p.Du.item():.4f} Dv={p.Dv.item():.4f} F={p.F.item():.4f} k={p.k.item():.4f}")

        with torch.no_grad():
            u_tmp, v_tmp = self.u_t.clone(), self.v_t.clone()
            _, v_t_pred, t_steps_taken = self.gen.simulate_to_steady_trunc_bptt(
                u=u_tmp, v=v_tmp,
                dt=self.dt,
                K=self.unroll_K,
                tol=self.tol,
                max_steps=self.max_steps,
                device=self.device,
                return_steps_taken=True
            )
            del u_tmp, v_tmp
            torch.cuda.empty_cache()
            _, _, ps_t_pred = power_spectrum_2d(v_t_pred, log=True)
            loss_check = F.mse_loss(ps_t_pred, self.ps_t_target)

        ok = torch.isfinite(v_t_pred).all().item() and v_t_pred.min() >= 0 and v_t_pred.max() <= 1 and \
            torch.isfinite(ps_t_pred).all().item() and \
            torch.isfinite(loss_check).item()
        print(f"Trial forward done. Real forward will take {t_steps_taken} (maximum {self.max_steps}) "
              f"steps without grad + {self.unroll_K} steps with grad to converge.")
        print("All finite and in [0,1]?", ok)
        if ok:
            return True
        else:
            print(f"Done trying No.{i + 1} {self.lrs_trial}.", end=" ")
            scale_lr(self.opt, self.shrink if shrink else self.jump)
            self.lrs_trial = [g["lr"] for g in self.opt.param_groups]
            print("After scale, next round lr:", self.lrs_trial)
            return False

    def train(self):
        print("\33[31mStart training...\33[0m")

        for it in trange(self.train_steps+1, desc="Training iters", leave=True):
            try:
                # TODO: try moving this block to __init__ of Trainer.
                with torch.no_grad():
                    # These things, a batch of initial state (u, v), sampled batch (same size) of target v and their
                    # power spectrum, are used in both the trial forward(s) and the real forward following.
                    self.u_t, self.v_t = init_state_batch(self.batch_sim, self.H, self.W, device=self.device)
                    target_idx_t = torch.randint(0, self.B_targets, (self.batch_sim,), device=self.device)
                    v_t_target = self.v_targets[target_idx_t]
                    _, _, self.ps_t_target = power_spectrum_2d(v_t_target, log=True)

                # To make the trial and real forwards use same elements, we first backprop with the previous trial
                # forward, and make the real forward afterward.
                # This necessitates the skipping of the first backprop as there is no loss at this time.
                if self.loss is not None:
                    # ---------- safe step ----------
                    # 1) get the base gradients (for later linear search of lr)
                    self.opt.zero_grad(set_to_none=True)
                    self.loss.backward()
                    # 2) backup things before repeated trials (because they require param resets)
                    self.opt_state0 = copy.deepcopy(self.opt.state_dict())
                    self.gen.backup_params()
                    if debug_freq(it, print_more=1):
                        print("\033[36mGrads (no backprop until later)\033[0m",
                              f"grad log_Du: {self.gen.log_Du.grad.abs().mean().item():.4e}",
                              f"grad log_Dv: {self.gen.log_Dv.grad.abs().mean().item():.4e}",
                              f"grad raw_F : {self.gen.raw_F.grad.abs().mean().item():.4e}",
                              f"grad raw_k : {self.gen.raw_k.grad.abs().mean().item():.4e}")

                    #### debug ####
                    max_v = 0.0
                    max_name = None
                    for n, p in self.gen.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            v = p.grad.detach().abs().max().item()
                            if v > max_v:
                                max_v = v
                                max_name = n
                    print(f"[it {it}] max|grad|={max_v:.3e} at {max_name}, lr={self.opt.param_groups[0]['lr']:.3e},"
                          f"lr*maxgrad={self.opt.param_groups[0]['lr']*max_v:.3e}")
                    #### end ####

                    print("\33[31mlr trials:\33[0m")
                    self.lrs_trial = self.lrs0
                    for i in range(self.max_tries):
                        if self.step_and_trial(i):
                            break
                    else:
                        self.lrs_trial = self.lrs0
                        for i in range(self.max_tries):
                            if self.step_and_trial(i, shrink=False):
                                break
                        else:
                            raise ValueError("All lrs failed with NaN.")

                    print("\n\33[31mEnd lr trials.\33[0m")

                u, v = self.u_t, self.v_t

                # differentiable forward (fixed K steps)
                u_pred, v_pred, steps_taken = self.gen.simulate_to_steady_trunc_bptt(
                    u=u, v=v,
                    dt=self.dt,
                    K=self.unroll_K,
                    tol=self.tol,
                    max_steps=self.max_steps,
                    device=self.device,
                    return_steps_taken=True,
                )

                energy, ps_mean, ps_pred = power_spectrum_2d(v_pred, log=True)   # [B,H,W]

                self.loss = F.mse_loss(ps_pred, self.ps_t_target)
                # self.loss += alpha * ((v_pred - v_tgt) ** 2).mean()
                print("real run steps taken:", steps_taken)

                #### debug ####
                if debug_freq(it, print_more=1):
                    '''
                    Information Output Block
                    '''
                    pred_std = v_pred.std(dim=(-2, -1))
                    print(f"\n\n\033[32mPrediction v's std: min {pred_std.min().item():.04e}, median {pred_std.median().item():.04e}, max {pred_std.max().item():.04e}")
                    print(f"Power spectrum mean: {energy.item():.4e} (with log: {ps_mean:.4e})\033[0m\n")

                    g_v = torch.autograd.grad(self.loss, v_pred, retain_graph=True, allow_unused=True)[0]
                    print("dL/dv_pred:", "None\n" if g_v is None else f"(min: {g_v.abs().mean().item():.4e}, max: {g_v.abs().max().item():.4e})\n")

                    with torch.no_grad():
                        p = self.gen.params_tensor()
                        print(
                            f"it={it:4d} loss={self.loss.item():.6f}",   # loss_total={self.loss_total.item():.6f}",
                            f"Du={p.Du.item():.4f} Dv={p.Dv.item():.4f} F={p.F.item():.4f} k={p.k.item():.4f}"
                        )
                #### end ####

            except KeyboardInterrupt:
                done = False
                break
            
        else:
            done = True
        with torch.no_grad():
            '''
            Information Output Block
            '''
            p = self.gen.params_tensor()
            print("\33[31mDone.\nLearned params (tensor):\33[0m" if done else
                  f"\33[31mStep {it} aborted.\nShowing params at step {it-1}:\33[0m")
            print("Du=", float(p.Du.cpu()),
                  "Dv=", float(p.Dv.cpu()),
                  "F=", float(p.F.cpu()),
                  "k=", float(p.k.cpu()))
            if not done:
                print(f"To restore training, use parameters"
                      f"log_Du={self.gen.log_Du.item()}, log_Dv={self.gen.log_Dv.item()},"
                      f"raw_F={self.gen.raw_F.item()}, raw_k={self.gen.raw_k.item()}.")

    def test(self):
        pass

    def save_model(self, path: str):
        pass

    def load_model(self, path: str):
        pass


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

