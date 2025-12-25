# trainer.py
import os
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools import GSParams, init_state_batch, debug_freq
from radial_power_spectrum_loss import power_spectrum_2d
from rd_generator import RDGenerator


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen = RDGenerator().to(device)

    # --- 1) Make / load targets (no_grad)
    B_targets = 512
    H = W = 128
    true_params = GSParams(Du=0.16, Dv=0.08, F=0.035, k=0.065)

    param_tag = (
        f"Du{true_params.Du}_"
        f"Dv{true_params.Dv}_"
        f"F{true_params.F}_"
        f"k{true_params.k}_"
        f"HW{H}x{W}_"
        f"B{B_targets}"
    )
    target_pt_path = f"targets_GS_{param_tag}.pt"
    
    if os.path.exists(target_pt_path):
        print(f"Loading cached target patterns from {target_pt_path}")
        v_targets = torch.load(target_pt_path, map_location=device)
    
    else:
        print(f"Generating target patterns using: {true_params}.")
        u_t0, v_t0 = init_state_batch(B_targets, H, W, device=device)
        print(f"Initial status obtained for a batch of {B_targets}, iterating...")
        v_targets = RDGenerator.generate_gray_scott_target_batch(
            u=u_t0, v=v_t0, dt=1.0,
            params=true_params, device=device,
            max_steps=40000, tol=1e-4, return_uv=False
        ).detach()  # [B_targets,1,H,W]
        torch.save(v_targets.cpu(), target_pt_path)
        print(f"Target patterns generated and saved to {target_pt_path}")

    # --- 2) Learnable params + optimizer
    opt = torch.optim.SGD(gen.parameters(), lr=1e-2, momentum=0.0, weight_decay=5e-4)

    # --- 3) Training hyperparams
    train_steps = 2000
    batch_sim = 8          # how many sims per iter
    unroll_K = 1000         # truncated backprop steps
    dt = 1.0
    alpha = 0.0           # pixel L2 weight
    grad_clip = None #1.5
    tol = 1e-3
    max_steps = 40000
    # lam_E = 3e-2

    # Optional: pick targets as batch too
    print("\33[31mStart training...\33[0m")
    for it in trange(train_steps, desc="Training iters", leave=True):
        try:
    
            # sample a batch of targets (distribution matching)
            idx = torch.randint(0, B_targets, (batch_sim,), device=device)
            v_tgt = v_targets[idx]  # [B,1,H,W]
    
            # random init every iter (matches "real world unknown init")
            u, v = init_state_batch(batch_sim, H, W, device=device)
    
            # differentiable forward (fixed K steps)
            u_pred, v_pred, steps_taken = gen.simulate_to_steady_trunc_bptt(
                u=u, v=v,
                batch_sim=batch_sim,
                dt=dt,
                K=unroll_K,
                tol=tol,
                max_steps=max_steps,
                device=device,
                return_steps_taken=True,
            )

            E, ps_mean, ps_pred = power_spectrum_2d(v_pred, log=True)   # [B,H,W]
            _, _, ps_tgt = power_spectrum_2d(v_tgt, log=True) # [B,H,W]

            loss = torch.nn.functional.mse_loss(ps_pred, ps_tgt)
            loss += alpha * ((v_pred - v_tgt) ** 2).mean()
            #loss_E = lam_E * (
            #    torch.relu(E_min - E) +
            #    torch.relu(E - E_max)
            #)
            #loss_total = loss + loss_E


            #### debug ####
            if debug_freq(it, print_more=4):

                pred_std = v_pred.std(dim=(-2, -1))
                print(f"\n\n\033[32mPrediction stats: {pred_std.min().item():.04e}, {pred_std.median().item():.04e}, {pred_std.max().item():.04e}")
                print(f"Power spectrum: {E.item():.4e}, w/ log: {ps_mean:.4e}\033[0m\n")

                g_v = torch.autograd.grad(loss, v_pred, retain_graph=True, allow_unused=True)[0]
                print("dL/dv_pred:", "None\n" if g_v is None else f"({g_v.abs().mean().item():.4e}, {g_v.abs().max().item():.4e})\n")

                loss2 = ((v_pred - v_tgt) ** 2).mean()
                g2 = torch.autograd.grad(loss2, gen.raw_F, retain_graph=True, allow_unused=True)[0]
                print("with L2, grad raw_F:", "None\n" if g2 is None else f"({g2.abs().mean().item():.4e}, {g2.abs().max().item():.4e})\n")

                print("\033[36mBefore backprop (last grads)\033[0m", f"grad log_Du: {gen.log_Du.grad.abs().mean().item():.4e}",
                      f"grad log_Dv: {gen.log_Dv.grad.abs().mean().item():.4e}",
                      f"grad raw_F : {gen.raw_F.grad.abs().mean().item():.4e}",
                      f"grad raw_k : {gen.raw_k.grad.abs().mean().item():.4e}")
            #### end ####

            opt.zero_grad(set_to_none=True)
            loss.backward()

            #### debug ####
            if debug_freq(it, print_more=4):
                print("\033[34mAfter backprop (before clip)\033[0m", f"grad log_Du: {gen.log_Du.grad.abs().mean().item():.04e}",
                      f"grad log_Dv: {gen.log_Dv.grad.abs().mean().item():.4e}",
                      f"grad raw_F : {gen.raw_F.grad.abs().mean().item():.4e}",
                      f"grad raw_k : {gen.raw_k.grad.abs().mean().item():.4e}")
                with torch.no_grad():
                    p = gen.params_tensor()
                    print(
                        f"it={it+1:4d} loss={loss.item():.6f}",   # loss_total={loss_total.item():.6f}",
                        f"Du={p.Du.item():.4f} Dv={p.Dv.item():.4f} F={p.F.item():.4f} k={p.k.item():.4f}"
                    )
    
            if grad_clip is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(gen.parameters(), grad_clip)
                if it != 0 and it % 25 == 0:
                    print("total_norm(before clip):", total_norm.item())
                    print("raw_k grad after clip, before optimization:", gen.raw_k.grad.abs().mean().item(), gen.raw_k.grad.abs().max().item())
            #### end ####

        except KeyboardInterrupt:
            done = False
            break
        
        try:
            opt.step()
        except KeyboardInterrupt:
            print("Ctrl+C ignored, continuing...")

    else:
        done = True
    with torch.no_grad():
        p = gen.params_tensor()
        print("\33[31mDone.\nLearned params (tensor):\33[0m" if done else f"\33[31mStep {it} aborted.\nShowing params at step {it-1}:\33[0m")
        print("Du=", float(p.Du.cpu()),
              "Dv=", float(p.Dv.cpu()),
              "F=", float(p.F.cpu()),
              "k=", float(p.k.cpu()))
        if not done:
            print(f"To restore training, use parameters log_Du={gen.log_Du.item()}, log_Dv={gen.log_Dv.item()}, raw_F={gen.raw_F.item()}, raw_k={gen.raw_k.item()}.")


if __name__ == "__main__":
    main()

