# Example usage
import torch
from rd_generator import RDGenerator
from tools import GSParams, init_state_batch

import matplotlib.pyplot as plt

import sys

def print_usage_and_exit():
    print(
        "Please specify the task you want to perform:\n"
        "1. generate targets\n"
        "2. test training forward (starting from noisy random parameters)\n"
        "\n"
        "Type: python generate_targets.py <number of your option>"
    )
    sys.exit(1)


# ---- parse command line ----
if len(sys.argv) < 2:
    print_usage_and_exit()

try:
    task_id = int(sys.argv[1])
except ValueError:
    print_usage_and_exit()

if task_id not in (1, 2):
    print_usage_and_exit()

task = 'test training forward' if task_id == 2 else ('generate targets' if task_id == 1 else '')
print(f"Current task: {task}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    params = GSParams(Du=0.1636, Dv=0.1555, F=0.0422, k=0.0589)
    #params = GSParams(Du=0.16, Dv=0.08, F=0.035, k=0.065)
    print('parameters generated.')

    print('Generating...')
    u, v = init_state_batch(B=4, H=128, W=128, device=device)
    if task == 'generate targets':
        v_batch = RDGenerator.generate_gray_scott_target_batch(u, v, params=params, device=device, tol=1e-4, max_steps=50000)
    elif task == 'test training forward':
        gen = RDGenerator()
        v_batch = gen.simulate_to_steady_trunc_bptt(u, v, batch_sim=4, device=device)[1]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    v_batch_np = v_batch.detach().cpu().numpy().squeeze(1)

    for ax, img in zip(axes.flatten(), v_batch_np):
        ax.imshow(img, cmap='viridis')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close(fig)

