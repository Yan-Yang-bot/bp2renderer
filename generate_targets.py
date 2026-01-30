# Example usage
import torch
from rd_generator import RDGenerator
from tools import GSParams, init_state_batch
import matplotlib.animation as animation
from animation import get_initial_artists, updatefig

import matplotlib.pyplot as plt

import sys


def print_usage_and_exit():
    print(
        "Please specify the task you want to perform:\n"
        "1. generate targets\n"
        "2. test training forward (starting from noisy random parameters)\n"
        "3. load existing targets and show the first 4 of them\n"
        "4. show stepping animation\n"
        "Type: python generate_targets.py <number of your option> <optional: path to existing targets>"
    )
    sys.exit(1)


# ---- parse command line ----
if len(sys.argv) < 2:
    print_usage_and_exit()

try:
    task_id = int(sys.argv[1])
except ValueError:
    print_usage_and_exit()

if task_id not in (1, 2, 3, 4):
    print_usage_and_exit()

if task_id == 3:
    target_pt_path = sys.argv[2]
task = 'test training forward' if task_id == 2 else \
       'generate targets' if task_id == 1 else \
       'show stored targets' if task_id ==3 else \
       'animation' if task_id == 4 else ''
print(f"Current task: {task}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    #params = GSParams(Du=0.1270, Dv=0.1269, F=0.0500, k=0.0501)
    params = GSParams(Du=0.1341, Dv=0.1198, F=0.0477, k=0.0580)
    #params = GSParams(Du=0.16, Dv=0.08, F=0.035, k=0.065)
    print('parameters generated.')

    print('Generating...')
    u, v = init_state_batch(B=4, H=128, W=128, device=device)
    if task == 'generate targets':
        v_batch = RDGenerator.generate_gray_scott_target_batch(u, v, params=params, device=device, tol=1e-8, max_steps=50000)
    elif task == 'test training forward':
        gen = RDGenerator()
        v_batch, overflow = gen.simulate_to_steady_trunc_bptt(u, v, device=device)[1:3]
        if overflow:
            print("The output images may not be accurate. Values out of [0, 1] range during stepping.")
    elif task == 'show stored targets':
        v_batch = torch.load(target_pt_path, map_location=device)
    elif task == 'animation':
        u, v = u[0].squeeze(0), v[0].squeeze(0)
        fig, im, txt = get_initial_artists(v.cpu().numpy())
        updates_per_frame = 4
        dt = 1.0
        animation_arguments = (updates_per_frame, im, txt, u, v, params, dt)
        ani = animation.FuncAnimation(fig,  # matplotlib figure
                                      updatefig,  # function that takes care of the update
                                      fargs=animation_arguments,  # arguments to pass to this function
                                      interval=1,  # update every `interval` milliseconds
                                      frames=50000,
                                      blit=False,  # optimize the drawing update
                                      )
        # show the animation
        plt.show()
        exit(0)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    v_batch_np = v_batch.detach().cpu().numpy().squeeze(1)

    for ax, img in zip(axes.flatten(), v_batch_np):
        ax.imshow(img, cmap='viridis')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    print("Range of pixel values:", v_batch.min().item(), v_batch.max().item(), v_batch.median().item(), v_batch.std(dim=(-1,-2)).mean().item())

