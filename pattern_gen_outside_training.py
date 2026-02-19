import numpy as np
import torch
from rd_generator import RDGenerator
from tools import GSParams, init_state_batch
import matplotlib.animation as animation
from animation import get_initial_artists, updatefig
from power_spectrum_2d import windowed_ps_2d_loss, ps_2d_loss

import matplotlib.pyplot as plt

import sys
import copy


def print_usage_and_exit():
    print(
        "Please specify the task you want to perform:\n"
        "1. generate targets\n"
        "2. test training forward (starting from noisy random parameters)\n"
        "3. load existing targets and show the first 4 of them\n"
        "4. show stepping animation\n"
        "5. plot the polyline of loss values on a slice of the parameter space\n"
        "Type: python pattern_gen_outside_training.py <number of your option> <optional: path to existing targets>"
    )
    sys.exit(1)


# ---- parse command line ----
if len(sys.argv) < 2:
    print_usage_and_exit()

try:
    task_id = int(sys.argv[1])
except ValueError:
    print_usage_and_exit()

if task_id not in (1, 2, 3, 4, 5):
    print_usage_and_exit()

if task_id == 3:
    target_pt_path = sys.argv[2]
task = 'test training forward' if task_id == 2 else \
       'generate targets' if task_id == 1 else \
       'show stored targets' if task_id ==3 else \
       'animation' if task_id == 4 else \
       'polyline' if task_id == 5 else \
       ''
print(f"Current task: {task}")

save_animation = False


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    params = [GSParams(Du=0.16, Dv=0.08, F=0.03, k=0.053), GSParams(Du=0.16, Dv=0.08, F=0.03, k=0.0555), GSParams(Du=0.16, Dv=0.08, F=0.03, k=0.058), GSParams(Du=0.16, Dv=0.08, F=0.03, k=0.0605), GSParams(Du=0.16, Dv=0.08, F=0.03, k=0.063), GSParams(Du=0.16, Dv=0.08, F=0.03, k=0.0655), GSParams(Du=0.16, Dv=0.08, F=0.03, k=0.068), GSParams(Du=0.16, Dv=0.08, F=0.03666666666666667, k=0.053), GSParams(Du=0.16, Dv=0.08, F=0.03666666666666667, k=0.0555), GSParams(Du=0.16, Dv=0.08, F=0.03666666666666667, k=0.058), GSParams(Du=0.16, Dv=0.08, F=0.03666666666666667, k=0.0605), GSParams(Du=0.16, Dv=0.08, F=0.03666666666666667, k=0.063), GSParams(Du=0.16, Dv=0.08, F=0.03666666666666667, k=0.0655), GSParams(Du=0.16, Dv=0.08, F=0.03666666666666667, k=0.068), GSParams(Du=0.16, Dv=0.08, F=0.043333333333333335, k=0.053), GSParams(Du=0.16, Dv=0.08, F=0.043333333333333335, k=0.0555), GSParams(Du=0.16, Dv=0.08, F=0.043333333333333335, k=0.058), GSParams(Du=0.16, Dv=0.08, F=0.043333333333333335, k=0.0605), GSParams(Du=0.16, Dv=0.08, F=0.043333333333333335, k=0.063), GSParams(Du=0.16, Dv=0.08, F=0.043333333333333335, k=0.0655), GSParams(Du=0.16, Dv=0.08, F=0.043333333333333335, k=0.068), GSParams(Du=0.16, Dv=0.08, F=0.05, k=0.053), GSParams(Du=0.16, Dv=0.08, F=0.05, k=0.0555), GSParams(Du=0.16, Dv=0.08, F=0.05, k=0.058), GSParams(Du=0.16, Dv=0.08, F=0.05, k=0.0605), GSParams(Du=0.16, Dv=0.08, F=0.05, k=0.063), GSParams(Du=0.16, Dv=0.08, F=0.05, k=0.0655), GSParams(Du=0.16, Dv=0.08, F=0.05, k=0.068), GSParams(Du=0.16, Dv=0.08, F=0.05666666666666667, k=0.053), GSParams(Du=0.16, Dv=0.08, F=0.05666666666666667, k=0.0555), GSParams(Du=0.16, Dv=0.08, F=0.05666666666666667, k=0.058), GSParams(Du=0.16, Dv=0.08, F=0.05666666666666667, k=0.0605), GSParams(Du=0.16, Dv=0.08, F=0.05666666666666667, k=0.063), GSParams(Du=0.16, Dv=0.08, F=0.05666666666666667, k=0.0655), GSParams(Du=0.16, Dv=0.08, F=0.05666666666666667, k=0.068), GSParams(Du=0.16, Dv=0.08, F=0.06333333333333334, k=0.053), GSParams(Du=0.16, Dv=0.08, F=0.06333333333333334, k=0.0555), GSParams(Du=0.16, Dv=0.08, F=0.06333333333333334, k=0.058), GSParams(Du=0.16, Dv=0.08, F=0.06333333333333334, k=0.0605), GSParams(Du=0.16, Dv=0.08, F=0.06333333333333334, k=0.063), GSParams(Du=0.16, Dv=0.08, F=0.06333333333333334, k=0.0655), GSParams(Du=0.16, Dv=0.08, F=0.06333333333333334, k=0.068), GSParams(Du=0.16, Dv=0.08, F=0.07, k=0.053), GSParams(Du=0.16, Dv=0.08, F=0.07, k=0.0555), GSParams(Du=0.16, Dv=0.08, F=0.07, k=0.058), GSParams(Du=0.16, Dv=0.08, F=0.07, k=0.0605), GSParams(Du=0.16, Dv=0.08, F=0.07, k=0.063), GSParams(Du=0.16, Dv=0.08, F=0.07, k=0.0655), GSParams(Du=0.16, Dv=0.08, F=0.07, k=0.068)]
    # params = GSParams(Du=0.16, Dv=0.08, F=0.035, k=0.065)

    print('Generating initial batch...')
    u, v = init_state_batch(B=4, H=128, W=128, device=device)
    if task == 'generate targets':
        v_batch = RDGenerator.generate_gray_scott_target_batch(u, v, params=params[3], device=device, tol=1e-8,
                                                               max_steps=50000)
    elif task == 'test training forward':
        gen = RDGenerator()
        v_batch, overflow = gen.simulate_to_steady_trunc_bptt(u, v, device=device)[1:3]
        if overflow:
            print("The output images may not be accurate. Values out of [0, 1] range during stepping.")
    elif task == 'show stored targets':
        v_batch = torch.load(target_pt_path, map_location=device)
    elif task == 'animation':
        if type(params) is list:
            l = len(params)
            repeat = [(l-1) // 4 + 1, 1, 1, 1]
            u_animation = [u_item.squeeze(0) for u_item in u.repeat(repeat)[:l]]
            v_animation = [v_item.squeeze(0) for v_item in v.repeat(repeat)[:l]]
        else:
            u_animation, v_animation = u[0].squeeze(0), v[0].squeeze(0)

        fig, im, txt = get_initial_artists(v_animation, per_line=7)
        updates_per_frame = 5
        dt = 1.0
        animation_arguments = (updates_per_frame, im, txt, u_animation, v_animation, params, dt)
        ani = animation.FuncAnimation(fig,  # matplotlib figure
                                      updatefig,  # function that takes care of the update
                                      fargs=animation_arguments,  # arguments to pass to this function
                                      interval=1,  # update every `interval` milliseconds
                                      frames=10000,
                                      blit=False,  # optimize the drawing update
                                      )
        if save_animation:
            print("Saving animation...")
            ani.save(
                "gray_scott.mp4",
                writer="ffmpeg",
                fps=30,
                dpi=150
            )
            print("Saved to gray_scott.mp4")
        # show the animation
        plt.show()
        exit(0)
    elif task == 'polyline':
        if len(sys.argv) < 6 and sys.argv[2] != "param_in_file":
            print("Type `python pattern_gen_outside_training.py 5 <name_of_param_to_slice> "
                  "<start_value> <end_value> <slice_num>` or `python pattern_gen_outside_training.py 5 param_in_file`.")
            sys.exit(1)
        p = GSParams(Du=0.16, Dv=0.08, F=0.035, k=0.065)
        up, vp = u.clone(), v.clone()
        # v_batch = RDGenerator.generate_gray_scott_target_batch(up, vp, params=p, device=device, tol=1e-8,
        #                                                        max_steps=50000)
        _, v_batch, _ = RDGenerator.simulate_constant_steps(up, vp, p, num_steps=40000)
        loss_values = []
        if sys.argv[2] != "param_in_file":
            if len(sys.argv) > 6:
                vname, vmin, vmax, steps = sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
                v2name, v2min, v2max, steps2 = sys.argv[6], float(sys.argv[7]), float(sys.argv[8]), int(sys.argv[9])
                values = [vmin + i * (vmax - vmin) / (steps - 1) for i in range(steps)]
                values2 = [v2min + i * (v2max - v2min) / (steps2 - 1) for i in range(steps2)]
                params = []
                for value in values:
                    p_inner_list = []
                    for value2 in values2:
                        _p = copy.deepcopy(p)
                        setattr(_p, vname, value)
                        setattr(_p, v2name, value2)
                        p_inner_list.append(_p)
                    params.append(p_inner_list)
            else:
                vname, vmin, vmax, steps = sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
                values = [vmin + i * (vmax - vmin) / (steps - 1) for i in range(steps)]
                params = []
                for value in values:
                    _p = copy.deepcopy(p)
                    setattr(_p, vname, value)
                    params.append(_p)
        else:
            values = [i for i, _ in enumerate(params)]
            vname = "Index of parameter sets"

        with torch.no_grad():
            if type(params[0]) is list:
                for _p in params:
                    inner_loss_list = []
                    for i in range(steps2):
                        _, v_final, _ = RDGenerator.simulate_constant_steps(u.clone(), v.clone(), _p[i], num_steps=25000)
                        inner_loss_list.append(ps_2d_loss(v_batch, v_final)[3].item())
                    loss_values.append(inner_loss_list)

                data_array = np.array(loss_values) # 0-F, 1-k
                plt.imshow(data_array, cmap='viridis', extent=(v2min, v2max, vmin, vmax))  # or 'hot', 'plasma', 'gray'
                plt.colorbar()
                plt.xlabel(v2name)
                plt.ylabel(vname)
                plt.gca().set_aspect((v2max - v2min) / (vmax - vmin))
                plt.title(f"Loss Landscape 2D Cross-Section ({v2name}-{vname})")
                plt.show()
            else:
                for _p in params:
                    # _, v_final, _, _ = RDGenerator(params=_p).simulate_to_steady_trunc_bptt(up, vp, device=device,
                    #                                                                         tol=1e-8, max_steps=50000,
                    #                                                                         disable_progress_bar=False)
                    _, v_final, _ = RDGenerator.simulate_constant_steps(u.clone(), v.clone(), _p, num_steps=40000)
                    loss_values.append(ps_2d_loss(v_batch, v_final)[3].item())
                                       # + torch.abs(v_batch.mean()-v_final.mean()).item())

                plt.plot(values, loss_values, marker='o', linestyle='-', color='b', lw=0.5, ms=3, markeredgewidth=0)
                plt.xlabel(vname)
                plt.ylabel('Loss')
                plt.title('Loss Landscape')
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

    print("Range of pixel values:", v_batch.min().item(), v_batch.max().item(),
          "Median and average std of pixels:", v_batch.median().item(), v_batch.std(dim=(-1, -2)).mean().item())

