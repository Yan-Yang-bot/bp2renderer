# import necessary libraries
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from tools import n_steps, GSParams
from matplotlib.colors import Normalize, LinearSegmentedColormap
import sys
import numpy as np
from typing import Union
np.set_printoptions(threshold=sys.maxsize)

cmap = LinearSegmentedColormap.from_list("mycmap", ["#604957", "#C1D795", "#C1D795", "#C1D795", "#739F62", "#739F62"])


def get_initial_artists(v: Union[torch.Tensor, list]):
    """return the matplotlib artists for animation"""
    if type(v) is list:
        l = len(v)
        assert l > 0
        w, h = 3, (l - 1) // 3 + 1
        fig, axes = plt.subplots(h, w, figsize=(4*w, 4*h))
        im_list = []
        for idx, ax in enumerate(axes.flatten()):
            if idx == h * w - 1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
            if idx >= l:
                ax.cla()
                ax.axis("off")
                ax.text(0.47, 0.46, "x", fontsize=22, color="gray")
                continue
            im = ax.imshow(v[idx].cpu().numpy(), cmap='Grays', animated=True)
            ax.set_title(f"{idx+1}")
            ax.axis("off")
            im_list.append(im)
            if idx == 0:
                txt = ax.text(
                    2, 2, "",  # left top
                    color="white",
                    fontsize=10,
                    ha="left",
                    va="top",
                    bbox=dict(facecolor="black", alpha=0.5, pad=2)
                )
        return fig, im_list, txt

    fig, ax = plt.subplots(1, 1, figsize=(3.7, 3.7))
    im = ax.imshow(v.cpu().numpy(), animated=True, vmin=0, cmap="Grays")
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    txt = ax.text(
        2, 2, "",               # left top
        color="white",
        fontsize=10,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.5, pad=2)
    )

    return fig, im, txt


def updatefig(frame_id: int, updates_per_frame: int, im, txt,
              u: Union[torch.Tensor, list[torch.Tensor]],
              v: Union[torch.Tensor, list[torch.Tensor]],
              p: Union[GSParams, list[GSParams]], dt: float):
    """Takes care of the matplotlib-artist update in the animation"""
    if type(im) is list:
        assert type(u) is list and type(v) is list and type(p) is list
        return _updatefig_multi(frame_id, updates_per_frame, im, txt, u, v, p, dt)

    else:
        assert type(u) is torch.Tensor and type(v) is torch.Tensor and type(p) is GSParams
        # update x times before updating the frame
        u2, v2, overflow = n_steps(u, v, p, dt, updates_per_frame)
        if overflow:
            raise

        v_np = v2.cpu().numpy()
        im.set_array(v_np)

        im.set_norm(Normalize(vmin=0.0, vmax=0.1))
        # im.set_norm(Normalize(vmin=np.amin(v_np),vmax=np.amax(v_np)))

        step = (frame_id + 1) * updates_per_frame
        txt.set_text(f"step ~ {step}")

        u.copy_(u2)
        v.copy_(v2)

        return im, txt


def _updatefig_multi(frame_id, updates_per_frame, im_list, txt, u_list, v_list, params_list, dt):
    for idx, (u, v, params) in enumerate(zip(u_list, v_list, params_list)):

        u2, v2, overflow = n_steps(u, v, params, dt, updates_per_frame)
        if overflow:
            raise

        v_np = v2.cpu().numpy()

        im_list[idx].set_array(v_np)
        im_list[idx].set_norm(Normalize(vmin=0.0, vmax=0.1))

        u_list[idx].copy_(u2)
        v_list[idx].copy_(v2)

    step = (frame_id + 1) * updates_per_frame
    txt.set_text(f"step ~ {step}")

    return im_list, txt

