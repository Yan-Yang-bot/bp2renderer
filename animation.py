# import necessary libraries
import torch
import matplotlib.pyplot as plt
from random import randint
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from tools import n_steps, GSParams
import matplotlib.animation as animation
from matplotlib.colors import Normalize, LinearSegmentedColormap
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

cmap = LinearSegmentedColormap.from_list("mycmap", ["#604957", "#C1D795", "#C1D795", "#C1D795", "#739F62", "#739F62"])


def get_initial_artists(v: np.ndarray):  #, V, title):
    """return the matplotlib artists for animation"""
    fig, ax = plt.subplots(1, 1, figsize=(3.7, 3.7))
    im = ax.imshow(v, animated=True, vmin=0, cmap="Grays")
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    txt = ax.text(
        2, 2, "",               # 左上角（像素坐标）
        color="white",
        fontsize=10,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.5, pad=2)
    )

    return fig, im, txt


def updatefig(frame_id: int, updates_per_frame: int, im, txt, u: torch.Tensor, v: torch.Tensor, p: GSParams, dt: float):
    """Takes care of the matplotlib-artist update in the animation"""
    # print(p, dt)

    # update x times before updating the frame
    u2, v2, overflow = n_steps(u, v, p, dt, updates_per_frame)
    if overflow:
        raise

    v_np = v2.cpu().numpy()
    im.set_array(v_np)

    im.set_norm(Normalize(vmin=0.0,vmax=0.1))
    # im.set_norm(Normalize(vmin=np.amin(v_np),vmax=np.amax(v_np)))

    step = (frame_id + 1) * updates_per_frame
    txt.set_text(f"step ~ {step}")

    u.copy_(u2)
    v.copy_(v2)

    return im, txt
