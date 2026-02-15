import torch
from typing import Union
import matplotlib.pyplot as plt
from power_spectrum_2d import ps_2d_loss, windowed_ps_2d_loss
import random


def plot_loss_matrix(
    M,
    *,
    title: str = "Loss matrix",
    figsize=(6, 5),
    v_min=None,
    v_max=None,
    cmap: str = "gray",
    show: bool = True,
    save_path: Union[str, None] = None,
):
    """
    M: torch.Tensor [B,B] or numpy array
    Draw grayscale image + colorbar scale.
    """
    if isinstance(M, torch.Tensor):
        M = M.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(M, cmap=cmap, vmin=v_min, vmax=v_max)
    ax.set_title(title)
    ax.set_xlabel("j")
    ax.set_ylabel("i")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("loss")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


v_all = torch.load("targets_GS_Du0.16_Dv0.08_F0.035_k0.065_HW128x128_B512.pt", map_location="cpu")
# v_all: [512, 1, 128, 128]

assert v_all.ndim == 4 and v_all.shape[1] == 1
B, _, H, W = v_all.shape
print("Loaded:", v_all.shape)

# [B, H*W]
v_flat = v_all.view(B, -1)

# pairwise squared L2
# dist[i,j] = mean((v_i - v_j)^2)
diff = v_flat[:, None, :] - v_flat[None, :, :]   # [B,B,HW]
l2_mat = (diff ** 2).mean(dim=-1)                 # [B,B]

print("L2 Loss matrix shape:", l2_mat.shape)
plot_loss_matrix(l2_mat, title="Pairwise L2 loss", cmap="gray")


def pairwise_2d_ps(v, idxs):
    n = len(idxs)
    mat = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            li = ps_2d_loss(v[idxs[i]:idxs[i] + 1], v[idxs[j]:idxs[j] + 1])
            mat[i, j] = mat[j, i] = li.item()
    return mat


# 随机抽 32 张
idxs = random.sample(range(B), 32)
ps_mat = pairwise_2d_ps(v_all, idxs)

print("2D Power Spectrum Loss Matrix shape:", ps_mat.shape)

vals = ps_mat[~torch.eye(ps_mat.shape[0], dtype=torch.bool)]
v_min = torch.quantile(vals, 0.05).item()
v_max = torch.quantile(vals, 0.95).item()
plot_loss_matrix(ps_mat, title="Pairwise power spectrum loss (5-95%)", v_min=v_min, v_max=v_max)


