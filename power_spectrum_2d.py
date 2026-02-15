import torch
from torch import Tensor
import torch.nn.functional as F


def power_spectrum_2d(x: Tensor,
                      eps: float = 1e-8,
                      log: bool = True,
                      window_size: int = 16,
                      windowed: bool = True
                      ) -> tuple[Tensor, float, Tensor]:
    """
    x: [B,1,H,W] or [B,H,W]
    return: [B,H,W]  (after fftshift, low frequency is at the center)
    """
    if x.dim() == 4:
        x = x[:, 0]  # [B,H,W]
    elif x.dim() != 3:
        raise ValueError("x must be [B,1,H,W] or [B,H,W]")

    # remove DC
    x = x - x.mean(dim=(-2, -1), keepdim=True)

    if windowed:
        X = torch.empty_like(x, dtype=torch.complex64)  # complex [B,H,W]
        H, W = x.shape[1], x.shape[2]
        for i in range((H - 1) // window_size + 1):
            for j in range((W - 1) // window_size + 1):
                start_h = i * window_size
                end_h = min((i + 1) * window_size, H)
                start_w = j * window_size
                end_w = min((j + 1) * window_size, W)
                # get complex [B,H,W]
                X[:, start_h:end_h, start_w:end_w] = torch.fft.fft2(x[:, start_h:end_h, start_w:end_w])
                # center low-freq
                X[:, start_h:end_h, start_w:end_w] = torch.fft.fftshift(X[:, start_h:end_h, start_w:end_w], dim=(-2, -1))
    else:
        X = torch.fft.fft2(x)
        X = torch.fft.fftshift(X, dim=(-2, -1))

    ps = (X.real ** 2 + X.imag ** 2)           # |X|^2  [B,H,W]

    # normalize to compare shapes only not the overall power strengths
    # ps_norm = ps / ps.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)

    if log:
        ps_log = torch.log(ps + eps)
    return ps.mean(), ps_log.mean().item(), ps_log


def windowed_ps_2d_loss(target, pred):
    energy_pred, ps_pred_mean, ps_pred = power_spectrum_2d(pred)
    _, _, ps_target = power_spectrum_2d(target)
    l = F.mse_loss(ps_pred, ps_target)
    return energy_pred, ps_pred_mean, ps_pred, l


def ps_2d_loss(target, pred):
    energy_pred, ps_pred_mean, ps_pred = power_spectrum_2d(pred, windowed=False)
    _, _, ps_target = power_spectrum_2d(target, windowed=False)
    l = F.mse_loss(ps_pred, ps_target)
    return energy_pred, ps_pred_mean, ps_pred, l
