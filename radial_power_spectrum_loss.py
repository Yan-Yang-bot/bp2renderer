import torch

def power_spectrum_2d(x: torch.Tensor, eps: float = 1e-8, log: bool = True) -> torch.Tensor:
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

    X = torch.fft.fft2(x)                      # complex [B,H,W]
    ps = (X.real ** 2 + X.imag ** 2)           # |X|^2  [B,H,W]
    ps = torch.fft.fftshift(ps, dim=(-2, -1))  # center low-freq

    # normalize to compare shapes only not the overall power strengths
    ps_norm = ps / (ps.sum(dim=(-2, -1), keepdim=True).clamp_min(eps))

    if log:
        ps_norm = torch.log(ps_norm + eps)
    return ps.mean().item(), ps_norm.mean().item(), ps_norm

