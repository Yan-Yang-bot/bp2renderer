import torch

def radial_profile(ps: torch.Tensor) -> torch.Tensor:
    """
    ps: [B, H, W] power spectrum (already shifted)
    returns: [B, R] radial mean profile, R = floor(min(H,W)/2)+1
    """
    B, H, W = ps.shape
    device = ps.device

    yy = torch.arange(H, device=device) - (H // 2)
    xx = torch.arange(W, device=device) - (W // 2)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    r = torch.sqrt(X.float() ** 2 + Y.float() ** 2)  # [H,W]

    rmax = int(min(H, W) // 2)
    rbin = torch.clamp(r.round().long(), 0, rmax)    # [H,W] integer bins

    # flatten
    rbin_f = rbin.view(-1)                 # [H*W]
    ps_f = ps.view(B, -1)                  # [B, H*W]

    # bin-counts (same for all batch)
    counts = torch.bincount(rbin_f, minlength=rmax + 1).float().clamp_min(1.0)  # [R]

    # sum within bins for each batch using scatter_add
    out = torch.zeros(B, rmax + 1, device=device, dtype=ps.dtype)
    out.scatter_add_(1, rbin_f.unsqueeze(0).expand(B, -1), ps_f)

    out = out / counts.unsqueeze(0)  # mean
    return out  # [B, R]

def radial_power_spectrum(x: torch.Tensor, eps: float = 1e-8, log: bool = True) -> torch.Tensor:
    """
    x: [B,1,H,W] or [B,H,W]
    return: [B, R] radial power spectrum (optionally log-scaled)
    """
    if x.dim() == 4:
        x = x[:, 0]  # [B,H,W]
    elif x.dim() != 3:
        raise ValueError("x must be [B,1,H,W] or [B,H,W]")

    # remove DC (mean) to avoid trivial matching
    x = x - x.mean(dim=(-2, -1), keepdim=True)

    # FFT -> power spectrum
    X = torch.fft.fft2(x)                           # complex [B,H,W]
    ps = (X.real ** 2 + X.imag ** 2)                # |X|^2  [B,H,W]
    ps = torch.fft.fftshift(ps, dim=(-2, -1))       # center low-freq

    # radial average
    rp = radial_profile(ps)                         # [B,R]

    # normalize (optional but usually helps)
    rp = rp / (rp.sum(dim=-1, keepdim=True).clamp_min(eps))

    if log:
        rp = torch.log(rp + eps)
    return rp

def radial_ps_loss(pred: torch.Tensor, target: torch.Tensor, includ_var: bool = True, l1: bool = False) -> torch.Tensor:
    """
    pred/target: [B,1,H,W] or [B,H,W]
    """
    rp_pred = radial_power_spectrum(pred, log=True)
    rp_tgt  = radial_power_spectrum(target, log=True)
    if includ_var:
        return (rp_pred.mean(0) - rp_tgt.mean(0)).pow(2).mean() \
             + 0.5 * (rp_pred.var(0) - rp_tgt.var(0)).pow(2).mean()
    if l1:
        return (rp_pred - rp_tgt).abs().mean()
    return ((rp_pred - rp_tgt) ** 2).mean()

