import torch
import torch.nn as nn
import torch.nn.functional as F

def laplacian_kernel(device, dtype):
    k = torch.tensor([[0.0, 1.0, 0.0],
                      [1.0,-4.0, 1.0],
                      [0.0, 1.0, 0.0]], device=device, dtype=dtype)
    return k.view(1,1,3,3)

def laplacian(x, k):
    # x: [B,1,H,W]
    return F.conv2d(x, k, padding=1)

class RDGenerator(nn.Module):
    def __init__(self, H, W, dt=1.0):
        super().__init__()
        self.H, self.W = H, W
        # self.T = T
        self.dt = dt

        # 用 softplus 保证非负（更稳）
        self.log_Du = nn.Parameter(torch.tensor(-2.0))
        self.log_Dv = nn.Parameter(torch.tensor(-2.0))

        # F,k 通常在 (0, 0.1) 左右，先用 sigmoid 映射到合理区间
        self.raw_F = nn.Parameter(torch.tensor(0.0))
        self.raw_k = nn.Parameter(torch.tensor(0.0))
        print(f"End of init: log_Du = {self.log_Du}, log_Dv = {self.log_Dv}, raw_F = {self.raw_F}, raw_k = {self.raw_k}")

    def forward(self, U0, V0):
        # U0,V0: [B,1,H,W]
        device, dtype = U0.device, U0.dtype
        k_lap = laplacian_kernel(device, dtype)

        Du = F.softplus(self.log_Du)  # >=0
        Dv = F.softplus(self.log_Dv)

        Ff = 0.1 * torch.sigmoid(self.raw_F)   # (0, 0.1)
        kk = 0.1 * torch.sigmoid(self.raw_k)

        U, V = U0, V0

        for _ in range(self.T):
            lapU = laplacian(U, k_lap)
            lapV = laplacian(V, k_lap)

            UVV = U * V * V
            dU = Du * lapU - UVV + Ff * (1.0 - U)
            dV = Dv * lapV + UVV - (Ff + kk) * V

            U = U + self.dt * dU
            V = V + self.dt * dV

            U = torch.sigmoid(U)
            V = torch.sigmoid(V)

        # 读出：直接用 V 当图
        return V

# 用法示例（单步训练）
B,H,W = 4,128,128
gen = RDGenerator(H,W,T=200,dt=1.0).cuda()

# 初始条件可学习也可固定
U0 = torch.ones(B,1,H,W, device="cuda")
V0 = torch.zeros(B,1,H,W, device="cuda")
# 加一个随机扰动 seed
V0 = V0 + 0.02*torch.rand_like(V0)

target = torch.rand(B,1,H,W, device="cuda")

out = gen(U0, V0)
loss = F.mse_loss(out, target)
loss.backward()

