import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from typing import Union
from .Cross_attention import CrossAttention

class STIM(nn.Module):
    def __init__(self, d_model, d_inner, bias, conv_bias, d_conv, dt_rank, d_state):
        super(STIM, self).__init__()

        self.in_proj = nn.Linear(d_model, d_inner, bias=bias)
        self.x_init = nn.Linear(d_model, d_model)

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )
        self.d_inner = d_inner
        self.dt_rank = dt_rank

        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)

        self.cross_attention = CrossAttention(d_model, n_heads=4, d_k=None, d_v=None, attn_dropout=0.0,proj_dropout=0.0, res_attention=0).to('cuda:0')

        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        #self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

    def forward(self, x, xx):
        x = self.in_proj(x)
        (b, l, d) = xx.shape
        xx = rearrange(xx, 'b l d_in -> b d_in l')
        xx = self.conv1d(xx)[:, :, :l]
        xx = rearrange(xx, 'b d_in l -> b l d_in')
        xx = F.silu(xx)
        xx = self.ssm(xx)
        #y = yy * F.silu(x)
        y, c = self.cross_attention(x, xx, xx, key_padding_mask=None, attn_mask=None)
        return y




    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y
