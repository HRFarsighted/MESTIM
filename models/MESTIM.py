import torch
import torch.nn as nn
from mamba_ssm import Mamba
from layers.STIM import STIM
from RevIN.RevIN import RevIN
import torch.nn.functional as F
import torch.fft
from layers.GCN import GraphBlock
from layers.Conv_Blocks import Inception_Block
from layers.Embed import DataEmbedding_wo_temp
from layers.Mamba_EncDec import Encoder, EncoderLayer

def FFT_for_Period(x, k=3):
    # [B, T, C]
    # x = x.permute(0, 2, 1)
    x = x.float()
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class Model(torch.nn.Module):
    def __init__(self, configs):  
        super(Model, self).__init__()
        self.configs = configs  
        if self.configs.revin == 1:  
            self.revin_layer = RevIN(self.configs.enc_in)  
        if self.configs.ch_ind == 1:
            self.d_model_param1 = 1
            self.d_model_param2 = 1
        else:
            self.d_model_param1 = self.configs.n2
            self.d_model_param2 = self.configs.n2

        self.x_embedding = DataEmbedding_wo_temp(self.configs.enc_in, self.configs.n1,
                                           self.configs.embed, self.configs.freq, self.configs.dropout)
        self.x_ive_embedding = DataEmbedding_wo_temp(self.configs.seq_len, self.configs.n2,
                                           self.configs.embed, self.configs.freq, self.configs.dropout)
        self.mamba = Mamba(d_model=self.configs.n1, d_state=self.configs.d_state, d_conv=self.configs.dconv, expand=self.configs.e_fact)
        self.encoder1 = Encoder(
            [
                EncoderLayer(
                        Mamba(
                            d_model=self.configs.n1,  
                            d_state=self.configs.d_state,  
                            d_conv=2, 
                            expand=1,  
                        ),
                        Mamba(
                            d_model=self.configs.n1,  
                            d_state=self.configs.d_state,  
                            d_conv=2,  
                            expand=1,  
                        ),
                    self.configs.n1,
                    512,
                    dropout=self.configs.dropout,
                    activation=torch.nn.ReLU(inplace=True)
                ) for l in range(2)
            ],
            norm_layer=torch.nn.LayerNorm(self.configs.n1)
        )
        self.encoder2 = Encoder(
            [
                EncoderLayer(
                        Mamba(
                            d_model=self.configs.n2,  
                            d_state=self.configs.d_state,  
                            d_conv=2,  
                            expand=1,  
                        ),
                        Mamba(
                            d_model=self.configs.n2,  
                            d_state=self.configs.d_state,  
                            d_conv=2,  
                            expand=1,  
                        ),
                    self.configs.n2,
                    512,
                    dropout=self.configs.dropout,
                    activation=torch.nn.ReLU(inplace=True)
                ) for l in range(2)
            ],
            norm_layer=torch.nn.LayerNorm(self.configs.n2)
        )
        self.STIM = STIM( d_model=self.configs.n1, d_inner=self.configs.n1, bias=True, conv_bias=True, d_conv=self.configs.dconv,
            dt_rank=32, d_state=64)
        self.dropout1 = torch.nn.Dropout(self.configs.dropout)
        self.linear1 = torch.nn.Linear(self.configs.n1, self.configs.enc_in)
        self.linear2 = torch.nn.Linear(self.configs.n1, self.configs.seq_len)
        self.linear_cat = torch.nn.Linear(2*self.configs.seq_len, self.configs.seq_len)
        self.linear3 = torch.nn.Linear(self.configs.seq_len, self.configs.pred_len)
        self.linear_n2_n1 = torch.nn.Linear(self.configs.n2, self.configs.n1)
        self.act1 = torch.nn.ReLU(inplace=True)
        self.k = self.configs.top_k
        self.seq_len = self.configs.seq_len
        self.pred_len = self.configs.pred_len
        self.enc_in = self.configs.enc_in
        self.gconv = GraphBlock(self.enc_in, self.configs.n2, 32, 32, 2, 0.1, 3, 32)  
        self.conv = nn.Sequential(
            Inception_Block(self.configs.n1, 16, num_kernels=3),
            nn.GELU(),
            Inception_Block(16, self.configs.n1, num_kernels=3)
                                 )
    def forward(self, x):
        if self.configs.revin == 1: 
            x = self.revin_layer(x, 'norm')  
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        x_ive = torch.permute(x, (0, 2, 1))
        x_ive = self.x_ive_embedding(x_ive)
        x = self.x_embedding(x)  

        B, T, N = x.size()
        scale_list, scale_weight = FFT_for_Period(x, self.k)
        res_f_m = []
        for i in range(self.k):
            scale = scale_list[i]  
            if (T) % scale != 0:
                length = (((T) // scale) + 1) * scale
                padding = torch.zeros([x.shape[0], (length - (T)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = T
                out = x
            out = out.reshape(B, length // scale, scale,
                              N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)  
            out_m = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            out_m = out_m[:, :T, :]
            out_m,a1 = self.encoder1(out_m)
            res_f_m.append(out_m)

        res_f_m = torch.stack(res_f_m, dim=-1)
        scale_weight = F.softmax(scale_weight, dim=1)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res_f_m = res_f_m * scale_weight
        out_m = torch.sum(res_f_m * scale_weight, -1)  + x

        out_g = self.gconv(x_ive)
        out_g,a2 = self.encoder2(out_g)  
        out_g = out_g + x_ive 
        out_g = self.linear_n2_n1(out_g)

        out_g_cross_out_m = self.STIM(out_g, out_m) + out_g
        out_m_cross_out_g = self.STIM(out_m, out_g) + out_m
     
        out_m_cross_out_g = self.linear1(out_m_cross_out_g)
        out_m_cross_out_g = torch.permute(out_m_cross_out_g, (0, 2, 1))
        out_g_cross_out_m = self.linear2(out_g_cross_out_m) 
        x_out_g_m = torch.cat([out_m_cross_out_g, out_g_cross_out_m], dim=-1)
        x_out = self.linear_cat(x_out_g_m)
        x_out = self.linear3(x_out)
        x_out = torch.permute(x_out, (0, 2, 1))
        if self.configs.revin == 1:
            x_out = self.revin_layer(x_out, 'denorm')
        else:
            x_out = x_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
            x_out = x_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))

        return x_out

