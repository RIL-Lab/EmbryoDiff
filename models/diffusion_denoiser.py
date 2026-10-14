import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
from models.diffusion_utils import *


class MixedConditionBlockV2(nn.Module):
    def __init__(self, d_model, d_cross, kernel_size, mlp_ratio, dropout_rate):
        super(MixedConditionBlockV2, self).__init__()
        
        self.d_model = d_model
        self.d_cross = d_cross
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.padding = (self.kernel_size // 2)
        
        assert(self.kernel_size % 2 == 1)

        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model + d_cross, d_model, kernel_size, padding=self.padding),
        )

        self.att_linear_q = nn.Linear(d_model + d_cross, d_model)
        self.att_linear_k = nn.Linear(d_model + d_cross, d_model)
        self.att_linear_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_rate)
        self.norm_x = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm_cross = nn.LayerNorm(d_cross, elementwise_affine=False, eps=1e-6)

        self.norm_ffn = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=d_model, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout_rate)

    def forward(self, x, x_sem, x_bound):
        """
        x: noise_feature (B, T, D)
        x_cross: condition feature (B, T, C)
        x_bound: boundary feature (B T C)
        """
        x_drop = self.dropout(x)
        x_sem_drop = self.dropout(x_sem)
        if x_bound is not None:
            x_bound_drop = self.dropout(x_bound)
            x_ = torch.cat([x_drop, x_bound_drop], dim=2)
        else:
            x_ = x_drop

        out1 = self.conv_block(x_.permute(0, 2, 1)).permute(0, 2, 1)
        out2 = self.attention(self.norm_x(x_drop), self.norm_cross(x_sem_drop))

        x = x + self.mlp(self.norm_ffn(out1 + out2))

        return x
        
    def attention(self, x, x_cross):
        """
        x: noise_feature (B, T, D)
        x_cross: condition feature (B, T, C)
        """

        x_concat = torch.cat([x, x_cross], dim=-1)  # (B, T, D+C)
        q = self.att_linear_q(x_concat)  # (B, T, D)
        k = self.att_linear_k(x_concat)  # (B, T, D)
        v = self.att_linear_v(x)         # (B, T, D)

        att = torch.einsum('btd,bsd->bts', q, k)  # (B, T, T)
        att = att / (self.d_model ** 0.5)
        att = F.softmax(att, dim=-1)
 
        out = torch.einsum('bts,bsd->btd', att, v)  # (B, T, D)
        return out
    

class MixedConditionModuleV2(nn.Module): # for decoder
    def __init__(self, num_layers, num_f_maps, input_dim_cross, kernel_size, dropout_rate, time_emb_dim=None):
        super(MixedConditionModuleV2, self).__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, num_f_maps)

        self.layers = nn.ModuleList([copy.deepcopy(
            MixedConditionBlockV2(num_f_maps, input_dim_cross, kernel_size, 2, dropout_rate)
        ) for i in range(num_layers)])
    
    def forward(self, x, x_sem, x_bound, time_emb=None):

        if time_emb is not None:
            x = x + self.time_proj(swish(time_emb)).unsqueeze(1)

        for layer in self.layers:
            x = layer(x, x_sem, x_bound)

        return x


class MixedConditionDecoderV2(nn.Module):
    def __init__(self, input_dim, num_classes,
        num_layers, num_f_maps, time_emb_dim, kernel_size, dropout_rate):
        super(MixedConditionDecoderV2, self).__init__()

        self.time_emb_dim = time_emb_dim

        self.time_in = nn.ModuleList([
            torch.nn.Linear(time_emb_dim, time_emb_dim),
            torch.nn.Linear(time_emb_dim, time_emb_dim)
        ])

        self.conv_in = nn.Linear(num_f_maps, num_f_maps)
        self.module = MixedConditionModuleV2(num_layers, num_f_maps, input_dim, kernel_size, dropout_rate, time_emb_dim)
        self.conv_out =  nn.Linear(num_f_maps, num_classes)


    def forward(self, encoder_outs, t, seq_diffused):
        """
        x: feature dict
        t: timestamp
        yt: diffused label sequence embedding
        """
        time_emb = get_timestep_embedding(t, self.time_emb_dim)
        time_emb = self.time_in[0](time_emb)
        time_emb = swish(time_emb)
        time_emb = self.time_in[1](time_emb)

        x_sem = encoder_outs["sem_encoder_feats"]
        x_bound = encoder_outs["bound_encoder_feats"]

        seq_diffused = self.conv_in(seq_diffused)
        seq_diffused = self.module(seq_diffused, x_sem, x_bound, time_emb)
        denoised_seq = self.conv_out(seq_diffused)

        return denoised_seq
    

