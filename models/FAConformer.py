import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class LogPowerLayer(nn.Module):
    def __init__(self, dim):
        super(LogPowerLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(torch.mean(x ** 2, dim=self.dim), 1e-4, 1e4))

class InterFre(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        out = sum(x)
        out = F.gelu(out)
        return out

class Stem(nn.Module):
    def __init__(self, data_name, in_planes, out_planes=64, kernel_size=63, patch_size=125, radix=2):
        nn.Module.__init__(self)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = out_planes * radix
        self.kernel_size = kernel_size
        self.radix = radix
        self.patch_size = patch_size
        self.data_name = data_name

        self.sconv = Conv(nn.Conv1d(self.in_planes, self.mid_planes, 1, bias=False, groups = radix),
                          bn=nn.BatchNorm1d(self.mid_planes), activation=None)

        self.tconv = nn.ModuleList()
        for _ in range(self.radix):
            self.tconv.append(Conv(nn.Conv1d(self.out_planes, self.out_planes, kernel_size, 1, groups=self.out_planes, padding=kernel_size // 2, bias=False,),
                                   bn=nn.BatchNorm1d(self.out_planes), activation=None))
            kernel_size //= 2

        self.interFre = InterFre()

        self.power = LogPowerLayer(dim=3)
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        N, C, T = x.shape
        out = self.sconv(x)

        out = torch.split(out, self.out_planes, dim=1)
        out = [m(x) for x, m in zip(out, self.tconv)]
        out = self.interFre(out)
        out = out.reshape(N, self.out_planes, T // self.patch_size, self.patch_size)
        out = self.power(out)
        out = self.dp(out)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=2,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, num_heads):
        super().__init__(*[TransformerEncoderBlock(emb_size, num_heads) for _ in range(depth)])

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

class TransformerResize(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerResize, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Linear layer to map input_dim to output_dim
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        batch_size, features, branch_num = x.shape
        x = x.permute(2, 0, 1)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        out = x.flatten(1)
        return x, out



class FAConformer(nn.Module):
    def __init__(self, data_name, in_planes, out_planes_branch, out_planes_total, kernel_size, radix, patch_size, time_points, num_classes, depth_branch, num_heads_branch, depth_total, num_heads_total, num_bands, dim_feedforward):
        r'''FAConformer model
        :param in_planes: Number of input EEG channels
        :param out_planes_branch: Number of output feature dimensions in each branch
        :param out_planes_total: Number of output feature dimensions in total module
        :param kernel_size: Temporal convolution kernel size
        :param radix: Number of input frequency bands
        :param patch_size: Temporal pooling size
        :param time_points: Input window length
        :param num_classes: Number of classes
        :param depth_branch: Depth of Trans blocks in each branch
        :param num_heads_branch: Number of attention heads in each branch
        :param depth_total: Depth of Trans blocks in total module
        :param num_heads_total: Number of attention heads in total module
        :param num_bands: Number of frequency bands
        :param dim_feedforward: Hidden dimension of the feed-forward network (FFN) inside Transformer layers
        '''
        nn.Module.__init__(self)
        self.in_planes = in_planes * radix
        self.out_planes_branch = out_planes_branch
        self.out_planes_total = out_planes_total
        self.data_name = data_name
        self.depth_branch = depth_branch
        self.emb_size = time_points // patch_size
        self.num_heads_branch = num_heads_branch
        self.num_bands = num_bands
        self.dim_feedforward = dim_feedforward
        self.stem1 = Stem(self.data_name, self.in_planes, self.out_planes_branch, kernel_size, patch_size=patch_size, radix=radix)
        self.stem2 = Stem(self.data_name, self.in_planes, self.out_planes_branch, kernel_size, patch_size=patch_size, radix=radix)
        self.stem3 = Stem(self.data_name, self.in_planes, self.out_planes_branch, kernel_size, patch_size=patch_size, radix=radix)
        self.stem4 = Stem(self.data_name, self.in_planes, self.out_planes_branch, kernel_size, patch_size=patch_size, radix=radix)
        self.stem5 = Stem(self.data_name, self.in_planes, self.out_planes_branch, kernel_size, patch_size=patch_size, radix=radix)
        self.stem6 = Stem(self.data_name, self.in_planes, self.out_planes_branch, kernel_size, patch_size=patch_size, radix=radix)
        self.stem7 = Stem(self.data_name, self.in_planes, self.out_planes_branch, kernel_size, patch_size=patch_size, radix=radix)
        self.stem8 = Stem(self.data_name, self.in_planes, self.out_planes_branch, kernel_size, patch_size=patch_size, radix=radix)
        self.trans1 = TransformerEncoder(self.depth_branch, self.emb_size, self.num_heads_branch)
        self.trans2 = TransformerEncoder(self.depth_branch, self.emb_size, self.num_heads_branch)
        self.trans3 = TransformerEncoder(self.depth_branch, self.emb_size, self.num_heads_branch)
        self.trans4 = TransformerEncoder(self.depth_branch, self.emb_size, self.num_heads_branch)
        self.trans5 = TransformerEncoder(self.depth_branch, self.emb_size, self.num_heads_branch)
        self.trans6 = TransformerEncoder(self.depth_branch, self.emb_size, self.num_heads_branch)
        self.trans7 = TransformerEncoder(self.depth_branch, self.emb_size, self.num_heads_branch)
        self.trans8 = TransformerEncoder(self.depth_branch, self.emb_size, self.num_heads_branch)
        self.TransformerResize = TransformerResize(input_dim=out_planes_branch * (time_points // patch_size), output_dim=self.out_planes_total, num_heads=num_heads_total, num_layers=depth_total, dim_feedforward=self.dim_feedforward, dropout=0.1)
        self.fc = nn.Sequential( LinearWithConstraint(self.out_planes_total * self.num_bands, num_classes, doWeightNorm=True),)
        self.fc1 = nn.Sequential( LinearWithConstraint(out_planes_branch * (time_points // patch_size) * 1, num_classes, doWeightNorm=True),)
        self.fc2 = nn.Sequential( LinearWithConstraint(out_planes_branch * (time_points // patch_size) * 1, num_classes, doWeightNorm=True),)
        self.fc3 = nn.Sequential( LinearWithConstraint(out_planes_branch * (time_points // patch_size) * 1, num_classes, doWeightNorm=True),)
        self.fc4 = nn.Sequential( LinearWithConstraint(out_planes_branch * (time_points // patch_size) * 1, num_classes, doWeightNorm=True),)
        self.fc5 = nn.Sequential( LinearWithConstraint(out_planes_branch * (time_points // patch_size) * 1, num_classes, doWeightNorm=True),)
        self.fc6 = nn.Sequential( LinearWithConstraint(out_planes_branch * (time_points // patch_size) * 1, num_classes, doWeightNorm=True),)
        self.fc7 = nn.Sequential( LinearWithConstraint(out_planes_branch * (time_points // patch_size) * 1, num_classes, doWeightNorm=True),)
        self.fc8 = nn.Sequential( LinearWithConstraint(out_planes_branch * (time_points // patch_size) * 1, num_classes, doWeightNorm=True),)

    def forward(self, x):
        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]
        x3 = x[:, 2, :, :]
        x4 = x[:, 3, :, :]
        x5 = x[:, 4, :, :]
        x6 = x[:, 5, :, :]
        x7 = x[:, 6, :, :]
        x8 = x[:, 7, :, :]

        out1 = self.stem1(x1)
        out2 = self.stem2(x2)
        out3 = self.stem3(x3)
        out4 = self.stem4(x4)
        out5 = self.stem5(x5)
        out6 = self.stem6(x6)
        out7 = self.stem7(x7)
        out8 = self.stem8(x8)

        out1 = self.trans1(out1)
        out2 = self.trans2(out2)
        out3 = self.trans3(out3)
        out4 = self.trans4(out4)
        out5 = self.trans5(out5)
        out6 = self.trans6(out6)
        out7 = self.trans7(out7)
        out8 = self.trans8(out8)

        out1 = out1.flatten(1)
        out2 = out2.flatten(1)
        out3 = out3.flatten(1)
        out4 = out4.flatten(1)
        out5 = out5.flatten(1)
        out6 = out6.flatten(1)
        out7 = out7.flatten(1)
        out8 = out8.flatten(1)

        x1 = out1.unsqueeze(2)
        x2 = out2.unsqueeze(2)
        x3 = out3.unsqueeze(2)
        x4 = out4.unsqueeze(2)
        x5 = out5.unsqueeze(2)
        x6 = out6.unsqueeze(2)
        x7 = out7.unsqueeze(2)
        x8 = out8.unsqueeze(2)

        out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=2)
        fea, out = self.TransformerResize(out)

        out = self.fc(out)
        out1 = self.fc1(out1)
        out2 = self.fc2(out2)
        out3 = self.fc3(out3)
        out4 = self.fc4(out4)
        out5 = self.fc5(out5)
        out6 = self.fc6(out6)
        out7 = self.fc7(out7)
        out8 = self.fc8(out8)

        return out, out1, out2, out3, out4, out5, out6, out7, out8