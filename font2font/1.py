import functools
import numpy as np
import torch
import torch.nn as nn
from model.ViT_helper import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

import math


class matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = x1 @ x2
        return x


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Upsample(nn.Module):
    def __init__(self, nhidden, norm_nc):
        super().__init__()

        self.Pixup = nn.PixelShuffle(2)

        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=1, padding=0)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=1, padding=0)

    def forward(self, x, segmap, H, W, tag=True,out=False):
        B, N, C = x.size()

        assert N == H * W
        x = x.permute(0, 2, 1)  # 交换维度 [B,N,C]-> [B,C.N]

        x = x.view(-1, C, H, W)  # [B,C,N]->[B,C,H,W]
        if tag:
            segmap = F.interpolate(segmap, size=(H, W))
            gamma = self.mlp_gamma(segmap)
            beta = self.mlp_beta(segmap)
            x = x * (1 + gamma) + beta

        x = self.Pixup(x)  # 上采样  PixelShuffle(2) 参数为放大的倍数  [B,C,H,W]->[B,c/4,H*2,W*2]

        B, C, H, W = x.size()
        if out:
            return x, H, W
        x = x.view(-1, C, H * W)  # [B,c/4,H*2,W*2]->[B,c/4,H*2*W*2]

        x = x.permute(0, 2, 1)  # [B,c/4,H*2*W*2]->[B,H*2*W*2,c/4]

        return x, H, W


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def get_attn_mask(N, w):
    mask = torch.zeros(1, 1, N, N)  # .cuda()
    for i in range(N):
        if i <= w:
            mask[:, :, i, 0:i + w + 1] = 1
        elif N - i <= w:
            mask[:, :, i, i - w:N] = 1
        else:
            mask[:, :, i, i:i + w + 1] = 1
            mask[:, :, i, i - w:i] = 1
    return mask


class Attention_old(nn.Module):
    def __init__(self, dim, num_heads=8, is_mask=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(0.2)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.2)
        self.mat = matmul()
        self.is_mask = is_mask
        self.mask = get_attn_mask(is_mask, 8)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        if self.is_mask:
            # attn = attn.masked_fill(mask.to(attn.get_device()) == 0, -1e9)
            attn = attn.masked_fill(self.mask.to('cpu') == 0, -1e9)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


from torch import nn, einsum
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


# Transformer Encoder  Block 块
class Block(nn.Module):
    # dim 输入的维度
    # num_heads 多头注意力的头数
    # drop 为nn.Dropout 的参数

    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=num_heads)

        self.drop_path = DropPath(0)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=gelu, drop=0)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class UNetGenerator(nn.Module):
    def __init__(self, ):
        super(UNetGenerator, self).__init__()

        # self.input = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1)

        self.bottom_width = 8
        self.embed_dim = 1024
        self.Linear1 = nn.Linear(1024, (self.bottom_width ** 2) * self.embed_dim)
        self.block1_1 = Block(1024, 1)
        # self.block1_2 = Block(1024, 4)
        # self.block1_3 = Block(1024, 4)
        # self.block1_4 = Block(1024, 4)

        self.block2_1 = Block(1024 // 4, 1)
        self.block2_2 = Block(1024 // 4, 1)
        # self.block2_3 = Block(1024 // 4, 4)

        self.block3_1 = Block(1024 // 16, 2)
        self.block3_2 = Block(1024 // 16, 4)
        # self.block3_3 = Block(1024 // 16, 1)

        self.block4_1 = Block(1024 // 64, 2)
        self.block4_2 = Block(1024 // 64, 4)
        # self.block4_3 = Block(1024 // 64, 4)

        self.block5 = Block(1024 // 256, 1)

        self.upsample_1 = Upsample(3, 1024)

        self.upsample_2 = Upsample(3, 256)

        self.upsample_3 = Upsample(3, 64)

        self.upsample_4 = Upsample(3, 16)

        self.upsample_5 = Upsample(3, 4)

        self.out = nn.ConvTranspose2d(1, 3, kernel_size=1, padding=0)

        self.tanh = nn.Tanh()
        patch_dim = 3 * 8 ** 2
        from einops.layers.torch import Rearrange

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=8, p2=8),
            nn.Linear(patch_dim, 64), )

    def forward(self, x):
        segmap = x
        z = self.to_patch_embedding(x)
        z = z.permute(0, 2, 1)

        '''
        input=self.input(x)  #将输入的[bach,3,256,256] 转为 [bach,1,256,256]
        input= F.interpolate(input, size=(32, 32))  # 双线性插值 改变特征图的大小
        input = input.view(-1,1024)#[bach,1024]
        #使用 MLP 映射
        z= self.Linear1(input).view(-1, self.bottom_width **2 , self.embed_dim) #[bach,1024] #
	   '''
        # Transformer block
        z_1 = self.block1_1(z)
        # z_1 = self.block1_2(z_1)
        # z_1 = self.block1_3(z_1)
        # z_1 = self.block1_3(z_1)

        H = 8
        W = 8
        up_1, H, W = self.upsample_1(z_1, segmap, H, W)

        z_2 = self.block2_1(up_1)
        z_2 = self.block2_2(z_2)
        # z_2 = self.block2_3(z_2)

        up_2, H, W = self.upsample_2(z_2, segmap, H, W)

        z_3 = self.block3_1(up_2)
        z_3 = self.block3_2(z_3)
        # z_3 = self.block3_3(z_3)

        up_3, H, W = self.upsample_3(z_3, segmap, H, W)

        z_4 = self.block4_1(up_3)
        z_4 = self.block4_2(z_4)
        # z_4 = self.block4_3(z_4)

        up_4, H, W = self.upsample_4(z_4, segmap, H, W)

        z_5 = self.block5(up_4)
        # print("z_5", z_5.shape)

        up_5, H, W = self.upsample_5(z_5, segmap, H, W)
        # z=up_5
        #B, N, C = up_5.size()
        # up_5 = up_5.permute(0, 2, 1)  # 交换维度 [B,N,C]-> [B,C.N]

        #up_5 = up_5.view(-1, C, 256, 256)  # [B,C,N]->[B,C,H,W]

        out = self.out(up_5)
        out = self.tanh(out)

        return out, z


if __name__ == '__main__':
    print(__name__)

    model = UNetGenerator()
    print(model)
    dummy_input = torch.rand(3, 3, 256, 256)  # 假设输入20张1*28*28的图片
    model.forward(dummy_input)


