## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from .mamba import Mamba

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=2, groups=hidden_features*2, bias=bias, dilation=2)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.fft_channel_weight = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))
        self.fft_channel_bias = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw//factor+1)*factor-hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad
    def unpad(self, x, t_pad):
        hw = x.shape[-1]
        return x[...,t_pad[0]:hw-t_pad[1]]

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x, pad_w = self.pad(x,2)
        x = torch.fft.rfft2(x)
        x = self.fft_channel_weight * x + self.fft_channel_bias
#        x = torch.nn.functional.normalize(x, 1)
        x = torch.fft.irfft2(x)
        x = self.unpad(x, pad_w)
        x1, x2 = x.chunk(2, dim=1)
        
        x = F.silu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.factor = 2
        self.idx_dict = {}
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw//factor+1)*factor-hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad
    def unpad(self, x, t_pad):
        hw = x.shape[-1]
        return x[...,t_pad[0]:hw-t_pad[1]]
        
    def comp2real(self, x):
        b, _, h, w = x.shape
        return torch.cat([x.real, x.imag], 1)
#        return torch.stack([x.real, x.imag], 2).view(b,-1,h,w)
    def real2comp(self, x):
        xr, xi = x.chunk(2, dim=1)
        return torch.complex(xr, xi)

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit  = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def get_idx_map(self, h, w):
        l1_u = torch.arange(h//2).view(1,1,-1,1)
        l2_u = torch.arange(w).view(1,1,1,-1)
        half_map_u = l1_u @ l2_u
        l1_d = torch.arange(h - h//2).flip(0).view(1,1,-1,1)
        l2_d = torch.arange(w).view(1,1,1,-1)
        half_map_d = l1_d @ l2_d
        return torch.cat([half_map_u, half_map_d], 2).view(1,1,-1).argsort(-1)
    def get_idx(self, x):
        h, w = x.shape[-2:]
        if (h, w) in self.idx_dict:
            return self.idx_dict[(h, w)]
        idx_map = self.get_idx_map(h, w).to(x.device).detach()
        self.idx_dict[(h, w)] = idx_map
        return idx_map
    def attn(self, qkv):
        h = qkv.shape[2]
        q,k,v = qkv.chunk(3, dim=1)
        
        q, pad_w, idx = self.fft(q)
        q, pad = self.pad(q, self.factor)
        k, pad_w, _ = self.fft(k)
        k, pad = self.pad(k, self.factor)
        v, pad_w, _ = self.fft(v)
        v, pad = self.pad(v, self.factor)
        
        q = rearrange(q, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)
        k = rearrange(k, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)
        v = rearrange(v, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head (c factor) hw -> b (head c) (factor hw)', head=self.num_heads, factor=self.factor)
        out = self.unpad(out, pad)
        out = self.ifft(out, pad_w, idx, h)
        return out
    def fft(self, x):
        x, pad = self.pad(x, 2)
        x = torch.fft.rfft2(x.float(), norm="ortho")
        x = self.comp2real(x)
        idx = self.get_idx(x)
        b, c = x.shape[:2]
        x = x.contiguous().view(b, c, -1)
        x = torch.gather(x, 2, index=idx.repeat(b,c,1)) # b, 6c, h*(w//2+1)
        return x, pad, idx
    def ifft(self, x, pad, idx, h):
        b, c = x.shape[:2]
        x = torch.scatter(x, 2, idx.repeat(b,c,1), x)
        x = x.view(b, c, h, -1)
        x = self.real2comp(x)
        x = torch.fft.irfft2( x, norm='ortho' )#.abs()
        x = self.unpad(x, pad)
        return x
    def forward(self, x):
        b,c,h,w = x.shape

        attn_map = x

        qkv = self.qkv_dwconv(self.qkv(x))

#        qkv, pad_w, idx = self.fft(qkv)
#        qkv, pad = self.pad(qkv, self.factor)

        attn_map = qkv  
        out = self.attn(qkv) 
        attn_map = out


#        out = self.unpad(out, pad)
#        out = self.ifft(out, pad_w, idx, h)

        out = self.project_out(out)
        attn_map = out
        return out

    '''
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    '''


##########################################################################
class TransMambaBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransMambaBlock, self).__init__()

        self.trans_block = TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.mamba_block = MambaBlock(dim, LayerNorm_type)
        self.conv = nn.Conv2d(int(dim*2), dim, kernel_size=1, bias=bias) 
    def forward(self, x):
        x1 = self.trans_block(x)
        x2 = self.mamba_block(x)
        out = torch.cat((x1, x2), 1)
        out = self.conv(out)
        return out

    

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class MambaBlock(nn.Module):
    def __init__(self, dim, LayerNorm_type,):
        super(MambaBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.mamba1 = DRMamba(dim, reverse=False)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.mamba2 = DRMamba(dim, reverse=True)# FeedForward(dim, ffn_expansion_factor, bias, True)

    def forward(self, x):
        x = x + self.mamba1(self.norm1(x))

        x = x + self.mamba2(self.norm2(x))

        return x

class DRMamba(nn.Module):
    def __init__(self, dim, reverse):
        super(DRMamba, self).__init__()
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.reverse = reverse

    def forward(self, x):
        b,c,h,w = x.shape
        if self.reverse:
            x = x.flip(1)
        x = self.mamba(x)
        if self.reverse:
            x = x.flip(1)
        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
class TransMamba(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(TransMamba, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransMambaBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransMambaBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransMambaBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransMambaBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransMambaBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransMambaBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransMambaBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransMambaBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1

