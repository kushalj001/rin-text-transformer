#### ANNOTATED RIN FILE FROM LUCIDRAINS ####
import math
from pathlib import Path
from random import random
from functools import partial
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.special import expm1
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from beartype import beartype

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from rin_pytorch.attend import Attend

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator, DistributedDataParallelKwargs

# helpers functions

def exists(x):
    return x is not None

def identity(x):
    return x

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def safe_div(numer, denom, eps = 1e-10):
    return numer / denom.clamp(min = eps)

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    num_sqrt = math.sqrt(num)
    return int(num_sqrt) == num_sqrt

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# use layernorm without bias, more stable

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class MultiHeadedRMSNorm(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# positional embeds

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim, # 256 (patches embedding dim since usually applied on patches)
        heads = 4,
        dim_head = 32,
        norm = False, # True
        qk_norm = False,
        time_cond_dim = None # 1024
    ):
        super().__init__()
        hidden_dim = dim_head * heads # 128
        self.scale = dim_head ** -0.5 # 1/sqrt(32)
        self.heads = heads # 4

        self.time_cond = None

        if exists(time_cond_dim):
            # taken
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                # [..., 1024] => [..., 512]
                Rearrange('b d -> b 1 d')
                # [bs, 1, latent_dim]
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)
            # initialize w and b of the linear layer with zeros.

        self.norm = LayerNorm(dim) if norm else nn.Identity()
        # 256

        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        # 256 => 128*3

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = MultiHeadedRMSNorm(dim_head, heads)
            self.k_norm = MultiHeadedRMSNorm(dim_head, heads)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(
        self,
        x,
        time = None
    ):
        # x = patches = [bs, p_h*p_w, dim]
        # time = t = [bs, time_dim]

        h = self.heads # 4
        x = self.norm(x)
        # [bs, p_h*p_w, dim]

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            # [bs, 1, 256], [bs, 1, 256]
            x = (x * (scale + 1)) + shift
            # [bs, p_h*p_w, dim]

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # ([bs, p_h*p_w, 128]) * 3

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # q = [bs, heads, p_h*p_w, head_dim]
        # k = [bs, heads, p_h*p_w, head_dim]
        # v = [bs, heads, p_h*p_w, head_dim]

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.softmax(dim = -1)
        # [bs, heads, p_h*p_w, head_dim]
        k = k.softmax(dim = -2)
        # [bs, heads, p_h*p_w, head_dim]
       
        q = q * self.scale
        # [bs, heads, p_h*p_w, head_dim]

        context = torch.einsum('b h n d, b h n e -> b h d e', k, v)
        # [bs, heads, head_dim, head_dim]
        # What is this intuitively?

        out = torch.einsum('b h d e, b h n d -> b h n e', context, q)
        # [bs, heads, p_h*p_w, head_dim]

        out = rearrange(out, 'b h n d -> b n (h d)')
        # [bs, p_h*p_w, hidden_dim]

        return self.to_out(out) # [bs, ph*p_w, dim]

class Attention(nn.Module):
    def __init__(
        self,
        dim, # 512 (this is actually latent_dim when projecting from patches to latents)
        dim_context = None, # 256
        heads = 4,
        dim_head = 32,
        norm = False, # True
        norm_context = False, # True
        time_cond_dim = None, # 1024 (passed via attn_kwargs)
        flash = False,
        qk_norm = False
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        # hidden_dim = 32 * 4 = 128

        dim_context = default(dim_context, dim)
        # 256

        self.time_cond = None

        if exists(time_cond_dim):
            # branch taken, time_cond_dim=1024
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                # [bs, 1024] => [bs, 1024]
                Rearrange('b d -> b 1 d')
                # [bs, 1, time_cond_dim] = [bs, 1, 1024]
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)
            # initialize weight and bias of linear layer with zeros

        self.scale = dim_head ** -0.5 # 1/sqrt(32)
        self.heads = heads # 4

        self.norm = LayerNorm(dim) if norm else nn.Identity()
        # 512
        self.norm_context = LayerNorm(dim_context) if norm_context else nn.Identity()
        # 256

        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        # [... , 512] => [... , 128]
        self.to_kv = nn.Linear(dim_context, hidden_dim * 2, bias = False)
        # [... , 256] => [..., 256]
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)
        # [... , 128] => [..., 512]

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = MultiHeadedRMSNorm(dim_head, heads)
            self.k_norm = MultiHeadedRMSNorm(dim_head, heads)

        self.attend = Attend(flash = flash)

    def forward(
        self,
        x,
        context = None,
        time = None
    ):
        # x = latents = [b, num_latents, latent_dim]
        # context = patches = [b, num_patches, dim_context]
        # t = time = [b, time_dim]
        h = self.heads # 4

        if exists(context): # TODO: is there any scenario where context is not present?
            # taken
            context = self.norm_context(context)
            # [bs, num_patches, dim_context]

        x = self.norm(x)
        # [bs, num_latents, latent_dim]

        context = default(context, x)
        # [bs, num_patches, dim_context]

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            # applies a SilU and a linear layer which does not change the dimensionality of a single
            # vector present in the tensor. Finally adds an additional dimension.
            # [bs, time_dim] => [bs, 1, time_dim] => [bs, 1, time_dim//2] (chunk) => [bs, 1, 512]
            # for each example, get a scale and shift vector of 512 dimensions.
            x = (x * (scale + 1)) + shift
            # [bs, num_latents, latent_dim]
            # TODO: what is this? why add 1 to the scale? what do scale and shift mean?

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        # ([bs, num_latents, 128], [bs, num_patches, 128], [bs, num_patches, 128])
        # ([bs, num_latents, head_dim*num_heads],..., ...)
        # latents are passed to query and patches are passed to key and value.
        # attention score is the amount of attention query token pays to the key token at any position.
        # So when the multiplication for q and k is a high value, it means that the latent (q) at that
        # position is looking for information that patch (k) had. Basically greater the similarity content
        # between the latents and the patches, greater the attn probability.

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # q = [bs, head, num_latents, head_dim]
        # k = [bs, head, num_patches, head_dim]
        # v = [bs, head, num_patches, head_dim]
        # num_latents is equivalent to seq_len.


        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        out = self.attend(q, k, v)
        # [bs, heads, seq_q, head_dim]
        # seq_q = num_latents

        out = rearrange(out, 'b h n d -> b n (h d)')
        # [bs, num_latents, hidden_dim]

        return self.to_out(out) # [bs, num_latents, latent_dim]

class PEG(nn.Module):
    def __init__(
        self,
        dim # 256
    ):
        super().__init__()
        self.ds_conv = nn.Conv2d(dim, dim, 3, padding = 1, groups = dim)

    def forward(self, x):
        # x = patches = [bs, p_h*p_w, dim]
        b, n, d = x.shape
        hw = int(math.sqrt(n))
        # 16
        x = rearrange(x, 'b (h w) d -> b d h w', h = hw)
        # [bs, dim, p_h, p_w]
        x = self.ds_conv(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        # [bs, ph*pw, dim]
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, time_cond_dim = None):
        super().__init__()
        self.norm = LayerNorm(dim)

        self.time_cond = None

        if exists(time_cond_dim):
            # never taken
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x, time = None):
        x = self.norm(x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        return self.net(x)

# model

class RINBlock(nn.Module):
    def __init__(
        self,
        dim, # 256
        latent_self_attn_depth, # 4
        dim_latent = None, # 512
        final_norm = True,
        patches_self_attn = True, # True
        **attn_kwargs # time_dim = 1024
    ):
        super().__init__()
        dim_latent = default(dim_latent, dim)
        # 512

        self.latents_attend_to_patches = Attention(
            dim_latent, dim_context = dim, norm = True, norm_context = True, **attn_kwargs
        )
        # cross attention between patches and latent. Here patch dimensions (larger) get 
        # compressed to latent dimensions (smaller).
        # dim_latent = 512, dim_context = 256

        self.latents_cross_attn_ff = FeedForward(dim_latent)

        self.latent_self_attns = nn.ModuleList([])
        for _ in range(latent_self_attn_depth):
            self.latent_self_attns.append(nn.ModuleList([
                Attention(dim_latent, norm = True, **attn_kwargs),
                FeedForward(dim_latent)
            ]))
            # add 4 blocks of self attention and ff layer to be computed on the latent side.

        self.latent_final_norm = LayerNorm(dim_latent) if final_norm else nn.Identity()
        # 512

        self.patches_peg = PEG(dim)
        # does a convolution over the image of patches, that is over 16X16 image.
        # just consider this as an isolated convolution operation that does not make any
        # tensor shape changes. Mostly would be irrelevant for text.

        self.patches_self_attn = patches_self_attn # true

        if patches_self_attn:
            self.patches_self_attn = LinearAttention(dim, norm = True, **attn_kwargs)
            self.patches_self_attn_ff = FeedForward(dim)

        self.patches_attend_to_latents = Attention(dim, dim_context = dim_latent, norm = True, norm_context = True, **attn_kwargs)
        self.patches_cross_attn_ff = FeedForward(dim)

    def forward(self, patches, latents, t):
        # patches = [b, ph*pw, dim]
        # latents = [b, num_latents, latent_dim]
        # In terms of its similarity with tensors for text sequences, you can think 
        # of the 2nd dimesnion being seq_len for both patches and latents (which it actually is).
        # The 2nd dim in patches is ph*pw => num_patches, and num_latents in latents.
        # So the transformation from patches to latents involves compressing information
        # along 2 axes, one across the sequence length (number of tokens) and the second 
        # across the dimensionality of each token embedding. In this running example, we're only
        # compressing across the first dimension and expanding across the second dimension
        # (which kind of does not make sense lol). 
        # t = [b, time_dim]

        patches = self.patches_peg(patches) + patches
        # [bs, p_h*p_w, dim]
        # latents extract or cluster information from the patches

        latents = self.latents_attend_to_patches(latents, patches, time = t) + latents
        # [bs, num_latents, latent_dim]

        latents = self.latents_cross_attn_ff(latents, time = t) + latents
        # [bs, num_latents, latent_dim]

        # latent self attention

        for attn, ff in self.latent_self_attns:
            latents = attn(latents, time = t) + latents
            latents = ff(latents, time = t) + latents
            # [bs, num_latents, latent_dim]

        if self.patches_self_attn:
            # additional patches self attention with linear attention

            patches = self.patches_self_attn(patches, time = t) + patches
            # [bs, p_h*p_w, dim]
            # TODO: didn't understand what this layer did and how it is different to the usual
            # self attention. Had some weird computations.
            patches = self.patches_self_attn_ff(patches) + patches
            # [bs, p_h*p_w, dim]

        # patches attend to the latents

        patches = self.patches_attend_to_latents(patches, latents, time = t) + patches
        # [bs, p_h*p_w, dim]

        patches = self.patches_cross_attn_ff(patches, time = t) + patches
        # [bs, p_h*p_w, dim]

        latents = self.latent_final_norm(latents)
        return patches, latents

class RIN(nn.Module):
    def __init__(
        self,
        dim,                           # 256, model dimensions (does this mean hidden layer dims?)
        image_size,                    # 128, image h and w
        patch_size = 8,       ## matching repo input
        channels = 3,
        depth = 6,                      # number of RIN blocks
        latent_self_attn_depth = 4,     # matching repo input
                                        # how many self attentions for the latent per each round of cross
                                        # attending from pixel space to latents and back
        dim_latent = 512,               ## matching repo input
                                        # will default to image dim (dim)
        num_latents = 128,     ## matching repo input
                                        # they still had to use a fair amount of latents for good results (256), 
                                        # in line with the Perceiver line of papers from Deepmind
        learned_sinusoidal_dim = 16,
        latent_token_time_cond = False, # whether to use 1 latent token as time conditioning, 
        # or do it the adaptive layernorm way 
        # (which is highly effective as shown by some other papers "Paella" - Dominic Rampas et al.)
        dual_patchnorm = True,
        patches_self_attn = True,   # the self attention in this repository 
                                    # is not strictly with the design proposed in the paper. 
                                    # offer way to remove it, in case it is the source of instability
        **attn_kwargs
    ):
        super().__init__()
        assert divisible_by(image_size, patch_size)
        dim_latent = default(dim_latent, dim)
        # defaults to dim = 512, can be greater than image dim for greater capacity
        # Q: what is difference between dim, dim_latent and num_latents

        self.image_size = image_size
        # integer denoting image h and w, 128 e.g.

        self.channels = channels 
        # 3
        # times 2 due to self-conditioning (why?)

        patch_height_width = image_size // patch_size
        # 128 // 8 = 16
        # NOTE: patch_height_width is unintuitive name, because patch height width is 8X8!
        num_patches = patch_height_width ** 2 # 16 patches across height and 16 across width
        # 256
        pixel_patch_dim = channels * (patch_size ** 2) 
        # 3 * 8^2 = 3 * 64 = 192
        # what does this signify? think its the number of pixels in a patch across all the channels
        # iirc: this is an intermediate dimension that is used to transform the image pixels to 
        # an embedding format; which can be further processed by the model layers.
       

        # time conditioning

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim) # 16
        # TODO: understand what learned sinusoidal pos embedding does.
        time_dim = dim * 4
        # 256 * 4 = 1024
        fourier_dim = learned_sinusoidal_dim + 1
        # 16 + 1 = 17

        self.latent_token_time_cond = latent_token_time_cond 
        # False; not sure what this means, or signifies
        time_output_dim = dim_latent if latent_token_time_cond else time_dim
        # 512 or 1024

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            # (b, 17)
            nn.Linear(fourier_dim, time_dim),
            # W = (17, 1024); out = (b, 1024)
            nn.GELU(),
            # (b, 1024)
            nn.Linear(time_dim, time_output_dim)
            # W = (1024, 1024); out = (b, 1024)
        )
        # pixels to patch and back

        self.to_patches = Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size),
            # (b, c, h, w) = (b, 3, 128, 128) => (b, (16*16), (3*8*8)) = [b, 256, 192]
            # imagine an image with 3 channels, and each channel is an image of 128X128. 
            # For each channel image, we want to divide it into 256 patches, each of size 8X8.
            # You can also think about creating patches of the entire image, with the channels all 
            # combined. 
            # So we technically get 256 patches in total, each of size 8X8X3. So each patch is
            # an embedding of dim 192.
            nn.LayerNorm(pixel_patch_dim * 2) if dual_patchnorm else None,
            # dual_patchnorm=True
            # out = (b, 256, 192 * 2) = (b, 256, 384)
            nn.Linear(pixel_patch_dim * 2, dim),
            # (b, 256, 256)
            nn.LayerNorm(dim) if dual_patchnorm else None,
            # (b, 256, 256)
        )

        # axial positional embeddings, parameterized by an MLP

        pos_emb_dim = dim // 2
        # 128

        self.axial_pos_emb_height_mlp = nn.Sequential(
            # input is height_range = [16]
            # positional embedding for patches 
            # this block embeds the patches along the height dimension
            # takes in a list of numbers from 1 to 16 and learns an embedding for each pos.
            Rearrange('... -> ... 1'),
            # [16, 1]
            nn.Linear(1, pos_emb_dim),
            # [16, 128]
            nn.SiLU(),
            nn.Linear(pos_emb_dim, pos_emb_dim),
            # [16, 128]
            nn.SiLU(),
            nn.Linear(pos_emb_dim, dim)
            # [16, 256]
        )
        
        # same as above
        self.axial_pos_emb_width_mlp = nn.Sequential(
            Rearrange('... -> ... 1'),
            nn.Linear(1, pos_emb_dim),
            nn.SiLU(),
            nn.Linear(pos_emb_dim, pos_emb_dim),
            nn.SiLU(),
            nn.Linear(pos_emb_dim, dim)
            # [16, 256]
        )

        # nn.Parameter(torch.randn(2, patch_height_width, dim) * 0.02)

        self.to_pixels = nn.Sequential(
            LayerNorm(dim),
            # [b, 256, 256]
            nn.Linear(dim, pixel_patch_dim),
            # [b, 256, 192]
            Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size, h = patch_height_width)
            # [b, (16, 16), (3, 8, 8)] -> [b, 3, 128, 128]
        )
        # projecting it back to the interface space, or pixel space

        self.latents = nn.Parameter(torch.randn(num_latents, dim_latent))
        # W = (128, 512)
        # 128 latent variables each of which is a 512-dimensional vector
        nn.init.normal_(self.latents, std = 0.02)

        self.init_self_cond_latents = nn.Sequential(
            FeedForward(dim_latent), # typical feedforward layer that does projection to a 
            # higher dim space and then projects back to the original model dim subspace.
            # so could be something like 512 -> 2048 -> 512
            LayerNorm(dim_latent)
        )

        nn.init.zeros_(self.init_self_cond_latents[-1].gamma)

        # the main RIN body parameters  - another attention is all you need moment

        if not latent_token_time_cond:
            attn_kwargs = {**attn_kwargs, 'time_cond_dim': time_dim} 
            # time_dim = 1024

        self.blocks = nn.ModuleList([RINBlock(dim, 
                                        dim_latent = dim_latent, 
                                        latent_self_attn_depth = latent_self_attn_depth, 
                                        patches_self_attn = patches_self_attn, **attn_kwargs) 
                                        for _ in range(depth)])
        # dim = 256, dim_latent = 512,
        # latent_self_attn_depth = 4 (number of self-attn + MLP layers on latent side)
        # patches_self_attn = True
        # depth = 6 (number of RIN blocks)
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x, # (b, c, h, w) = (b, c, 128, 128)
        time, # int; [b]
        x_self_cond = None,
        latent_self_cond = None,
        return_latents = False
    ):
        batch = x.shape[0]

        x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        # [b, c, 128, 128]
        # initialized to be zeros if None is passed in `x_self_cond`. 
        # The default function returns the first argument if it not None or executes the
        # function passed in the second arg if first arg is None.
        # TODO: check how this is done in the paper.

        x = torch.cat((x_self_cond, x), dim = 1)
        # [b, c*2, 128, 128] = [b,c*2, 128, 128]

        # prepare time conditioning

        t = self.time_mlp(time)
        # [bs] => sinusoidal_emb [bs, 17] => linear [bs, 1024] => Gelu => linear [bs, 1024]
        # if batched, [b, 1024] (batched based on operations below)

        # prepare latents

        latents = repeat(self.latents, 'n d -> b n d', b = batch)
        # [n, d] = [num_latents, latent_dim]
        # [b, n, d] = [b, num_latents, latent_dim] = [b, 128, 512]
        # repeat the latents tensor bs times. Does it imply that for each training example, we have
        # a dedicated latent weight matrix or just that we are using the same latent matrix for
        # each computation and hence it is common for all examples?

        # the warm starting of latents as in the paper

        if exists(latent_self_cond):
            latents = latents + self.init_self_cond_latents(latent_self_cond)
            # [b, n, d]

        ## not sure what this is

        # whether the time conditioning is to be treated as one latent token or for 
        # projecting into scale and shift for adaptive layernorm

        if self.latent_token_time_cond:
            t = rearrange(t, 'b d -> b 1 d')
            # [b, time_dim] -> [b, 1, time_dim]
            latents = torch.cat((latents, t), dim = -2)
            # if latent_token_time_cond = True, time_dim = latent_dim
            # [b, n+1, latent_dim]

        # to patches

        patches = self.to_patches(x)
        # patches = [b, p_h*p_w, dim] = [b, 256, 256] ; p_h,w = patch height and width or patch_size
        # Sequential module of a rearrange operation that takes in an image in its raw form and then
        # applies a LN => Linear => LN. 
        # input = [b, c, 128, 128] = [b, 6, 128, 128]
        # => [b, 256, 384] (Rearrange) => [b, 256, 384] (LN) => [b, 256, 256] (Linear) => [b, 256, 256] (LN)

        height_range = width_range = torch.linspace(0., 1., steps = int(math.sqrt(patches.shape[-2])), device = self.device)
        # height_range = width_range = [16]
        pos_emb_h, pos_emb_w = self.axial_pos_emb_height_mlp(height_range), self.axial_pos_emb_width_mlp(width_range)
        # [16, 256], [16, 256]
        # axial positional embedding: add a batch dimension to the height and width range list ([0,1,.., 16]),
        # then pass through 3 linear layers interspersed with SiLU activation.
        # Each position is projected to pos_emb_dim (128) and then to dim (256).
        # Thinking of this just as positional embeddings for the patches.

        pos_emb = rearrange(pos_emb_h, 'i d -> i 1 d') + rearrange(pos_emb_w, 'j d -> 1 j d')
        # h = [16, 1, 256], w = [1, 16, 256]
        # h + w = [16, 16, 256]
        # TODO: not sure how this is working intuitively

        patches = patches + rearrange(pos_emb, 'i j d -> (i j) d')
        # [b, ph*pw, dim] + [dim, dim]
        # [b, dim, dim]
        # adds positional embeddings to patches


        ## summary
        # 1. convert discrete time values to time embeddings
        # 2. initialize latents
        # 3. convert raw images to patch embeddings 
        # 4. create position embeddings for patches. The sequence length or number of patches is determined
        # by taking square root of the patch_h*patch_w (-2nd dimension of patches tensor)
        # Add these position embeddings to patches.
        

        # patches have shape [bs, patch_h*patch_w, dim]
        # 128 x 128 image has been converted to 16 x 16 representation. 
        # patch_size = 8, so for each 8X8 patch in the original image, we have an embedding in the above
        # tensor. Each 8X8 patch in the original image is now a single pixel in the 16X16 representation.
        # And for each such patch we have an embedding of `dim` size.
        # Each patch can be analogously thought of as a token in a sequence and similarly the 
        # patch emebdding as just a token embedding.        

        # the recurrent interface network body

        for block in self.blocks:
            # patches = [b, p_h*p_w, dim]
            # latents = [b, num_latents, latent_dim]
            # t = [b, time_dim]
            patches, latents = block(patches, latents, t)
            # patches = [bs, p_h*p_w, dim]
            # latents = [bs, num_latents, latent_dim]

        # to pixels

        pixels = self.to_pixels(patches)
        # => [bs, p_h*p_w, pixel_patch_dim] = [bs, 256, 192] (linear) => [bs, c, 128, 128] (rearrange)

        if not return_latents:
            return pixels

        # remove time conditioning token, if that is the settings

        if self.latent_token_time_cond:
            latents = latents[:, :-1]
            # [bs, num_latents - 1, latent_dim]
            # equivalent to removing the last predicted token

        return pixels, latents

# normalize and unnormalize image

def normalize_img(x):
    return x * 2 - 1

def unnormalize_img(x):
    return (x + 1) * 0.5

# normalize variance of noised image, if scale is not 1

def normalize_img_variance(x, eps = 1e-5):
    std = reduce(x, 'b c h w -> b 1 1 1', partial(torch.std, unbiased = False))
    return x / std.clamp(min = eps)

# helper functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    # 3
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))
    # [bs, 1, 1, 1]

# noise schedules

def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)

# converting gamma to alpha, sigma or logsnr

def gamma_to_alpha_sigma(gamma, scale = 1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)

def gamma_to_log_snr(gamma, scale = 1, eps = 1e-5):
    return log(gamma * (scale ** 2) / (1 - gamma), eps = eps)

# gaussian diffusion

@beartype
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: RIN,
        *,
        timesteps = 1000,
        use_ddim = True,
        noise_schedule = 'sigmoid',
        objective = 'v',
        schedule_kwargs: dict = dict(),
        time_difference = 0.,
        min_snr_loss_weight = True,
        min_snr_gamma = 5,
        train_prob_self_cond = 0.9,
        scale = 1. # this will be set to < 1. for better convergence when training on higher resolution images
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels #

        assert objective in {'x0', 'eps', 'v'}, 'objective must be either predict x0 or noise'
        self.objective = objective # v TODO: what does v stand for?

        self.image_size = model.image_size # 128

        if noise_schedule == "linear":
            self.gamma_schedule = simple_linear_schedule
        elif noise_schedule == "cosine":
            self.gamma_schedule = cosine_schedule
        elif noise_schedule == "sigmoid": # taken
            self.gamma_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        # the main finding presented in Ting Chen's paper - that higher resolution 
        # images requires more noise for better training

        assert scale <= 1, 'scale must be less than or equal to 1'
        self.scale = scale # 1
        self.maybe_normalize_img_variance = normalize_img_variance if scale < 1 else identity
        # identity

        # gamma schedules

        self.gamma_schedule = partial(self.gamma_schedule, **schedule_kwargs)

        self.timesteps = timesteps # 1000
        self.use_ddim = use_ddim # true

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the 
        # number of sampling timesteps is < 400

        ## TODO: didn't get the above^

        self.time_difference = time_difference # 0

        # probability for self conditioning during training

        self.train_prob_self_cond = train_prob_self_cond # 0.9
        # what is this? maybe the probability of self conditioning, that is we don't do it
        # for every training example or batch, but do it for sampled number of examples. More like 
        # dropout or teacher-forcing.

        # min snr loss weight

        self.min_snr_loss_weight = min_snr_loss_weight # true
        self.min_snr_gamma = min_snr_gamma # 5

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    @torch.no_grad()
    def ddpm_sample(self, shape, time_difference = None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        img = torch.randn(shape, device=device)

        x_start = None
        last_latents = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.timesteps):

            # add the time delay

            time_next = (time_next - self.time_difference).clamp(min = 0.)

            noise_cond = time

            # get predicted x0

            maybe_normalized_img = self.maybe_normalize_img_variance(img)
            model_output, last_latents = self.model(maybe_normalized_img, noise_cond, x_start, last_latents, return_latents = True)

            # get log(snr)

            gamma = self.gamma_schedule(time)
            gamma_next = self.gamma_schedule(time_next)
            gamma, gamma_next = map(partial(right_pad_dims_to, img), (gamma, gamma_next))

            # get alpha sigma of time and next time

            alpha, sigma = gamma_to_alpha_sigma(gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(gamma_next, self.scale)

            # calculate x0 and noise

            if self.objective == 'x0':
                x_start = model_output

            elif self.objective == 'eps':
                x_start = safe_div(img - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * img - sigma * model_output

            # clip x0

            x_start.clamp_(-1., 1.)

            # derive posterior mean and variance

            log_snr, log_snr_next = map(gamma_to_log_snr, (gamma, gamma_next))

            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # get noise

            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )

            img = mean + (0.5 * log_variance).exp() * noise

        return unnormalize_img(img)

    @torch.no_grad()
    def ddim_sample(self, shape, time_difference = None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        img = torch.randn(shape, device = device)

        x_start = None
        last_latents = None

        for times, times_next in tqdm(time_pairs, desc = 'sampling loop time step'):

            # get times and noise levels

            gamma = self.gamma_schedule(times)
            gamma_next = self.gamma_schedule(times_next)

            padded_gamma, padded_gamma_next = map(partial(right_pad_dims_to, img), (gamma, gamma_next))

            alpha, sigma = gamma_to_alpha_sigma(padded_gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(padded_gamma_next, self.scale)

            # add the time delay

            times_next = (times_next - time_difference).clamp(min = 0.)

            # predict x0

            maybe_normalized_img = self.maybe_normalize_img_variance(img)
            model_output, last_latents = self.model(maybe_normalized_img, times, x_start, last_latents, return_latents = True)

            # calculate x0 and noise

            if self.objective == 'x0':
                x_start = model_output

            elif self.objective == 'eps':
                x_start = safe_div(img - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * img - sigma * model_output

            # clip x0

            x_start.clamp_(-1., 1.)

            # get predicted noise

            pred_noise = safe_div(img - alpha * x_start, sigma)

            # calculate x next

            img = x_start * alpha_next + pred_noise * sigma_next

        return unnormalize_img(img)

    @torch.no_grad()
    def sample(self, batch_size = 16):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))

    def forward(self, img, *args, **kwargs):
        batch, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # batch = 32 e.g.
        # c = 3, h = 128, w = 128, img_size = 128
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # sample random times
        times = torch.zeros((batch,), device = device).float().uniform_(0, 1.)
        # [bs]
        # randomly sampled times between 0 and 1. They don't adhere to any schedule as of yet.

        # convert image to bit representation
        img = normalize_img(img)
        # does img * 2 - 1. Not sure why.

        # noise sample
        noise = torch.randn_like(img)
        # [bs, c, h, w]

        gamma = self.gamma_schedule(times)
        # [bs]; the plot of values in gamma still don't make sense. I don't think plotting
        # values for a batch would make sense anyways because these are values for different 
        # samples at the current time instant and not something that varies over time. So there should
        # be no pattern. 
        padded_gamma = right_pad_dims_to(img, gamma)
        # [bs, 1, 1, 1]
        alpha, sigma =  gamma_to_alpha_sigma(padded_gamma, self.scale)
        # alpha = sigma = [bs, 1,1,1]
        # upon plotting alpha and sigma are mirror image values of each other.

        noised_img = alpha * img + sigma * noise
        # [bs, c, h,w]

        noised_img = self.maybe_normalize_img_variance(noised_img)
        # identity function called, same as above

        # in the paper, they had to use a really high probability of 
        # latent self conditioning, up to 90% of the time
        # slight drawback

        self_cond = self_latents = None

        if random() < self.train_prob_self_cond:
            with torch.no_grad():
                model_output, self_latents = self.model(noised_img, times, return_latents = True)
                # model_output = [bs, c, 128, 128], 
                # self_latents = [bs, num_latents, latent_dim]
                # returns latents, because return_latents is True

                self_latents = self_latents.detach()
                # no gradients

                if self.objective == 'x0':
                    self_cond = model_output

                elif self.objective == 'eps':
                    self_cond = safe_div(noised_img - sigma * model_output, alpha)

                elif self.objective == 'v':
                    self_cond = alpha * noised_img - sigma * model_output
                    # [bs, c, h, w]
                    # TODO: what does this operation do?

                self_cond.clamp_(-1., 1.)
                self_cond = self_cond.detach()

        # predict and take gradient step

        pred = self.model(noised_img, times, self_cond, self_latents)
        # pred = [bs, c, h, w]; no latents

        if self.objective == 'eps':
            target = noise

        elif self.objective == 'x0':
            target = img

        elif self.objective == 'v':
            target = alpha * noise - sigma * img

        loss = F.mse_loss(pred, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        # min snr loss weight

        snr = (alpha * alpha) / (sigma * sigma)
        maybe_clipped_snr = snr.clone()

        if self.min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = self.min_snr_gamma)

        if self.objective == 'eps':
            loss_weight = maybe_clipped_snr / snr

        elif self.objective == 'x0':
            loss_weight = maybe_clipped_snr

        elif self.objective == 'v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        return (loss * loss_weight).mean()

# dataset classes

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class

@beartype
class Trainer(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        max_grad_norm = 1.,
        ema_update_every = 10,
        ema_decay = 0.995,
        betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no',
            kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
        )

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = betas)

        # for logging results in a folder periodically

        self.results_folder = Path(results_folder)

        if self.accelerator.is_local_main_process:
            self.results_folder.mkdir(exist_ok = True)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)


        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step + 1,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                # save milestone on every local main process, sample only on global main process

                if accelerator.is_local_main_process:
                    milestone = self.step // self.save_and_sample_every
                    save_and_sample = self.step != 0 and self.step % self.save_and_sample_every == 0
                    
                    if accelerator.is_main_process:
                        self.ema.to(device)
                        self.ema.update()

                        if save_and_sample:
                            self.ema.ema_model.eval()

                            with torch.no_grad():
                                batches = num_to_groups(self.num_samples, self.batch_size)
                                all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                            all_images = torch.cat(all_images_list, dim = 0)
                            utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                    if save_and_sample:
                        self.save(milestone)

                self.step += 1
                pbar.update(1)

        accelerator.print('training complete')
