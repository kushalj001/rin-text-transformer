import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import sys
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange


class Config:
    def __init__(
        self,
        num_heads,
        head_dim,
        time_dim,
        attn_dropout,
        mult_factor,
        input_norm,
        context_norm,
        qk_norm
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.time_dim = time_dim
        self.mult_factor = mult_factor
        self.attn_dropout = attn_dropout
        self.input_norm = input_norm
        self.context_norm = context_norm
        self.qk_norm = qk_norm
        

def timestep_embedding(timesteps, dim, max_period=10000):
    # timesteps = [bs]
    # dim = 128
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    # 64
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    # exp{-(T * t)/64}, where T is the time period, t is the current time instant
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        # TODO: why cating zeros?
    return embedding


class LearnedSinusoidalPosEmb(nn.Module): # used in RIN (image modality)
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
    

class Attention(nn.Module):
    def __init__(
        self,
        output_dim:int, # the expected output dimensionality after cross attention (query dimensionality)
        context_dim:int, # dimensionality of the current stream of computation (kv dimensionality)
        num_heads:int,
        head_dim:int,
        time_dim:int,
        attn_dropout:float,
        input_norm:bool,
        context_norm:bool,
        qk_norm:bool
    ):

        super().__init__()
        
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        
        if context_dim is None:
            context_dim = output_dim
        
        if time_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_dim, output_dim*2),
                Rearrange('b d -> b 1 d')
            )

            # TODO: does this make sense for text?
            nn.init.zeros_(self.time_mlp[-2].weight)
            nn.init.zeros_(self.time_mlp[-2].bias)

        # TODO: needed for text?
        self.input_ln = nn.LayerNorm(output_dim) if input_norm else nn.Identity()
        self.context_ln = nn.LayerNorm(context_dim) if context_norm else nn.Identity()

        self.to_q = nn.Linear(output_dim, attn_dim, bias=False)
        # TODO: how is this different than separately projecting k and v
        self.to_kv = nn.Linear(context_dim, attn_dim*2, bias=False)
        self.to_out = nn.Linear(attn_dim, output_dim, bias=False)

    def attend(self, q, k, v, mask=None):
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if mask is not None: # TODO: check if we need masking
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        return out
        

    def forward(
        self,
        inp,
        context,
        time
    ):
        h = self.num_heads
        if context is not None:
            context = self.context_ln(context)

        inp = self.input_ln(inp)
        if context is None:
            context = inp

        scale, shift = self.time_mlp(time).chunk(2, dim=-1)
        inp = (inp * (scale + 1)) + shift

        qkv = (self.to_q(inp), *self.to_kv(context).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        out = self.attend(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)        
    

class FeedForwardLayer(nn.Module):
    def __init__(
        self,
        input_dim:int,
        time_dim:int,
        mult_factor:int
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        if time_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_dim, input_dim*2),
                Rearrange("b d -> b 1 d")
            )
            nn.init.zeros_(self.time_mlp[-2].weight)
            nn.init.zeros_(self.time_mlp[-2].bias)

        inner_dim = input_dim*mult_factor
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, input_dim)
        )

    def forward(self, x, time=None):

        x = self.norm(x)
        if time is not None:
            scale, shift = self.time_mlp(time).chunk(2, dim=-1)
            x = (x*(scale+1)) + shift

        return self.ffn(x)


class RINBlock(nn.Module):
    def __init__(
        self,
        config,
        interface_dim:int,
        num_latent_attn_layers:int,
        latent_dim:int,
        time_dim:int,
        final_norm:bool
    ):
        super().__init__()

        attn_kwargs = {
            "num_heads": config.num_heads,
            "head_dim": config.head_dim,
            "time_dim": config.time_dim,
            "attn_dropout": config.attn_dropout,
            "input_norm": config.input_norm,
            "context_norm": config.context_norm,
            "qk_norm": config.qk_norm
        }
        self.interface_to_latents_cross_attn = Attention(
            output_dim=latent_dim,
            context_dim=interface_dim,
            **attn_kwargs
        )

        self.latents_ffn = FeedForwardLayer(
            input_dim=latent_dim,
            time_dim=time_dim,
            mult_factor=config.mult_factor
        )

        self.latent_self_attn_layers = nn.ModuleList([])
        for _ in range(num_latent_attn_layers):
            self.latent_self_attn_layers.append(
                nn.ModuleList([
                    Attention(
                        output_dim=latent_dim,
                        context_dim=None,
                        **attn_kwargs
                    ),
                    FeedForwardLayer(
                        input_dim=latent_dim,
                        time_dim=time_dim,
                        mult_factor=config.mult_factor
                    )
                ])
            )

        self.latent_final_norm = nn.LayerNorm(latent_dim) if final_norm else nn.Identity()
        self.latents_to_interface_cross_attn = Attention(
            output_dim=interface_dim,
            context_dim=latent_dim,
            **attn_kwargs
        )
        self.interface_ffn = FeedForwardLayer(
            input_dim=interface_dim,
            time_dim=time_dim,
            mult_factor=config.mult_factor
        )

    def forward(
        self,
        interface,
        latents,
        time
    ):
        latents = self.interface_to_latents_cross_attn(latents, interface, time) + latents
        latents = self.latents_ffn(latents, time) + latents
        
        for attn, ffn in self.latent_self_attn_layers:
            latents = attn(latents, latents, time=time) + latents
            latents = ffn(latents, time) + latents
        
        interface = self.latents_to_interface_cross_attn(interface, latents, time) + interface
        interface = self.interface_ffn(interface, time) + interface
        
        latents = self.latent_final_norm(latents)
        
        return interface, latents


# Instead of code related to patches, we need to have something related to text.
# the process of converting image to patches and then to patch embedding needs to be replaced
# by something like word embeddings. 
# This could be byte embeddings or word embeddings (sub-word) to begin with.
class RIN(nn.Module): ### WIP ###
    def __init__(
        self,
        config,
        hidden_dim:int,
        num_rin_blocks:int,
        num_latent_attn_layers:int,
        vocab_size:int,
        interface_emb_dim:int, # embedding size of words in interface
        latent_dim:int,
        num_latents:int,
        learned_sinusoidal_dim:int, # TODO: check if needed
        latent_token_time_cond:bool,        
    ):
        super().__init__()
        self.latent_token_time_cond = latent_token_time_cond
        self.interface_embedding = nn.Embedding(vocab_size, interface_emb_dim)

        sinusoidal_pos_embedding = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        time_dim = hidden_dim*4
        fourier_dim = learned_sinusoidal_dim + 1
        self.time_mlp = nn.Sequential(
            sinusoidal_pos_embedding,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        nn.init.normal_(self.latents, std=0.02)
        self.initial_latents = nn.Sequential(
            FeedForwardLayer(latent_dim),
            nn.LayerNorm(latent_dim)
        )
        nn.init.zeros_(self.initial_latents[-1].gamma)

        self.rin_blocks = nn.ModuleList([
            RINBlock(
                config,
                hidden_dim,
                num_latent_attn_layers,
                latent_dim,
                time_dim,
                config.final_norm
            ) for _ in range(num_rin_blocks)
        ])
        self.unembed = nn.Linear(hidden_dim, vocab_size)
        # TODO: check if this should be tied to embedding weight

    def forward(self, input_tokens, time, input_self_cond, latent_self_cond, return_latents):
        # check if input_self_cond (input self conditioning - at interface) is done. 
        # I only remember it being done for latents. Not adding code for that right now.

        batch_size = input_tokens.shape[0]
        time_emb = self.time_mlp(time)

        latents = repeat(self.latents, 'n d -> b n d', b=batch_size)
        if latent_self_cond is not None:
            latents = latents + self.initial_latents(latent_self_cond)

        # TODO: not sure about the theory behind this. Just copied for now. Check if needed.
        if self.latent_token_time_cond:
            # whether the time conditioning is to be treated as one latent token or for 
            # projecting into scale and shift for adaptive layernorm
            time_emb = rearrange(time_emb, 'b d -> b 1 d')
            latents = torch.cat([latents, time_emb], dim=-2)

        interface_emb = self.interface_embedding(input_tokens)
        # TODO: add positional embedding for tokens

        for rin_block in self.rin_blocks:
            interface, latents = rin_block(interface_emb, latents, time_emb)

        out = self.unembed(interface)
        # need to take softmax to get the predicted tokens

        return out, latents