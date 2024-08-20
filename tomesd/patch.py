import copy
import csv
import datetime
import math
import os
import warnings
from functools import reduce
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from rich.console import Console
from rich.table import Table
from sympy import igcd as gcd
from sympy import primefactors

from opensora.models.layers.blocks import t2i_modulate

from . import merge
from .utils import init_generator, isinstance_str


def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any], ratio: float) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]

    if ratio > 0 and downsample <= args["max_downsample"]:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        
        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        m, u = merge.bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], ratio, num_token_groups=1,
                                                      no_rand=not use_rand, generator=args["generator"])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)

    m_i, u_i = (m, u) if args["merge_in"]      else (merge.do_nothing, merge.do_nothing)
    m_a, u_a = (m, u) if args["merge_attn"]      else (merge.do_nothing, merge.do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (merge.do_nothing, merge.do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"]       else (merge.do_nothing, merge.do_nothing)

    return m_i, m_a, m_c, m_m, u_i, u_a, u_c, u_m  # Okay this is probably not very good


def compute_merge_temp(x: torch.Tensor, tome_info: Dict[str, Any], ratio: float) -> Tuple[Callable, ...]:
    original_t = tome_info["num_frames"]
    original_tokens = original_t
    downsample = original_tokens // x.shape[1]

    args = tome_info["args"]

    if ratio > 0 and downsample <= args["max_downsample"]:
        h, w = tome_info["size"]
        t = int(math.ceil(original_t / downsample))

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        
        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        m, u = merge.bipartite_soft_matching_random1d(x, t, w, h, args["sx"], ratio, num_token_groups=1,
                                                      no_rand=not use_rand, generator=args["generator"])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)

    m_a, u_a = (m, u) if args["merge_attn"]      else (merge.do_nothing, merge.do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (merge.do_nothing, merge.do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"]       else (merge.do_nothing, merge.do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good

# Function to find all common divisors using prime factors
def common_divisors(num1, num2):
    # Find gcd of the numbers
    gcd_num = gcd(num1, num2)
    
    # Find prime factors of the gcd
    pf = primefactors(gcd_num)
    
    # Generate all divisors from prime factors of gcd
    divisors = set()
    def generate_divisors(current_divisor, prime_factors):
        if not prime_factors:
            if num1 % current_divisor == 0 and num2 % current_divisor == 0:
                divisors.add(current_divisor)
            return
        prime = prime_factors[0]
        remaining_factors = prime_factors[1:]
        exponent = 0
        while current_divisor <= gcd_num:
            generate_divisors(current_divisor, remaining_factors)
            exponent += 1
            current_divisor *= prime

    generate_divisors(1, pf)
    return sorted(divisors)


def check_greater_than_zero(x: Union[float, list[float]]) -> bool:
    if isinstance(x, list):
        return any(value > 0 for value in x)
    else:
        return x > 0
    
    
def compute_spatial_merge(x: torch.Tensor, t: int, h: int, w: int, tome_info: Dict[str, Any], fm_ratio: Union[float, list[float]], tm_ratio: Union[float, list[float]]) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // (h * w))))

    args = tome_info["args"]

    if check_greater_than_zero(fm_ratio) and check_greater_than_zero(tm_ratio) and downsample <= args["max_downsample"]:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        
        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        possible_token_groups = common_divisors(h, w)
        
        num_token_groups = 8 // downsample
        # == Check if the number of token groups is possible ==
        
        if num_token_groups not in possible_token_groups:
            # == Find the closest possible number of token groups ==
            old_token_groups = num_token_groups
            num_token_groups = min(possible_token_groups, key=lambda x:abs(x-old_token_groups))
            warnings.warn(f"Number of token groups {old_token_groups} is not possible. Using {num_token_groups} instead.")
        
        m, u = merge.bipartite_soft_matching_random3d_spatial(x, t, w, h, args["st"], args["sx"], args["sy"], fm_ratio, tm_ratio,
                                                      num_token_groups=num_token_groups, no_rand=not use_rand, 
                                                      generator=args["generator"])
    elif check_greater_than_zero(fm_ratio) > 0 and downsample <= args["max_downsample"]:
        fm_ratio = fm_ratio[0] if isinstance(fm_ratio, list) else fm_ratio

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        
        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        
        possible_token_groups = common_divisors(h, w)
        
        num_token_groups = 8 // downsample
        # == Check if the number of token groups is possible ==
        
        if num_token_groups not in possible_token_groups:
            # == Find the closest possible number of token groups ==
            old_token_groups = num_token_groups
            num_token_groups = min(possible_token_groups, key=lambda x:abs(x-old_token_groups))
            warnings.warn(f"Number of token groups {old_token_groups} is not possible. Using {num_token_groups} instead.")

        m, u = merge.bipartite_soft_matching_random1d(x, t, w, h, args["st"], fm_ratio, num_token_groups=4,
                                                      no_rand=not use_rand, generator=args["generator"])
    elif check_greater_than_zero(tm_ratio) > 0 and downsample <= args["max_downsample"]:
        tm_ratio = tm_ratio[0] if isinstance(tm_ratio, list) else tm_ratio
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        x = x.reshape(x.shape[0] * x.shape[1], h * w, -1)

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        
        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        
        possible_token_groups = common_divisors(h, w)
        
        num_token_groups = 8 // downsample
        # == Check if the number of token groups is possible ==
        
        if num_token_groups not in possible_token_groups:
            # == Find the closest possible number of token groups ==
            old_token_groups = num_token_groups
            num_token_groups = min(possible_token_groups, key=lambda x:abs(x-old_token_groups))
            warnings.warn(f"Number of token groups {old_token_groups} is not possible. Using {num_token_groups} instead.")
        
        m, u = merge.bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], tm_ratio, num_token_groups=num_token_groups, 
                                                      no_rand=not use_rand, generator=args["generator"])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)

    m_a, u_a = (m, u) if args["merge_attn"]      else (merge.do_nothing, merge.do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (merge.do_nothing, merge.do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"]       else (merge.do_nothing, merge.do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good


def compute_temporal_merge(x: torch.Tensor, t: int, h: int, w: int, tome_info: Dict[str, Any], bm_ratio: Union[float, list[float]], tm_ratio: Union[float, list[float]]) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // (h * w))))

    args = tome_info["args"]

    if check_greater_than_zero(bm_ratio) and check_greater_than_zero(tm_ratio) and downsample <= args["max_downsample"]:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        
        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        num_token_groups = 8 // downsample
        m, u = merge.bipartite_soft_matching_random3d_temporal(x, t, w, h, args["st"], args["sx"], args["sy"], bm_ratio, tm_ratio,
                                                      num_token_groups=4, no_rand=not use_rand, 
                                                      generator=args["generator"])
    elif check_greater_than_zero(bm_ratio) > 0 and downsample <= args["max_downsample"]:
        bm_ratio = bm_ratio[0] if isinstance(bm_ratio, list) else bm_ratio
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        
        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        num_token_groups = 8 // downsample
        m, u = merge.bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], bm_ratio, num_token_groups=4,
                                                      no_rand=not use_rand, generator=args["generator"])
    elif check_greater_than_zero(tm_ratio) > 0 and downsample <= args["max_downsample"]:
        tm_ratio = tm_ratio[0] if isinstance(tm_ratio, list) else tm_ratio
        x = x.reshape(x.shape[0] * x.shape[1], t, -1)

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        
        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        m, u = merge.bipartite_soft_matching_random1d(x, t, w, h, args["st"], tm_ratio, num_token_groups=1,
                                                      no_rand=not use_rand, generator=args["generator"])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)

    m_a, u_a = (m, u) if args["merge_attn"]      else (merge.do_nothing, merge.do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (merge.do_nothing, merge.do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"]       else (merge.do_nothing, merge.do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good


def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(x, self._tome_info)

            # This is where the meat of the computation happens
            x = u_a(self.attn1(m_a(self.norm1(x)), context=context if self.disable_self_attn else None)) + x
            x = u_c(self.attn2(m_c(self.norm2(x)), context=context)) + x
            x = u_m(self.ff(m_m(self.norm3(x)))) + x

            return x
    
    return ToMeBlock


def make_tome_temp_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(x, self._tome_info)

            # This is where the meat of the computation happens
            x = u_a(self.attn1(m_a(self.norm1(x)), context=context if self.disable_self_attn else None)) + x
            x = u_c(self.attn2(m_c(self.norm2(x)), context=context)) + x
            x = u_m(self.ff(m_m(self.norm3(x)))) + x

            return x
    
    return ToMeBlock


def get_frame_merge_func(hidden_states, num_frames, height, width, tome_info, enabled=True):
    if enabled:
        batch_frames, seq_length, channels = hidden_states.shape
        batch_size = batch_frames // num_frames
        hidden_states = hidden_states.reshape(batch_frames, height, width, channels).permute(0, 3, 1, 2)
        # TODO: downsample spatially to save time, should not be hardcoded
        height_down = 4
        width_down = 4
        hidden_states = F.interpolate(hidden_states, size=[height_down, width_down], mode="area")
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, num_frames, height_down * width_down * channels)
        # hidden_states = hidden_states.reshape(batch_size, num_frames, seq_length * channels)
        fm_a, fm_c, fm_m, fu_a, fu_c, fu_m = compute_merge_temp(hidden_states, tome_info, tome_info["args"]["fm_ratio"])
        # hidden_states = hidden_states.reshape(batch_frames, seq_length, channels)
        return fm_a, fm_c, fm_m, fu_a, fu_c, fu_m
    else:
        return merge.do_nothing, merge.do_nothing, merge.do_nothing, merge.do_nothing, merge.do_nothing, merge.do_nothing


def get_token_merge_func(hidden_states, tome_info, enabled=True):
    if enabled:
        tm_a, tm_c, tm_m, tu_a, tu_c, tu_m = compute_merge(hidden_states, tome_info, tome_info["args"]["tm_ratio"])
        return tm_a, tm_c, tm_m, tu_a, tu_c, tu_m
    else:
        return merge.do_nothing, merge.do_nothing, merge.do_nothing, merge.do_nothing, merge.do_nothing, merge.do_nothing
    

def get_block_merge_func(hidden_states, seq_length, tome_info, enabled=True):
    if enabled:
        # batch_frames, num_frames, channels = hidden_states.shape
        # batch_size = batch_frames // seq_length
        # hidden_states = hidden_states.permute(0, 2, 1)
        # # TODO: downsample spatially to save time, should not be hardcoded
        # num_frames_down = 4
        # hidden_states = F.interpolate(hidden_states, size=num_frames_down, mode="area")
        # hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, seq_length, num_frames_down * channels)
        bm_a, bm_c, bm_m, bu_a, bu_c, bu_m = compute_merge(hidden_states, tome_info, tome_info["args"]["bm_ratio"])
        return bm_a, bm_c, bm_m, bu_a, bu_c, bu_m
    else:
        return merge.do_nothing, merge.do_nothing, merge.do_nothing, merge.do_nothing, merge.do_nothing, merge.do_nothing
    

def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            x,
            y,
            t,
            mask=None,  # text mask
            x_mask=None,  # temporal mask
            t0=None,  # t with timestamp=0
            T=None,  # number of frames
            S=None,  # number of pixel patches
            height=None, # height of pixel patch
            width=None, # width of pixel patch
        ) -> torch.Tensor:
            
            # print(" == In Omnimerge function (Spatial merge) == ")
            # print(f"Passed height {height} and width {width}")
            # print(f"Number of frames {T}")
            
            # prepare modulate parameters
            B, N, C = x.shape
            
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
            if x_mask is not None:
                shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                    self.scale_shift_table[None] + t0.reshape(B, 6, -1)
                ).chunk(6, dim=1)

            # modulate (attention)
            x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
            if x_mask is not None:
                x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero, scale_msa_zero)
                x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)
            
            orig_x_m = x_m.clone()
            x_m = rearrange(x_m, "B (T S) C -> B T (S C)", T=T, S=S)
            fm_a, fm_c, fm_m, fu_a, fu_c, fu_m = compute_spatial_merge(x_m, T, height, 
                                                                        width, self._tome_info, 
                                                                        self._tome_info["args"]["fm_ratio"],
                                                                        self._tome_info["args"]["tm_ratio"])
            x_m = orig_x_m
            # == Merge (attention) ==
            x_m = fm_a(x_m)
            # attention
            x_m = self.attn(x_m)

            # == Unmerge (attention) ==
            x_m = fu_a(x_m) 
            x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)
    
            # modulate (attention)
            x_m_s = gate_msa * x_m
            if x_mask is not None:
                x_m_s_zero = gate_msa_zero * x_m
                x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

            # residual
            x = x + self.drop_path(x_m_s)

            # cross attention
            x = x + self.cross_attn(x, y, mask)

            # modulate (MLP)
            x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
            if x_mask is not None:
                x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero)
                x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

            # MLP
            x_m = self.mlp(x_m)

            # modulate (MLP)
            x_m_s = gate_mlp * x_m
            if x_mask is not None:
                x_m_s_zero = gate_mlp_zero * x_m
                x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

            # residual
            x = x + self.drop_path(x_m_s)
            
            return x

    return ToMeBlock


def make_diffusers_tome_temp_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeTempBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            seq_length: int = None,
            height: int = None,
            width: int = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
        ) -> torch.FloatTensor:
            # Notice that normalization is always applied before the real computation in the following blocks.
            # 0. Self-Attention
            batch_frames, num_frames, channels = hidden_states.shape
            batch_size = batch_frames // seq_length

            # Block merge
            # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # starter.record()
            hidden_states = hidden_states.reshape(batch_size, seq_length, num_frames * channels)
            # bm_a, bm_c, bm_m, bu_a, bu_c, bu_m = get_block_merge_func(hidden_states, seq_length, self._tome_info)
            bm_a, bm_c, bm_m, bu_a, bu_c, bu_m = compute_temporal_merge(hidden_states, num_frames, 
                                                                        height, width, self._tome_info, 
                                                                        self._tome_info["args"]["bm_ratio"],
                                                                        self._tome_info["args"]["tm_ratio"])
            hidden_states = hidden_states.reshape(batch_frames, num_frames, channels)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"get_block_merge_func runtime: {curr_time}ms")

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.use_layer_norm:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.use_ada_layer_norm_single:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_frames, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                norm_hidden_states = norm_hidden_states.squeeze(1)
            else:
                raise ValueError("Incorrect norm used")

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            # 1. Retrieve lora scale.
            lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

            # 2. Prepare GLIGEN inputs
            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

            # Block merge for self attention
            # starter.record()
            # norm_hidden_states = norm_hidden_states.reshape(batch_size, seq_length, num_frames * channels)
            # bm_a, _, _, bu_a, _, _ = get_block_merge_func(norm_hidden_states, self._tome_info, enabled=self._tome_info["args"]["merge_attn"])
            norm_hidden_states = bm_a(norm_hidden_states)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"bm_a runtime: {curr_time}ms")
            # seq_length_down = norm_hidden_states.shape[1]
            # norm_hidden_states = norm_hidden_states.reshape(batch_size * seq_length_down, num_frames, channels)
            
            # starter.record()
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"attn1 runtime: {curr_time}ms")

            # Block unmerge for self attention
            # starter.record()
            # attn_output = attn_output.reshape(batch_size, seq_length_down, num_frames * channels)
            attn_output = bu_a(attn_output)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"bu_a runtime: {curr_time}ms")

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.use_ada_layer_norm_single:
                attn_output = gate_msa * attn_output

            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            # 2.5 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

            # # 3. Cross-Attention
            # if self.attn2 is not None:
            #     if self.use_ada_layer_norm:
            #         norm_hidden_states = self.norm2(hidden_states, timestep)
            #     elif self.use_ada_layer_norm_zero or self.use_layer_norm:
            #         norm_hidden_states = self.norm2(hidden_states)
            #     elif self.use_ada_layer_norm_single:
            #         # For PixArt norm2 isn't applied here:
            #         # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
            #         norm_hidden_states = hidden_states
            #     else:
            #         raise ValueError("Incorrect norm")

            #     if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
            #         norm_hidden_states = self.pos_embed(norm_hidden_states)

            #     attn_output = self.attn2(
            #         norm_hidden_states,
            #         encoder_hidden_states=encoder_hidden_states,
            #         attention_mask=encoder_attention_mask,
            #         **cross_attention_kwargs,
            #     )
            #     hidden_states = attn_output + hidden_states

            # 4. Feed-forward
            # if not self.use_ada_layer_norm_single:
            #     norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            if self.use_ada_layer_norm_single:
                # norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = self.norm3(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            # Block merge for mlp
            # starter.record()
            # norm_hidden_states = norm_hidden_states.reshape(batch_size, seq_length, num_frames * channels)
            # _, _, bm_m, _, _, bu_m = get_block_merge_func(norm_hidden_states, self._tome_info, enabled=self._tome_info["args"]["merge_mlp"])
            norm_hidden_states = bm_m(norm_hidden_states)
            # seq_length_down = norm_hidden_states.shape[1]
            # norm_hidden_states = norm_hidden_states.reshape(batch_size * seq_length_down, num_frames, channels)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"bm_m runtime: {curr_time}ms")
            
            # starter.record()
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                    raise ValueError(
                        f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                    )

                num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
                ff_output = torch.cat(
                    [
                        self.ff(hid_slice, scale=lora_scale)
                        for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)
                    ],
                    dim=self._chunk_dim,
                )
            else:
                ff_output = self.ff(norm_hidden_states, scale=lora_scale)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"ff runtime: {curr_time}ms")

            # Block unmerge for mlp
            # starter.record()
            # ff_output = ff_output.reshape(batch_size, seq_length_down, num_frames * channels)
            ff_output = bu_m(ff_output)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"bu_m runtime: {curr_time}ms")

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.use_ada_layer_norm_single:
                ff_output = gate_mlp * ff_output

            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            return hidden_states

    return ToMeTempBlock


def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[-2] // model.patch_size, args[0].shape[-1] // model.patch_size)
        module._tome_info["num_frames"] = args[0].shape[2]
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def apply_patch(
        model: torch.nn.Module,
        fm_ratio: float = 0.5,
        tm_ratio: float = 0.5,
        bm_ratio: float = 0.5,
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2, st: int = 2,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = True,
        merge_mlp: bool = True):
    """
    Patches a stable diffusion model with ToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.
    
    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply ToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Doesn't have to divide image size.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_crossattn: Whether or not to merge tokens for cross attention (not recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).
    """

    # Make sure the module is not currently patched
    remove_patch(model)

    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")
    
    # == Diffusion Model == #
    diffusion_model = model

    # if not is_diffusers:
    #     if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
    #         # Provided model not supported
    #         raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
    #     diffusion_model = model.model.diffusion_model
    # else:
    #     # Supports "pipe.unet" and "unet"
    #     diffusion_model = model.unet if hasattr(model, "unet") else model


    diffusion_model._tome_info = {
        "size": (30, 53),
        "num_frames": 15,
        "hooks": [],
        "args": {
            "fm_ratio": fm_ratio,  # frame merge ratio
            "tm_ratio": tm_ratio,  # token merge ratio
            "bm_ratio": bm_ratio,  # block merge ratio
            "max_downsample": max_downsample,
            "sx": sx, "sy": sy, "st": st,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp
        }
    }
    # hook_tome_model(diffusion_model)

    include_names_gp1 = [
        # "spatial_blocks.0",
        # "spatial_blocks.1",
        # "spatial_blocks.2",
        # "spatial_blocks.3",
        # "spatial_blocks.4",
        # "spatial_blocks.5",
        # "spatial_blocks.6",
        # "spatial_blocks.7",
        "spatial_blocks.8",
        "spatial_blocks.9",
        "spatial_blocks.10",
        "spatial_blocks.11",
        "spatial_blocks.12",
        "spatial_blocks.13",
        "spatial_blocks.14",
        "spatial_blocks.15",
        "spatial_blocks.16",
        "spatial_blocks.17",
        "spatial_blocks.18",
        "spatial_blocks.19",
        # "spatial_blocks.20",
        # "spatial_blocks.21",
        # "spatial_blocks.22",
        # "spatial_blocks.23",
        # "spatial_blocks.24",
        # "spatial_blocks.25",
        # "spatial_blocks.26",
        # "spatial_blocks.27",
    ]

    include_names_gp2 = [
        # "spatial_blocks.0",
        # "spatial_blocks.1",
        # "spatial_blocks.2",
        "spatial_blocks.3",
        "spatial_blocks.4",
        "spatial_blocks.5",
        "spatial_blocks.6",
        "spatial_blocks.7",
        "spatial_blocks.20",
        "spatial_blocks.21",
        "spatial_blocks.22",
        "spatial_blocks.23",
        "spatial_blocks.24",
        "spatial_blocks.25",
        # "spatial_blocks.26",
        # "spatial_blocks.27",
    ]

    include_names_gp3 = [
        # "temporal_blocks.0",
        # "temporal_blocks.1",
        # "temporal_blocks.2",
        # "temporal_blocks.3",
        # "temporal_blocks.4",
        # "temporal_blocks.5",
        # "temporal_blocks.6",
        # "temporal_blocks.7",
        "temporal_blocks.8",
        "temporal_blocks.9",
        "temporal_blocks.10",
        "temporal_blocks.11",
        "temporal_blocks.12",
        "temporal_blocks.13",
        "temporal_blocks.14",
        "temporal_blocks.15",
        "temporal_blocks.16",
        "temporal_blocks.17",
        "temporal_blocks.18",
        "temporal_blocks.19",
        # "temporal_blocks.20",
        # "temporal_blocks.21",
        # "temporal_blocks.22",
        # "temporal_blocks.23",
        # "temporal_blocks.24",
        # "temporal_blocks.25",
        # "temporal_blocks.26",
        # "temporal_blocks.27",
    ]

    include_names_gp4 = [
        # "temporal_blocks.0",
        # "temporal_blocks.1",
        # "temporal_blocks.2",
        # "temporal_blocks.3",
        # "temporal_blocks.4",
        # "temporal_blocks.5",
        "temporal_blocks.6",
        "temporal_blocks.7",
        "temporal_blocks.20",
        "temporal_blocks.21",
        # "temporal_blocks.22",
        # "temporal_blocks.23",
        # "temporal_blocks.24",
        # "temporal_blocks.25",
        # "temporal_blocks.26",
        # "temporal_blocks.27",
    ]

    
    for name, module in diffusion_model.named_modules():
        # zyhe: change the name to select specific transfomer blocks
        if isinstance_str(module, "STDiT3Block"):
            if any([include_name == name for include_name in list(include_names_gp1)]):
                make_tome_block_fn = make_diffusers_tome_block
                module.__class__ = make_tome_block_fn(module.__class__)
                module._tome_info = copy.deepcopy(diffusion_model._tome_info)
                print(f"Patching {name} with fm_ratio: {module._tome_info['args']['fm_ratio']}, tm_ratio: {module._tome_info['args']['tm_ratio']}")
                # print(">"*50)

                # # Something introduced in SD 2.0 (LDM only)
                # if not hasattr(module, "disable_self_attn") and not is_diffusers:
                #     module.disable_self_attn = False

                # # Something needed for older versions of diffusers
                # if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                #     module.use_ada_layer_norm = False
                #     module.use_ada_layer_norm_zero = False
            # elif any([include_name == name for include_name in include_names_gp2]):
            #     make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
            #     module.__class__ = make_tome_block_fn(module.__class__)
            #     module._tome_info = copy.deepcopy(diffusion_model._tome_info)
            #     module._tome_info["args"]["fm_ratio"] = 0.2
            #     module._tome_info["args"]["tm_ratio"] = 0
            #     module._tome_info["args"]["merge_mlp"] = False

            #     # Something introduced in SD 2.0 (LDM only)
            #     if not hasattr(module, "disable_self_attn") and not is_diffusers:
            #         module.disable_self_attn = False

            #     # Something needed for older versions of diffusers
            #     if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
            #         module.use_ada_layer_norm = False
            #         module.use_ada_layer_norm_zero = False
        # if isinstance_str(module, "BasicTransformerBlock_"):
        #     if any([include_name == name for include_name in include_names_gp3]):
        #         make_tome_block_fn = make_diffusers_tome_temp_block if is_diffusers else make_tome_temp_block
        #         module.__class__ = make_tome_block_fn(module.__class__)
        #         module._tome_info = diffusion_model._tome_info

        #         # Something introduced in SD 2.0 (LDM only)
        #         if not hasattr(module, "disable_self_attn") and not is_diffusers:
        #             module.disable_self_attn = False

        #         # Something needed for older versions of diffusers
        #         if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
        #             module.use_ada_layer_norm = False
        #             module.use_ada_layer_norm_zero = False
        #     elif any([include_name == name for include_name in include_names_gp4]):
        #         make_tome_block_fn = make_diffusers_tome_temp_block if is_diffusers else make_tome_temp_block
        #         module.__class__ = make_tome_block_fn(module.__class__)
        #         module._tome_info = copy.deepcopy(diffusion_model._tome_info)
        #         module._tome_info["args"]["bm_ratio"] = 0

        #         # Something introduced in SD 2.0 (LDM only)
        #         if not hasattr(module, "disable_self_attn") and not is_diffusers:
        #             module.disable_self_attn = False

        #         # Something needed for older versions of diffusers
        #         if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
        #             module.use_ada_layer_norm = False
        #             module.use_ada_layer_norm_zero = False
    return model


def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent
    
    return model
    return model
    return model

def print_profiler_results(model: torch.nn.Module, save_to_file: bool = False, filename: str = "profiler_results"):
    table = Table(title="Profiler Results")
    table.add_column("Block Name", style="cyan")
    operations = ["temporal_attn", "spatial_attn", "cross_attn", "mlp"]
    for opr in operations:
        table.add_column(f"{opr} (ms)", style="green")

    results = []
    for name, module in model.named_modules():
        if hasattr(module, '_profiler_outputs'):
            row = [name]
            for opr in operations:
                cuda_time = module._profiler_outputs.get(opr, {}).get('cuda_time', 0)
                row.append(f"{cuda_time:.2f}")
            results.append(row)
            table.add_row(*row)

    console = Console()
    console.print(table)

    if save_to_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # == Check if the filename contains directory path ==
        filename = filename.split("/")
        profiler_dir = "/".join(filename[:-1])
        filename = filename[-1]
        
        # == Strip .csv from filename if it exists ==
        filename = filename.replace(".csv", "")
        
        # == If profiler_dir above is not passed with the path == #
        if profiler_dir == "":
            profiler_dir = "profiler_results"
        os.makedirs(profiler_dir, exist_ok=True)
        filename = os.path.join(profiler_dir, filename+f"_{timestamp}.csv")
        
        with open(filename, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Block Name"] + [f"{opr} (ms)" for opr in operations])
            writer.writerows(results)
        
        print(f"Profiler results saved to {filename}")