from typing import Callable, Tuple, Union

import torch


def do_nothing(x: torch.Tensor, mode:str=None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_random1d(metric: torch.Tensor,
                                     t: int, w: int, h: int, sx: int, r: float, num_token_groups: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - t: number of frames in tokens
     - sx: stride in the t dimension for dst, must divide t
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape
    T = t
    H = h
    W = w
    C = metric.shape[-1] // (h * w)
    GT = num_token_groups

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        # Hierarchical merging
        if GT > 1:
            metric = metric.reshape(B, T, GT, H // GT, GT, W // GT, -1).permute(0, 2, 4, 1, 3, 5, 6)
            metric = metric.reshape(B * GT * GT, T, -1)
            h = H // GT
            w = W // GT
        
        wsx = t // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sx, size=(wsx, 1), device=generator.device, generator=generator).to(metric.device)
        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(wsx, sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=1, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(wsx, sx).reshape(wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (wsx * sx) < t:
            idx_buffer = torch.zeros(t, device=metric.device, dtype=torch.int64)
            idx_buffer[:(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = wsx
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst

        def split(x):
            b, n, c = x.shape
            src = gather(x, dim=1, index=a_idx.expand(b, n - num_dst, c))
            dst = gather(x, dim=1, index=b_idx.expand(b, num_dst, c))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = int(metric.shape[1] * r)
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        reshape = False
        if x.shape[0] != B:
            reshape = True
            _, seq_len, channel = x.shape
            x = x.reshape(B, -1, seq_len * channel)
        if GT > 1:
            x = x.reshape(B, T, GT, H // GT, GT, W // GT, -1).permute(0, 2, 4, 1, 3, 5, 6)
            x = x.reshape(B * GT * GT, T, -1)
        src, dst = split(x)
        n, t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        out = torch.cat([unm, dst], dim=1)
        if GT > 1:
            out = out.reshape(B, GT, GT, out.shape[1], H // GT, W // GT, -1).permute(0, 3, 1, 4, 2, 5, 6)
            out = out.reshape(B, out.shape[1], -1)
        if reshape:
            out = out.reshape(-1, seq_len, channel)
            
        return out

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        reshape = False
        if x.shape[0] != B:
            reshape = True
            _, seq_len, channel = x.shape
            x = x.reshape(B, -1, seq_len * channel)
        if GT > 1:
            x = x.reshape(B, x.shape[1], GT, H // GT, GT, W // GT, -1).permute(0, 2, 4, 1, 3, 5, 6) 
            x = x.reshape(B * GT * GT, x.shape[3], -1)
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        b, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(b, r, c))

        # Combine back to the original shape
        out = torch.zeros(b, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(b, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(b, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, a_idx.shape[1], 1), dim=1, index=src_idx).expand(b, r, c), src=src)
        if GT > 1:
            out = out.reshape(B, GT, GT, T, H // GT, W // GT, -1).permute(0, 3, 1, 4, 2, 5, 6)
            out = out.reshape(B, T, -1)
        if reshape:
            out = out.reshape(-1, seq_len, channel)

        return out

    return merge, unmerge

def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: float, num_token_groups: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape
    H = h
    W = w
    C = metric.shape[-1]
    GT = num_token_groups

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        # Hierarchical merging
        metric = metric.reshape(B, GT, H // GT, GT, W // GT, -1).permute(0, 1, 3, 2, 4, 5)
        metric = metric.reshape(B * GT * GT, (H // GT) * (W // GT), -1)
        h = H // GT
        w = W // GT
        
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)
        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst

        def split(x):
            b, n, c = x.shape
            src = gather(x, dim=1, index=a_idx.expand(b, n - num_dst, c))
            dst = gather(x, dim=1, index=b_idx.expand(b, num_dst, c))
            return src, dst
        
        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = int(metric.shape[1] * r)
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        reshape = False
        if x.shape[0] != B:
            reshape = True
            _, frame_len, channel = x.shape
            x = x.reshape(B, -1, frame_len * channel)
        x = x.reshape(B, GT, H // GT, GT, W // GT, -1).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B * GT * GT, (H // GT) * (W // GT), -1)
        src, dst = split(x)
        n, t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        if mode is not None:
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        out = torch.cat([unm, dst], dim=1)
        out = out.reshape(B, -1, out.shape[-1])
        if reshape:
            out = out.reshape(-1, frame_len, channel)
            
        return out

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        reshape = False
        if x.shape[0] != B:
            reshape = True
            _, seq_len, channel = x.shape
            x = x.reshape(B, -1, seq_len * channel)
        x = x.reshape(B * GT * GT, x.shape[1] // (GT * GT), -1) 
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        b, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(b, r, c))

        # Combine back to the original shape
        out = torch.zeros(b, (H // GT) * (W // GT), c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(b, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(b, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, a_idx.shape[1], 1), dim=1, index=src_idx).expand(b, r, c), src=src)
        out = out.reshape(B, GT, GT, H // GT, W // GT, -1).permute(0, 1, 3, 2, 4, 5)
        out = out.reshape(B, H*W, -1)
        if reshape:
            out = out.reshape(-1, seq_len, channel)
            
        return out

    return merge, unmerge


def random1d(t, st, metric, generator, no_rand=False):
    tst = t // st

    # For each sy by sx kernel, randomly assign one token to be dst and the rest src
    if no_rand:
        rand_idx = torch.zeros(tst, 1, device=metric.device, dtype=torch.int64)
    else:
        rand_idx = torch.randint(st, size=(tst, 1), device=generator.device, generator=generator).to(metric.device)
    
    # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
    idx_buffer_view = torch.zeros(tst, st, device=metric.device, dtype=torch.int64)
    idx_buffer_view.scatter_(dim=1, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
    idx_buffer_view = idx_buffer_view.view(tst, st).reshape(tst * st)

    # Image is not divisible by sx or sy so we need to move it into a new buffer
    if (tst * st) < t:
        idx_buffer = torch.zeros(t, device=metric.device, dtype=torch.int64)
        idx_buffer[:(tst * st)] = idx_buffer_view
    else:
        idx_buffer = idx_buffer_view

    # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
    rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

    # We're finished with these
    del idx_buffer, idx_buffer_view

    # rand_idx is currently dst|src, so split them
    num_dst = tst
    a_idx = rand_idx[:, num_dst:, :] # src
    b_idx = rand_idx[:, :num_dst, :] # dst

    return num_dst, a_idx, b_idx


def random2d(h, sy, w, sx, metric, generator, no_rand=False):
    hsy, wsx = h // sy, w // sx
    
    
    num_dst_tokens = 2

    # For each sy by sx kernel, randomly assign one token to be dst and the rest src
    if no_rand:
        rand_idx = torch.zeros(hsy, wsx, num_dst_tokens, device=metric.device, dtype=torch.int64)
    else:
        rand_idx = torch.randint(sy*sx, size=(hsy, wsx, num_dst_tokens), device=generator.device, generator=generator).to(metric.device)
    
    # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
    idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
    idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
    idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

    # Image is not divisible by sx or sy so we need to move it into a new buffer
    if (hsy * sy) < h or (wsx * sx) < w:
        idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
        idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
    else:
        idx_buffer = idx_buffer_view

    # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
    rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

    # We're finished with these
    del idx_buffer, idx_buffer_view

    # rand_idx is currently dst|src, so split them
    num_dst = hsy * wsx * num_dst_tokens
    a_idx = rand_idx[:, num_dst:, :] # src
    b_idx = rand_idx[:, :num_dst, :] # dst

    return num_dst, a_idx, b_idx


def random3d(t, st, h, sy, w, sx, metric, generator, no_rand=False):
    tst, hsy, wsx = t // st, h // sy, w // sx

    # For each sy by sx kernel, randomly assign one token to be dst and the rest src
    if no_rand:
        rand_idx = torch.zeros(tst, hsy, wsx, 1, device=metric.device, dtype=torch.int64)
    else:
        rand_idx = torch.randint(st*sy*sx, size=(tst, hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)
    
    # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
    idx_buffer_view = torch.zeros(tst, hsy, wsx, st*sy*sx, device=metric.device, dtype=torch.int64)
    idx_buffer_view.scatter_(dim=3, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
    idx_buffer_view = idx_buffer_view.view(tst, hsy, wsx, st, sy, sx).permute(0, 3, 1, 4, 2, 5).reshape(tst * st, hsy * sy, wsx * sx)

    # Image is not divisible by sx or sy so we need to move it into a new buffer
    if (hsy * sy) < h or (wsx * sx) < w or (tst * st) < t:
        idx_buffer = torch.zeros(t, h, w, device=metric.device, dtype=torch.int64)
        idx_buffer[:(tst * st), :(hsy * sy), :(wsx * sx)] = idx_buffer_view
    else:
        idx_buffer = idx_buffer_view

    # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
    rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

    # We're finished with these
    del idx_buffer, idx_buffer_view

    # rand_idx is currently dst|src, so split them
    num_dst = tst * hsy * wsx
    a_idx = rand_idx[:, num_dst:, :] # src
    b_idx = rand_idx[:, :num_dst, :] # dst

    return num_dst, a_idx, b_idx


def random2d_diagnal(t, st, s, sx, metric, generator, tm_ratio=None, no_rand=False):
    tst, ssx = t // st, s // sx

    # For each sy by sx kernel, randomly assign one token to be dst and the rest src
    if no_rand:
        rand_idx = torch.zeros(tst, ssx, 1, device=metric.device, dtype=torch.int64)
    else:
        mapping = torch.arange(sx - 1, -1, -1).to(metric.device)
        rand_idx1 = torch.randint(sx, size=(ssx, 1), device=generator.device, generator=generator).to(metric.device)
        rand_idx2 = mapping[rand_idx1]
        rand_idx = torch.stack([rand_idx1, rand_idx2], dim=0)

    # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
    idx_buffer_view = torch.zeros(tst, ssx, st*sx, device=metric.device, dtype=torch.int64)
    idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
    idx_buffer_view = idx_buffer_view.view(tst, ssx, st, sx).permute(0, 2, 1, 3).reshape(tst * st, ssx * sx)

    # Image is not divisible by sx or sy so we need to move it into a new buffer
    if (ssx * sx) < s or (tst * st) < t:
        idx_buffer = torch.zeros(t, s, device=metric.device, dtype=torch.int64)
        idx_buffer[:(tst * st), :(ssx * sx)] = idx_buffer_view
    else:
        idx_buffer = idx_buffer_view

    # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
    rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

    # We're finished with these
    del idx_buffer, idx_buffer_view

    # rand_idx is currently dst|src, so split them
    num_dst = tst * ssx
    # if tm_ratio is not None:
    #     # num_dst = (tst * hsy * wsx * 2) // 3
    #     num_dst = int(tst * st * ssx * sx * (1 - tm_ratio))
    #     indices = torch.randperm(tst * ssx).to(metric.device)
    #     indices1 = indices[num_dst:].unsqueeze(0).unsqueeze(-1)
    #     indices2 = indices[:num_dst].unsqueeze(0).unsqueeze(-1)
    #     a_idx = torch.cat([torch.gather(rand_idx, 1, indices1), rand_idx[:, tst * ssx:, :]], dim=1) # src
    #     b_idx = torch.gather(rand_idx, 1, indices2) # dst
    # else:
    a_idx = rand_idx[:, num_dst:, :] # src
    b_idx = rand_idx[:, :num_dst, :] # dst

    return num_dst, a_idx, b_idx


def random3d_diagnal(t, st, h, sy, w, sx, metric, generator, tm_ratio, no_rand=False):
    tst, hsy, wsx = t // st, h // sy, w // sx
    
    # Number of tokens we want to be destination tokens
    num_dst_tokens = 1

    # For each sy by sx kernel, randomly assign one token to be dst and the rest src
    if no_rand:
        rand_idx = torch.zeros(tst, hsy, wsx, num_dst_tokens, device=metric.device, dtype=torch.int64)
    else:
        mapping = torch.tensor([3, 2, 1, 0]).to(metric.device)
        rand_idx1 = torch.randint(sy*sx, size=(hsy, wsx, num_dst_tokens), device=generator.device, generator=generator).to(metric.device)
        rand_idx2 = mapping[rand_idx1]
        rand_idx = torch.stack([rand_idx1, rand_idx2], dim=0)

    # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
    idx_buffer_view = torch.zeros(tst, hsy, wsx, st*sy*sx, device=metric.device, dtype=torch.int64)
    idx_buffer_view.scatter_(dim=3, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
    idx_buffer_view = idx_buffer_view.view(tst, hsy, wsx, st, sy, sx).permute(0, 3, 1, 4, 2, 5).reshape(tst * st, hsy * sy, wsx * sx)
    
    # Image is not divisible by sx or sy so we need to move it into a new buffer
    if (hsy * sy) < h or (wsx * sx) < w or (tst * st) < t:
        idx_buffer = torch.zeros(t, h, w, device=metric.device, dtype=torch.int64)
        idx_buffer[:(tst * st), :(hsy * sy), :(wsx * sx)] = idx_buffer_view
    else:
        idx_buffer = idx_buffer_view

    # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
    rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

    # We're finished with these
    del idx_buffer, idx_buffer_view

    # rand_idx is currently dst|src, so split them
    num_dst = int(tst * hsy * wsx * num_dst_tokens)
    # num_dst += ((2*num_dst)//3)
    # print(f"Number of dst tokens {num_dst}")
    # num_dst = (tst * hsy * wsx * 2) // 3
    # num_dst = int(hsy * sy * wsx * sx * (1 - tm_ratio))
    # indices = torch.randperm(tst * hsy * wsx).to(metric.device)
    # indices1 = indices[num_dst:].unsqueeze(0).unsqueeze(-1)
    # indices2 = indices[:num_dst].unsqueeze(0).unsqueeze(-1)
    # a_idx = torch.cat([torch.gather(rand_idx, 1, indices1), rand_idx[:, tst * hsy * wsx:, :]], dim=1) # src
    # b_idx = torch.gather(rand_idx, 1, indices2) # dst
    a_idx = rand_idx[:, num_dst:, :] # src
    b_idx = rand_idx[:, :num_dst, :] # dst

    return num_dst, a_idx, b_idx


def bipartite_soft_matching_random3d_spatial(metric: torch.Tensor,
                                     t: int, w: int, h: int, st: int, sx: int, sy: int, 
                                     fm_ratio: Union[float, list[float]], tm_ratio: Union[float, list[float]],
                                     num_frame_groups: int = 1, num_token_groups: int = 8,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B = metric.shape[0]
    T = t
    H = h
    W = w
    C = metric.shape[-1] // (h * w)

    if not isinstance(fm_ratio, list):
        fm_ratio = [fm_ratio]
    if not isinstance(tm_ratio, list):
        tm_ratio = [tm_ratio]
    assert len(fm_ratio) == len(tm_ratio)
    
    L = len(fm_ratio)
    GF = num_frame_groups
    GT = num_token_groups
    
    # print(f"num_frame_groups: {GF}, num_token_groups: {GT}")
    
    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        unm_src_idx_frame_list = []
        mrg_src_idx_frame_list = []
        mrg_dst_idx_frame_list = []
        unm_idx_mrg_token_list = []
        src_idx_mrg_token_list = []
        dst_idx_mrg_token_list = []
        unm_idx_unm_token_list = []
        src_idx_unm_token_list = []
        dst_idx_unm_token_list = []
        num_dst_frame_list = []
        a_idx_frame_list = []
        b_idx_frame_list = []
        num_dst_mrg_token_list = []
        a_idx_mrg_token_list = []
        b_idx_mrg_token_list = []
        num_dst_unm_token_list = []
        a_idx_unm_token_list = []
        b_idx_unm_token_list = []
        fr_list = []
        tr_mrg_list = []
        tr_unm_list = []

        def split_temp(x, level):
            b, n, c = x.shape
            src = gather(x, dim=1, index=a_idx_frame_list[level].expand(b, n - num_dst_frame_list[level], c))
            dst = gather(x, dim=1, index=b_idx_frame_list[level].expand(b, num_dst_frame_list[level], c))
            return src, dst
        
        def split_spat_merge(x, level):
            b, n, c = x.shape
            src = gather(x, dim=1, index=a_idx_mrg_token_list[level].expand(b, n - num_dst_mrg_token_list[level], c))
            dst = gather(x, dim=1, index=b_idx_mrg_token_list[level].expand(b, num_dst_mrg_token_list[level], c))
            return src, dst
        
        def split_spat_unmerge(x, level):
            b, n, c = x.shape
            src = gather(x, dim=1, index=a_idx_unm_token_list[level].expand(b, n - num_dst_unm_token_list[level], c))
            dst = gather(x, dim=1, index=b_idx_unm_token_list[level].expand(b, num_dst_unm_token_list[level], c))
            return src, dst
        
        def merge_func_level_1(metric, level, t, h, w):
            # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # starter.record()
            num_dst_frame, a_idx_frame, b_idx_frame = random1d(t, st, metric, generator)
            num_dst_frame_list.append(num_dst_frame)
            a_idx_frame_list.append(a_idx_frame)
            b_idx_frame_list.append(b_idx_frame)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"random1d runtime: {curr_time}ms")
            
            # Cosine similarity between A and B
            # starter.record()
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a_temp, b_temp = split_temp(metric, level)
            scores = a_temp @ b_temp.transpose(-1, -2)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"metric runtime: {curr_time}ms")

            # Can't reduce more than the # tokens in src
            fr = int(metric.shape[1] * fm_ratio[level])
            fr = min(a_temp.shape[1], fr)
            fr_list.append(fr)

            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_src_idx_frame = edge_idx[..., fr:, :]  # Unmerged Tokens
            mrg_src_idx_frame = edge_idx[..., :fr, :]  # Merged Tokens
            mrg_dst_idx_frame = gather(node_idx[..., None], dim=-2, index=mrg_src_idx_frame)  # Merged Dst
            # unm_dst_idx_frame = gather(node_idx[..., None], dim=-2, index=unm_src_idx_frame)  # Unmerged Dst

            # starter.record()
            n, t1, c = a_temp.shape
            _, t2, _ = b_temp.shape
            unm_src = gather(a_temp, dim=-2, index=unm_src_idx_frame.expand(n, t1 - fr, c))
            mrg_src = gather(a_temp, dim=-2, index=mrg_src_idx_frame.expand(n, fr, c)).view(n, fr, 1, c)
            mrg_dst = gather(b_temp, dim=-2, index=mrg_dst_idx_frame.expand(n, fr, c)).view(n, fr, 1, c)
            # unm_dst = gather(b_temp, dim=-2, index=unm_dst_idx_frame.expand(n, t2 - fr, c))
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"gather runtime: {curr_time}ms")

            unm_src_idx_frame_list.append(unm_src_idx_frame)
            mrg_src_idx_frame_list.append(mrg_src_idx_frame)
            mrg_dst_idx_frame_list.append(mrg_dst_idx_frame)

            # Do token merging for merged frames
            # starter.record()
            mrg_metric = torch.cat([mrg_src, mrg_dst], dim=2).reshape(n * fr, 2 * h * w, -1)
            # mrg_metric = torch.mean(torch.cat([mrg_src, mrg_dst], dim=2), dim=2).reshape(n*fr, h*w, -1)
            num_dst_mrg_token, a_idx_mrg_token, b_idx_mrg_token = random3d_diagnal(2, 1, h, sy, w, sx, mrg_metric, generator, tm_ratio[level])
            # num_dst_mrg_token, a_idx_mrg_token, b_idx_mrg_token = random2d(2*h, 2*sy, w, sx, mrg_metric, generator)
            num_dst_mrg_token_list.append(num_dst_mrg_token)
            a_idx_mrg_token_list.append(a_idx_mrg_token)
            b_idx_mrg_token_list.append(b_idx_mrg_token)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"random3d_diagnal runtime: {curr_time}ms")
            
            # Cosine similarity between A and B
            # starter.record()
            mrg_metric = mrg_metric / mrg_metric.norm(dim=-1, keepdim=True)
            a_mrg_token, b_mrg_token = split_spat_merge(mrg_metric, level)
            scores = a_mrg_token @ b_mrg_token.transpose(-1, -2)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"mrg_metric runtime: {curr_time}ms")

            # Can't reduce more than the # tokens in src
            tr_mrg = h*w + int(h * w * tm_ratio[level])
            # tr_mrg = int(h * w * tm_ratio)
            tr_mrg = min(a_mrg_token.shape[1], tr_mrg)
            tr_mrg_list.append(tr_mrg)

            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx_mrg_token = edge_idx[..., tr_mrg:, :]  # Unmerged Tokens (@zyhe: What if there are no unmerge tokens?)
            src_idx_mrg_token = edge_idx[..., :tr_mrg, :]  # Merged Tokens
            dst_idx_mrg_token = gather(node_idx[..., None], dim=-2, index=src_idx_mrg_token)

            unm_idx_mrg_token_list.append(unm_idx_mrg_token)
            src_idx_mrg_token_list.append(src_idx_mrg_token)
            dst_idx_mrg_token_list.append(dst_idx_mrg_token)

            # Do token merging for unmerged frames (TODO: use the entire dst for now)
            # starter.record()
            unm_metric = torch.cat([unm_src, b_temp], dim=1)
            unm_metric = unm_metric.reshape(unm_metric.shape[0]*unm_metric.shape[1], h*w, -1)
            num_dst_unm_token, a_idx_unm_token, b_idx_unm_token = random2d(h, sy, w, sx, unm_metric, generator)
            num_dst_unm_token_list.append(num_dst_unm_token)
            a_idx_unm_token_list.append(a_idx_unm_token)
            b_idx_unm_token_list.append(b_idx_unm_token)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"random2d runtime: {curr_time}ms")
            
            # Cosine similarity between A and B
            # starter.record()
            unm_metric = unm_metric / unm_metric.norm(dim=-1, keepdim=True)
            a_unm_token, b_unm_token = split_spat_unmerge(unm_metric, level)
            scores = a_unm_token @ b_unm_token.transpose(-1, -2)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"unm_metric runtime: {curr_time}ms")

            # Can't reduce more than the # tokens in src
            tr_unm = int(h * w * tm_ratio[level])
            tr_unm = min(a_unm_token.shape[1], tr_unm)
            tr_unm_list.append(tr_unm)

            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx_unm_token = edge_idx[..., tr_unm:, :]  # Unmerged Tokens
            src_idx_unm_token = edge_idx[..., :tr_unm, :]  # Merged Tokens
            dst_idx_unm_token = gather(node_idx[..., None], dim=-2, index=src_idx_unm_token)

            unm_idx_unm_token_list.append(unm_idx_unm_token)
            src_idx_unm_token_list.append(src_idx_unm_token)
            dst_idx_unm_token_list.append(dst_idx_unm_token)

            return a_temp, b_temp, a_mrg_token, b_mrg_token, a_unm_token, b_unm_token
        
        def merge_func_level_2(metric, level, t, s):
            num_dst_frame, a_idx_frame, b_idx_frame = random1d(t, st, metric, generator)
            num_dst_frame_list.append(num_dst_frame)
            a_idx_frame_list.append(a_idx_frame)
            b_idx_frame_list.append(b_idx_frame)

            # Cosine similarity between A and B
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a_temp, b_temp = split_temp(metric, level)
            scores = a_temp @ b_temp.transpose(-1, -2)

            # Can't reduce more than the # tokens in src
            fr = int(metric.shape[1] * fm_ratio[level])
            fr = min(a_temp.shape[1], fr)
            fr_list.append(fr)

            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_src_idx_frame = edge_idx[..., fr:, :]  # Unmerged Tokens
            mrg_src_idx_frame = edge_idx[..., :fr, :]  # Merged Tokens
            mrg_dst_idx_frame = gather(node_idx[..., None], dim=-2, index=mrg_src_idx_frame)  # Merged Dst
            # unm_dst_idx_frame = gather(node_idx[..., None], dim=-2, index=unm_src_idx_frame)  # Unmerged Dst

            n, t1, c = a_temp.shape
            _, t2, _ = b_temp.shape
            unm_src = gather(a_temp, dim=-2, index=unm_src_idx_frame.expand(n, t1 - fr, c))
            mrg_src = gather(a_temp, dim=-2, index=mrg_src_idx_frame.expand(n, fr, c)).view(n, fr, 1, c)
            mrg_dst = gather(b_temp, dim=-2, index=mrg_dst_idx_frame.expand(n, fr, c)).view(n, fr, 1, c)
            # unm_dst = gather(b_temp, dim=-2, index=unm_dst_idx_frame.expand(n, t2 - fr, c))

            unm_src_idx_frame_list.append(unm_src_idx_frame)
            mrg_src_idx_frame_list.append(mrg_src_idx_frame)
            mrg_dst_idx_frame_list.append(mrg_dst_idx_frame)

            # Do token merging for merged frames
            mrg_metric = torch.cat([mrg_src, mrg_dst], dim=2).reshape(n * fr, 2 * s, -1)
            num_dst_mrg_token, a_idx_mrg_token, b_idx_mrg_token = random2d_diagnal(2, 1, s, sx * 2, mrg_metric, generator)
            num_dst_mrg_token_list.append(num_dst_mrg_token)
            a_idx_mrg_token_list.append(a_idx_mrg_token)
            b_idx_mrg_token_list.append(b_idx_mrg_token)
            
            # Cosine similarity between A and B
            mrg_metric = mrg_metric / mrg_metric.norm(dim=-1, keepdim=True)
            a_mrg_token, b_mrg_token = split_spat_merge(mrg_metric, level)
            scores = a_mrg_token @ b_mrg_token.transpose(-1, -2)

            # Can't reduce more than the # tokens in src
            tr_mrg = s + int(s * tm_ratio[level])
            tr_mrg = min(a_mrg_token.shape[1], tr_mrg)
            tr_mrg_list.append(tr_mrg)

            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx_mrg_token = edge_idx[..., tr_mrg:, :]  # Unmerged Tokens
            src_idx_mrg_token = edge_idx[..., :tr_mrg, :]  # Merged Tokens
            dst_idx_mrg_token = gather(node_idx[..., None], dim=-2, index=src_idx_mrg_token)

            unm_idx_mrg_token_list.append(unm_idx_mrg_token)
            src_idx_mrg_token_list.append(src_idx_mrg_token)
            dst_idx_mrg_token_list.append(dst_idx_mrg_token)

            # Do token merging for unmerged frames (TODO: use the entire dst for now)
            unm_metric = torch.cat([unm_src, b_temp], dim=1)
            unm_metric = unm_metric.reshape(unm_metric.shape[0]*unm_metric.shape[1], s, -1)
            num_dst_unm_token, a_idx_unm_token, b_idx_unm_token = random1d(s, sx, unm_metric, generator)
            num_dst_unm_token_list.append(num_dst_unm_token)
            a_idx_unm_token_list.append(a_idx_unm_token)
            b_idx_unm_token_list.append(b_idx_unm_token)
            
            # Cosine similarity between A and B
            unm_metric = unm_metric / unm_metric.norm(dim=-1, keepdim=True)
            a_unm_token, b_unm_token = split_spat_unmerge(unm_metric, level)
            scores = a_unm_token @ b_unm_token.transpose(-1, -2)

            # Can't reduce more than the # tokens in src
            tr_unm = int(s * tm_ratio[level])
            tr_unm = min(a_unm_token.shape[1], tr_unm)
            tr_unm_list.append(tr_unm)

            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx_unm_token = edge_idx[..., tr_unm:, :]  # Unmerged Tokens
            src_idx_unm_token = edge_idx[..., :tr_unm, :]  # Merged Tokens
            dst_idx_unm_token = gather(node_idx[..., None], dim=-2, index=src_idx_unm_token)

            unm_idx_unm_token_list.append(unm_idx_unm_token)
            src_idx_unm_token_list.append(src_idx_unm_token)
            dst_idx_unm_token_list.append(dst_idx_unm_token)
            
            return a_temp, b_temp, a_mrg_token, b_mrg_token, a_unm_token, b_unm_token
            
        # Hierarchical merging
        metric = metric.reshape(B * GF, T // GF, GT, H // GT, GT, W // GT, -1).permute(0, 2, 4, 1, 3, 5, 6)
        metric = metric.reshape(B * GF * GT * GT, T // GF, -1)
        t = metric.shape[1]
        h = H // GT
        w = W // GT
        _, _, a_mrg_token, b_mrg_token, a_unm_token, b_unm_token = merge_func_level_1(metric, 0, t, h, w)
        
        for level in range(1, L, 1):
            n, t1, c = a_mrg_token.shape
            unm_src_mrg_token = gather(a_mrg_token, dim=-2, index=unm_idx_mrg_token_list[level-1].expand(n, t1 - tr_mrg_list[level-1], c))
            mrg_src_mrg_token = gather(a_mrg_token, dim=-2, index=src_idx_mrg_token_list[level-1].expand(n, tr_mrg_list[level-1], c))
            dst_mrg_token = b_mrg_token
            dst_mrg_token = dst_mrg_token.scatter_reduce(-2, dst_idx_mrg_token_list[level-1].expand(n, tr_mrg_list[level-1], c), mrg_src_mrg_token, reduce="mean")
            out_mrg = torch.cat([unm_src_mrg_token, dst_mrg_token], dim=1)
            out_mrg = out_mrg.reshape(B * GF * GT * GT, -1, out_mrg.shape[1]*out_mrg.shape[2])
            
            n, t1, c = a_unm_token.shape
            unm_src_unm_token = gather(a_unm_token, dim=-2, index=unm_idx_unm_token_list[level-1].expand(n, t1 - tr_unm_list[level-1], c))
            mrg_src_unm_token = gather(a_unm_token, dim=-2, index=src_idx_unm_token_list[level-1].expand(n, tr_unm_list[level-1], c))
            dst_unm_token = b_unm_token
            dst_unm_token = dst_unm_token.scatter_reduce(-2, dst_idx_unm_token_list[level-1].expand(n, tr_unm_list[level-1], c), mrg_src_unm_token, reduce="mean")
            out_unm = torch.cat([unm_src_unm_token, dst_unm_token], dim=1)
            out_unm = out_unm.reshape(B * GF * GT * GT, -1, out_unm.shape[1]*out_unm.shape[2])
            out_unm_src = out_unm[:, :unm_src_idx_frame_list[level-1].shape[1], :]
            out_unm_dst = out_unm[:, unm_src_idx_frame_list[level-1].shape[1]:, :]

            out_unm_dst.scatter_(dim=-2, index=mrg_dst_idx_frame_list[level-1].expand(out_mrg.shape), src=out_mrg)
            out = torch.cat([out_unm_src, out_unm_dst], dim=1)
            metric = out
            t = metric.shape[1]
            s = metric.shape[2] // c
            _, _, a_mrg_token, b_mrg_token, a_unm_token, b_unm_token = merge_func_level_2(metric, level, t, s)
            
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        x = x.reshape(B * GF, T // GF, GT, H // GT, GT, W // GT, C).permute(0, 2, 4, 1, 3, 5, 6)
        x = x.reshape(B * GF * GT * GT, T // GF, -1)
        for level in range(L):
            b = mrg_dst_idx_frame_list[level].shape[0]
            s = x.shape[-1] // C
            # Do frame merging
            src_frame, dst_frame = split_temp(x, level)

            n, t1, c = src_frame.shape
            _, t2, _ = dst_frame.shape
            unm_src_frame = gather(src_frame, dim=-2, index=unm_src_idx_frame_list[level].expand(n, t1 - fr_list[level], c))
            mrg_src_frame = gather(src_frame, dim=-2, index=mrg_src_idx_frame_list[level].expand(n, fr_list[level], c)).view(n, fr_list[level], 1, c)
            mrg_dst_frame = gather(dst_frame, dim=-2, index=mrg_dst_idx_frame_list[level].expand(n, fr_list[level], c)).view(n, fr_list[level], 1, c)
            # unm_dst_frame = gather(dst_frame, dim=-2, index=unm_dst_idx_frame.expand(n, t2 - fr, c))
            
            # Do token merging for merged frames
            x = torch.cat([mrg_src_frame, mrg_dst_frame], dim=2)
            x = x.reshape(n * x.shape[1], 2 * s, -1)
            # x = torch.mean(torch.cat([mrg_src_frame, mrg_dst_frame], dim=2), dim=2)
            # x = x.reshape(n*x.shape[1], H*W, -1)
            src, dst = split_spat_merge(x, level)
            n, t1, c = src.shape
            
            unm_src_mrg_token = gather(src, dim=-2, index=unm_idx_mrg_token_list[level].expand(n, t1 - tr_mrg_list[level], c))
            mrg_src_mrg_token = gather(src, dim=-2, index=src_idx_mrg_token_list[level].expand(n, tr_mrg_list[level], c))
            dst_mrg_token = dst
            if mode is not None:
                dst_mrg_token = dst_mrg_token.scatter_reduce(-2, dst_idx_mrg_token_list[level].expand(n, tr_mrg_list[level], c), mrg_src_mrg_token, reduce=mode)
            
            # print(f"unm_src_mrg_token: {unm_src_mrg_token.shape}, dst_mrg_token: {dst_mrg_token.shape}")
            out_mrg = torch.cat([unm_src_mrg_token, dst_mrg_token], dim=1)
            # print(f"out_mrg: {out_mrg.shape}")
            out_mrg = out_mrg.reshape(b, -1, out_mrg.shape[1]*out_mrg.shape[2])
            # print(f"out_mrg: {out_mrg.shape}")

            # Do token merging for unmerged frames (TODO: use the entire dst for now)
            x = torch.cat([unm_src_frame, dst_frame], dim=1)
            x = x.reshape(x.shape[0]*x.shape[1], s, -1)
            src, dst = split_spat_unmerge(x, level)
            n, t1, c = src.shape
            
            unm_src_unm_token = gather(src, dim=-2, index=unm_idx_unm_token_list[level].expand(n, t1 - tr_unm_list[level], c))
            mrg_src_unm_token = gather(src, dim=-2, index=src_idx_unm_token_list[level].expand(n, tr_unm_list[level], c))
            dst_unm_token = dst
            if mode is not None:
                dst_unm_token = dst_unm_token.scatter_reduce(-2, dst_idx_unm_token_list[level].expand(n, tr_unm_list[level], c), mrg_src_unm_token, reduce=mode)
            
            out_unm = torch.cat([unm_src_unm_token, dst_unm_token], dim=1)
            out_unm = out_unm.reshape(b, -1, out_unm.shape[1]*out_unm.shape[2])
            out_unm_src = out_unm[:, :unm_src_idx_frame_list[level].shape[1], :]
            out_unm_dst = out_unm[:, unm_src_idx_frame_list[level].shape[1]:, :]

            # print(f"out_unm_dst: {out_unm_dst.shape}, mrg_dst_idx_frame_list[level]: {mrg_dst_idx_frame_list[level].shape}")
            # if out_mrg.shape[-1] > out_unm_dst.shape[-1]:
            #     out_mrg = out_mrg[..., :out_unm_dst.shape[-1]]
            out_unm_dst.scatter_(dim=-2, index=mrg_dst_idx_frame_list[level].expand(out_mrg.shape), src=out_mrg)
            out = torch.cat([out_unm_src, out_unm_dst], dim=1)
            x = out
            
        out = out.reshape(B * GF, GT, GT, out.shape[1], -1, C).permute(0, 3, 1, 2, 4, 5)
        out = out.reshape(out.shape[0] * out.shape[1], -1, C)

        return out

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(B * GF, -1, GT * GT, x.shape[1] // (GT * GT), x.shape[2]).permute(0, 2, 1, 3, 4)
        x = x.reshape(B * GF * GT * GT, x.shape[2], x.shape[3] * x.shape[4])
        for level in range(L - 1, -1, -1):
            b = mrg_dst_idx_frame_list[level].shape[0]
            s = x.shape[-1] // C # C is the number of channels so, s is the number of tokens = H' * W'
            # Split unmerged tokens and merged tokens first
            unm_token = x
            unm_token = unm_token.reshape(unm_token.shape[0] * unm_token.shape[1], s, -1)
            unm_frame_len = unm_src_idx_frame_list[level].shape[1]
            mrg_token = x[:, unm_frame_len:, :]
            n, _, c = mrg_token.shape
            mrg_token = gather(mrg_token, dim=-2, index=mrg_dst_idx_frame_list[level].expand(n, fr_list[level], c))
            mrg_token = mrg_token.reshape(mrg_token.shape[0] * mrg_token.shape[1], s, -1)
            
            # Token unmerge: Split unm and dst for unmerged tokens
            unm_len = unm_idx_unm_token_list[level].shape[1]
            unm, dst = unm_token[..., :unm_len, :], unm_token[..., unm_len:, :]
            n, _, c = unm.shape

            src = gather(dst, dim=-2, index=dst_idx_unm_token_list[level].expand(n, tr_unm_list[level], c))

            # Combine back to the original shape
            s = unm_len + tr_unm_list[level] + num_dst_unm_token_list[level]
            out_unm = torch.zeros(n, s, c, device=x.device, dtype=x.dtype)
            out_unm.scatter_(dim=-2, index=b_idx_unm_token_list[level].expand(n, num_dst_unm_token_list[level], c), src=dst)
            out_unm.scatter_(dim=-2, index=gather(a_idx_unm_token_list[level].expand(n, a_idx_unm_token_list[level].shape[1], 1), dim=1, index=unm_idx_unm_token_list[level]).expand(n, unm_len, c), src=unm)
            out_unm.scatter_(dim=-2, index=gather(a_idx_unm_token_list[level].expand(n, a_idx_unm_token_list[level].shape[1], 1), dim=1, index=src_idx_unm_token_list[level]).expand(n, tr_unm_list[level], c), src=src)
            out_unm = out_unm.reshape(b, -1, s * c)
            unm_src = out_unm[:, :unm_src_idx_frame_list[level].shape[1], :]
            unm_dst = out_unm[:, unm_src_idx_frame_list[level].shape[1]:, :]

            # Token unmerge: Split unm and dst for merged tokens
            unm_len = unm_idx_mrg_token_list[level].shape[1]
            unm, dst = mrg_token[..., :unm_len, :], mrg_token[..., unm_len:, :]
            n, _, c = unm.shape

            src = gather(dst, dim=-2, index=dst_idx_mrg_token_list[level].expand(n, tr_mrg_list[level], c))

            # Combine back to the original shape
            out_mrg = torch.zeros(n, 2 * s, c, device=x.device, dtype=x.dtype)
            # out_mrg = torch.zeros(b, H*W, c, device=x.device, dtype=x.dtype)
            out_mrg.scatter_(dim=-2, index=b_idx_mrg_token_list[level].expand(n, num_dst_mrg_token_list[level], c), src=dst)
            out_mrg.scatter_(dim=-2, index=gather(a_idx_mrg_token_list[level].expand(n, a_idx_mrg_token_list[level].shape[1], 1), dim=1, index=unm_idx_mrg_token_list[level]).expand(n, unm_len, c), src=unm)
            out_mrg.scatter_(dim=-2, index=gather(a_idx_mrg_token_list[level].expand(n, a_idx_mrg_token_list[level].shape[1], 1), dim=1, index=src_idx_mrg_token_list[level]).expand(n, tr_mrg_list[level], c), src=src)
            
            # Frame unmerge
            out_mrg = out_mrg.reshape(b, -1, 2, s * c)
            mrg_src = out_mrg[:, :, 0, :]
            mrg_dst = out_mrg[:, :, 1, :]

            # Combine back to the original shape
            _, t1, c = unm_src.shape
            _, t2, c = mrg_src.shape
            _, t3, c = unm_dst.shape
            out = torch.zeros(b, t1 + t2 + t3, c, device=x.device, dtype=x.dtype)
            # mrg_dst = out_mrg.reshape(B, -1, c)
            unm_dst.scatter_(dim=-2, index=mrg_dst_idx_frame_list[level].expand(mrg_dst.shape), src=mrg_dst)
            # mrg_src = gather(unm_dst, dim=-2, index=mrg_dst_idx_frame.expand(B, fr, c))
            out.scatter_(dim=-2, index=b_idx_frame_list[level].expand(unm_dst.shape), src=unm_dst)
            out.scatter_(dim=-2, index=gather(a_idx_frame_list[level].expand(b, a_idx_frame_list[level].shape[1], 1), dim=1, index=unm_src_idx_frame_list[level]).expand(unm_src.shape), src=unm_src)
            out.scatter_(dim=-2, index=gather(a_idx_frame_list[level].expand(b, a_idx_frame_list[level].shape[1], 1), dim=1, index=mrg_src_idx_frame_list[level]).expand(mrg_src.shape), src=mrg_src)
            x = out
            
        out = out.reshape(B * GF, GT, GT, T // GF, H // GT, W // GT, C).permute(0, 3, 1, 4, 2, 5, 6)
        out = out.reshape(B*T, H*W, C)

        return out
    
    return merge, unmerge


def bipartite_soft_matching_random3d_temporal(metric: torch.Tensor,
                                     t: int, w: int, h: int, st: int, sx: int, sy: int, 
                                     bm_ratio: Union[float, list[float]], tm_ratio: Union[float, list[float]],
                                     num_frame_groups: int = 1, num_token_groups: int = 8,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B = metric.shape[0]
    T = t
    H = h
    W = w
    C = metric.shape[-1] // t

    if not isinstance(bm_ratio, list):
        bm_ratio = [bm_ratio]
    if not isinstance(tm_ratio, list):
        tm_ratio = [tm_ratio]
    assert len(bm_ratio) == len(tm_ratio)
    
    L = len(bm_ratio)
    GF = num_frame_groups
    GT = num_token_groups
    
    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        unm_src_idx_block_list = []
        mrg_src_idx_block_list = []
        mrg_dst_idx_block_list = []
        unm_idx_mrg_token_list = []
        src_idx_mrg_token_list = []
        dst_idx_mrg_token_list = []
        unm_idx_unm_token_list = []
        src_idx_unm_token_list = []
        dst_idx_unm_token_list = []
        num_dst_block_list = []
        a_idx_block_list = []
        b_idx_block_list = []
        num_dst_mrg_token_list = []
        a_idx_mrg_token_list = []
        b_idx_mrg_token_list = []
        num_dst_unm_token_list = []
        a_idx_unm_token_list = []
        b_idx_unm_token_list = []
        br_list = []
        tr_mrg_list = []
        tr_unm_list = []

        def split_spat(x, level):
            b, n, c = x.shape
            src = gather(x, dim=1, index=a_idx_block_list[level].expand(b, n - num_dst_block_list[level], c))
            dst = gather(x, dim=1, index=b_idx_block_list[level].expand(b, num_dst_block_list[level], c))
            return src, dst
        
        def split_temp_merge(x, level):
            b, n, c = x.shape
            src = gather(x, dim=1, index=a_idx_mrg_token_list[level].expand(b, n - num_dst_mrg_token_list[level], c))
            dst = gather(x, dim=1, index=b_idx_mrg_token_list[level].expand(b, num_dst_mrg_token_list[level], c))
            return src, dst
        
        def split_temp_unmerge(x, level):
            b, n, c = x.shape
            src = gather(x, dim=1, index=a_idx_unm_token_list[level].expand(b, n - num_dst_unm_token_list[level], c))
            dst = gather(x, dim=1, index=b_idx_unm_token_list[level].expand(b, num_dst_unm_token_list[level], c))
            return src, dst
        
        def merge_func_level_1(metric, level, t, h, w):
            # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # starter.record()
            num_dst_block, a_idx_block, b_idx_block = random2d(h, sy, w, sx, metric, generator)
            num_dst_block_list.append(num_dst_block)
            a_idx_block_list.append(a_idx_block)
            b_idx_block_list.append(b_idx_block)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"random1d runtime: {curr_time}ms")
            
            # Cosine similarity between A and B
            # starter.record()
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a_block, b_block = split_spat(metric, level)
            scores = a_block @ b_block.transpose(-1, -2)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"metric runtime: {curr_time}ms")

            # Can't reduce more than the # tokens in src
            br = int(metric.shape[1] * bm_ratio[level])
            br = min(a_block.shape[1], br)
            br_list.append(br)

            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_src_idx_block = edge_idx[..., br:, :]  # Unmerged Tokens
            mrg_src_idx_block = edge_idx[..., :br, :]  # Merged Tokens
            mrg_dst_idx_block = gather(node_idx[..., None], dim=-2, index=mrg_src_idx_block)  # Merged Dst
            # unm_dst_idx_block = gather(node_idx[..., None], dim=-2, index=unm_src_idx_block)  # Unmerged Dst

            # starter.record()
            n, t1, c = a_block.shape
            _, t2, _ = b_block.shape
            unm_src = gather(a_block, dim=-2, index=unm_src_idx_block.expand(n, t1 - br, c))
            mrg_src = gather(a_block, dim=-2, index=mrg_src_idx_block.expand(n, br, c)).view(n, br, 1, c)
            mrg_dst = gather(a_block, dim=-2, index=mrg_dst_idx_block.expand(n, br, c)).view(n, br, 1, c)
            # unm_dst = gather(b_temp, dim=-2, index=unm_dst_idx_frame.expand(n, t2 - br, c))
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"gather runtime: {curr_time}ms")

            unm_src_idx_block_list.append(unm_src_idx_block)
            mrg_src_idx_block_list.append(mrg_src_idx_block)
            mrg_dst_idx_block_list.append(mrg_dst_idx_block)

            # Do token merging for merged blocks -> first time
            # starter.record()
            mrg_metric = torch.cat([mrg_src, mrg_dst], dim=2).reshape(n * br, 2 * t, -1)
            # mrg_metric = torch.mean(torch.cat([mrg_src, mrg_dst], dim=2), dim=2).reshape(n*fr, h*w, -1)
            num_dst_mrg_token, a_idx_mrg_token, b_idx_mrg_token = random2d_diagnal(2, 1, t, int(st*1.5), mrg_metric, generator, tm_ratio[level])
            # num_dst_mrg_token, a_idx_mrg_token, b_idx_mrg_token = random1d(2*t, 2*st, mrg_metric, generator)
            num_dst_mrg_token_list.append(num_dst_mrg_token)
            a_idx_mrg_token_list.append(a_idx_mrg_token)
            b_idx_mrg_token_list.append(b_idx_mrg_token)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"random3d_diagnal runtime: {curr_time}ms")
            
            # Cosine similarity between A and B
            # starter.record()
            mrg_metric = mrg_metric / mrg_metric.norm(dim=-1, keepdim=True)
            a_mrg_token, b_mrg_token = split_temp_merge(mrg_metric, level)
            scores = a_mrg_token @ b_mrg_token.transpose(-1, -2)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"mrg_metric runtime: {curr_time}ms")

            # Can't reduce more than the # tokens in src
            # tr_mrg = t
            tr_mrg = t + int(t * tm_ratio[level])
            tr_mrg = min(a_mrg_token.shape[1], tr_mrg)
            tr_mrg_list.append(tr_mrg)

            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx_mrg_token = edge_idx[..., tr_mrg:, :]  # Unmerged Tokens
            src_idx_mrg_token = edge_idx[..., :tr_mrg, :]  # Merged Tokens
            dst_idx_mrg_token = gather(node_idx[..., None], dim=-2, index=src_idx_mrg_token)

            unm_idx_mrg_token_list.append(unm_idx_mrg_token)
            src_idx_mrg_token_list.append(src_idx_mrg_token)
            dst_idx_mrg_token_list.append(dst_idx_mrg_token)
            
            # n, t1, c = a_mrg_token.shape
            # unm_src_mrg_token = gather(a_mrg_token, dim=-2, index=unm_idx_mrg_token.expand(n, t1 - tr_mrg, c))
            # mrg_src_mrg_token = gather(a_mrg_token, dim=-2, index=src_idx_mrg_token.expand(n, tr_mrg, c))
            # dst_mrg_token = b_mrg_token
            # dst_mrg_token = dst_mrg_token.scatter_reduce(-2, dst_idx_mrg_token.expand(n, tr_mrg, c), mrg_src_mrg_token, reduce="mean")
            
            # # Do token merging for merged blocks -> second time
            # mrg_metric = torch.cat([unm_src_mrg_token, dst_mrg_token], dim=1)
            # # mrg_metric = torch.mean(torch.cat([mrg_src, mrg_dst], dim=2), dim=2).reshape(n*fr, h*w, -1)
            # num_dst_mrg_token, a_idx_mrg_token, b_idx_mrg_token = random1d(t, st, mrg_metric, generator)
            # # num_dst_mrg_token, a_idx_mrg_token, b_idx_mrg_token = random1d(2*t, 2*st, mrg_metric, generator)
            # num_dst_mrg_token_list.append(num_dst_mrg_token)
            # a_idx_mrg_token_list.append(a_idx_mrg_token)
            # b_idx_mrg_token_list.append(b_idx_mrg_token)
            
            # # Cosine similarity between A and B
            # mrg_metric = mrg_metric / mrg_metric.norm(dim=-1, keepdim=True)
            # a_mrg_token, b_mrg_token = split_temp_merge(mrg_metric, level + 1)
            # scores = a_mrg_token @ b_mrg_token.transpose(-1, -2)

            # # Can't reduce more than the # tokens in src
            # tr_mrg = int(t * tm_ratio[level])
            # tr_mrg = min(a_mrg_token.shape[1], tr_mrg)
            # tr_mrg_list.append(tr_mrg)

            # # Find the most similar greedily
            # node_max, node_idx = scores.max(dim=-1)
            # edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            # unm_idx_mrg_token = edge_idx[..., tr_mrg:, :]  # Unmerged Tokens
            # src_idx_mrg_token = edge_idx[..., :tr_mrg, :]  # Merged Tokens
            # dst_idx_mrg_token = gather(node_idx[..., None], dim=-2, index=src_idx_mrg_token)

            # unm_idx_mrg_token_list.append(unm_idx_mrg_token)
            # src_idx_mrg_token_list.append(src_idx_mrg_token)
            # dst_idx_mrg_token_list.append(dst_idx_mrg_token)

            # Do token merging for unmerged blocks (TODO: use the entire dst for now)
            # starter.record()
            unm_metric = torch.cat([unm_src, b_block], dim=1)
            unm_metric = unm_metric.reshape(unm_metric.shape[0]*unm_metric.shape[1], t, -1)
            num_dst_unm_token, a_idx_unm_token, b_idx_unm_token = random1d(t, st, unm_metric, generator)
            num_dst_unm_token_list.append(num_dst_unm_token)
            a_idx_unm_token_list.append(a_idx_unm_token)
            b_idx_unm_token_list.append(b_idx_unm_token)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"random2d runtime: {curr_time}ms")
            
            # Cosine similarity between A and B
            # starter.record()
            unm_metric = unm_metric / unm_metric.norm(dim=-1, keepdim=True)
            a_unm_token, b_unm_token = split_temp_unmerge(unm_metric, level)
            scores = a_unm_token @ b_unm_token.transpose(-1, -2)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # print(f"unm_metric runtime: {curr_time}ms")

            # Can't reduce more than the # tokens in src
            tr_unm = int(t * tm_ratio[level])
            tr_unm = min(a_unm_token.shape[1], tr_unm)
            tr_unm_list.append(tr_unm)

            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx_unm_token = edge_idx[..., tr_unm:, :]  # Unmerged Tokens
            src_idx_unm_token = edge_idx[..., :tr_unm, :]  # Merged Tokens
            dst_idx_unm_token = gather(node_idx[..., None], dim=-2, index=src_idx_unm_token)

            unm_idx_unm_token_list.append(unm_idx_unm_token)
            src_idx_unm_token_list.append(src_idx_unm_token)
            dst_idx_unm_token_list.append(dst_idx_unm_token)

            return a_block, b_block, a_mrg_token, b_mrg_token, a_unm_token, b_unm_token
        
        def merge_func_level_2(metric, level, t, s):
            num_dst_block, a_idx_block, b_idx_block = random1d(s, sx, unm_metric, generator)
            num_dst_block_list.append(num_dst_block)
            a_idx_block_list.append(a_idx_block)
            b_idx_block_list.append(b_idx_block)

            # Cosine similarity between A and B
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a_block, b_block = split_spat(metric, level)
            scores = a_block @ b_block.transpose(-1, -2)

            # Can't reduce more than the # tokens in src
            br = int(metric.shape[1] * bm_ratio[level])
            br = min(a_block.shape[1], br)
            br_list.append(br)

            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_src_idx_block = edge_idx[..., br:, :]  # Unmerged Tokens
            mrg_src_idx_block = edge_idx[..., :br, :]  # Merged Tokens
            mrg_dst_idx_block = gather(node_idx[..., None], dim=-2, index=mrg_src_idx_block)  # Merged Dst
            # unm_dst_idx_block = gather(node_idx[..., None], dim=-2, index=unm_src_idx_block)  # Unmerged Dst

            n, t1, c = a_block.shape
            _, t2, _ = b_block.shape
            unm_src = gather(a_block, dim=-2, index=unm_src_idx_block.expand(n, t1 - br, c))
            mrg_src = gather(a_block, dim=-2, index=mrg_src_idx_block.expand(n, br, c)).view(n, br, 1, c)
            mrg_dst = gather(a_block, dim=-2, index=mrg_dst_idx_block.expand(n, br, c)).view(n, br, 1, c)
            # unm_dst = gather(b_temp, dim=-2, index=unm_dst_idx_frame.expand(n, t2 - fr, c))

            unm_src_idx_block_list.append(unm_src_idx_block)
            mrg_src_idx_block_list.append(mrg_src_idx_block)
            mrg_dst_idx_block_list.append(mrg_dst_idx_block)

            # Do token merging for merged blocks
            mrg_metric = torch.cat([mrg_src, mrg_dst], dim=2).reshape(n * br, 2 * t, -1)
            num_dst_mrg_token, a_idx_mrg_token, b_idx_mrg_token = random2d_diagnal(2, 1, t, st * 2, mrg_metric, generator, tm_ratio[level])
            num_dst_mrg_token_list.append(num_dst_mrg_token)
            a_idx_mrg_token_list.append(a_idx_mrg_token)
            b_idx_mrg_token_list.append(b_idx_mrg_token)
            
            # Cosine similarity between A and B
            mrg_metric = mrg_metric / mrg_metric.norm(dim=-1, keepdim=True)
            a_mrg_token, b_mrg_token = split_temp_merge(mrg_metric, level)
            scores = a_mrg_token @ b_mrg_token.transpose(-1, -2)

            # Can't reduce more than the # tokens in src
            tr_mrg = t + int(t * tm_ratio[level])
            tr_mrg = min(a_mrg_token.shape[1], tr_mrg)
            tr_mrg_list.append(tr_mrg)

            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx_mrg_token = edge_idx[..., tr_mrg:, :]  # Unmerged Tokens
            src_idx_mrg_token = edge_idx[..., :tr_mrg, :]  # Merged Tokens
            dst_idx_mrg_token = gather(node_idx[..., None], dim=-2, index=src_idx_mrg_token)

            unm_idx_mrg_token_list.append(unm_idx_mrg_token)
            src_idx_mrg_token_list.append(src_idx_mrg_token)
            dst_idx_mrg_token_list.append(dst_idx_mrg_token)

            # Do token merging for unmerged frames (TODO: use the entire dst for now)
            unm_metric = torch.cat([unm_src, b_block], dim=1)
            unm_metric = unm_metric.reshape(unm_metric.shape[0]*unm_metric.shape[1], t, -1)
            num_dst_unm_token, a_idx_unm_token, b_idx_unm_token = random1d(t, st, unm_metric, generator)
            num_dst_unm_token_list.append(num_dst_unm_token)
            a_idx_unm_token_list.append(a_idx_unm_token)
            b_idx_unm_token_list.append(b_idx_unm_token)
            
            # Cosine similarity between A and B
            unm_metric = unm_metric / unm_metric.norm(dim=-1, keepdim=True)
            a_unm_token, b_unm_token = split_temp_unmerge(unm_metric, level)
            scores = a_unm_token @ b_unm_token.transpose(-1, -2)

            # Can't reduce more than the # tokens in src
            tr_unm = int(t * tm_ratio[level])
            tr_unm = min(a_unm_token.shape[1], tr_unm)
            tr_unm_list.append(tr_unm)

            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx_unm_token = edge_idx[..., tr_unm:, :]  # Unmerged Tokens
            src_idx_unm_token = edge_idx[..., :tr_unm, :]  # Merged Tokens
            dst_idx_unm_token = gather(node_idx[..., None], dim=-2, index=src_idx_unm_token)

            unm_idx_unm_token_list.append(unm_idx_unm_token)
            src_idx_unm_token_list.append(src_idx_unm_token)
            dst_idx_unm_token_list.append(dst_idx_unm_token)
            
            return a_block, b_block, a_mrg_token, b_mrg_token, a_unm_token, b_unm_token
            
        # Hierarchical merging
        metric = metric.reshape(B, GT, H // GT, GT, W // GT, GF, T // GF, -1).permute(0, 1, 3, 5, 2, 4, 6, 7)
        metric = metric.reshape(B * GT * GT * GF, (H // GT) * (W // GT), -1)
        t = T // GF
        h = H // GT
        w = W // GT
        _, _, a_mrg_token, b_mrg_token, a_unm_token, b_unm_token = merge_func_level_1(metric, 0, t, h, w)
        
        for level in range(1, L, 1):
            n, t1, c = a_mrg_token.shape
            unm_src_mrg_token = gather(a_mrg_token, dim=-2, index=unm_idx_mrg_token_list[level].expand(n, t1 - tr_mrg_list[level], c))
            mrg_src_mrg_token = gather(a_mrg_token, dim=-2, index=src_idx_mrg_token_list[level].expand(n, tr_mrg_list[level], c))
            dst_mrg_token = b_mrg_token
            dst_mrg_token = dst_mrg_token.scatter_reduce(-2, dst_idx_mrg_token_list[level].expand(n, tr_mrg_list[level], c), mrg_src_mrg_token, reduce="mean")
            out_mrg = torch.cat([unm_src_mrg_token, dst_mrg_token], dim=1)
            out_mrg = out_mrg.reshape(B * GT * GT * GF, -1, out_mrg.shape[1]*out_mrg.shape[2])
            
            n, t1, c = a_unm_token.shape
            unm_src_unm_token = gather(a_unm_token, dim=-2, index=unm_idx_unm_token_list[level-1].expand(n, t1 - tr_unm_list[level-1], c))
            mrg_src_unm_token = gather(a_unm_token, dim=-2, index=src_idx_unm_token_list[level-1].expand(n, tr_unm_list[level-1], c))
            dst_unm_token = b_unm_token
            dst_unm_token = dst_unm_token.scatter_reduce(-2, dst_idx_unm_token_list[level-1].expand(n, tr_unm_list[level-1], c), mrg_src_unm_token, reduce="mean")
            out_unm = torch.cat([unm_src_unm_token, dst_unm_token], dim=1)
            out_unm = out_unm.reshape(B * GT * GT * GF, -1, out_unm.shape[1]*out_unm.shape[2])
            out_unm_src = out_unm[:, :unm_src_idx_block_list[level-1].shape[1], :]
            out_unm_dst = out_unm[:, unm_src_idx_block_list[level-1].shape[1]:, :]

            out_unm_dst.scatter_(dim=-2, index=mrg_dst_idx_block_list[level-1].expand(out_mrg.shape), src=out_mrg)
            out = torch.cat([out_unm_src, out_unm_dst], dim=1)
            metric = out
            s = metric.shape[1]
            t = metric.shape[2] // c
            _, _, a_mrg_token, b_mrg_token, a_unm_token, b_unm_token = merge_func_level_2(metric, level, t, s)
            
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        x = x.reshape(B, GT, H // GT, GT, W // GT, GF, T // GF, C).permute(0, 1, 3, 5, 2, 4, 6, 7)
        x = x.reshape(B * GT * GT * GF, (H // GT) * (W // GT), -1)
        for level in range(L):
            b = mrg_dst_idx_block_list[level].shape[0]
            t = x.shape[-1] // C
            # Do block merging
            src_block, dst_block = split_spat(x, level)

            n, t1, c = src_block.shape
            _, t2, _ = dst_block.shape
            unm_src_block = gather(src_block, dim=-2, index=unm_src_idx_block_list[level].expand(n, t1 - br_list[level], c))
            mrg_src_block = gather(src_block, dim=-2, index=mrg_src_idx_block_list[level].expand(n, br_list[level], c)).view(n, br_list[level], 1, c)
            mrg_dst_block = gather(dst_block, dim=-2, index=mrg_dst_idx_block_list[level].expand(n, br_list[level], c)).view(n, br_list[level], 1, c)
            # unm_dst_frame = gather(dst_frame, dim=-2, index=unm_dst_idx_frame.expand(n, t2 - fr, c))
            
            # Do token merging for merged blocks -> first time
            x = torch.cat([mrg_src_block, mrg_dst_block], dim=2)
            x = x.reshape(n * x.shape[1], 2 * t, -1)
            # x = torch.mean(torch.cat([mrg_src_block, mrg_dst_block], dim=2), dim=2)
            # x = x.reshape(n*x.shape[1], t, -1)
            src, dst = split_temp_merge(x, level)
            n, t1, c = src.shape
            
            unm_src_mrg_token = gather(src, dim=-2, index=unm_idx_mrg_token_list[level].expand(n, t1 - tr_mrg_list[level], c))
            mrg_src_mrg_token = gather(src, dim=-2, index=src_idx_mrg_token_list[level].expand(n, tr_mrg_list[level], c))
            dst_mrg_token = dst
            if mode is not None:
                dst_mrg_token = dst_mrg_token.scatter_reduce(-2, dst_idx_mrg_token_list[level].expand(n, tr_mrg_list[level], c), mrg_src_mrg_token, reduce=mode)
            
            out_mrg = torch.cat([unm_src_mrg_token, dst_mrg_token], dim=1)
            # out_mrg = out_mrg.reshape(b, -1, out_mrg.shape[1]*out_mrg.shape[2])
            
            # # Do token merging for merged blocks -> second time
            # x = out_mrg
            # # x = torch.mean(torch.cat([mrg_src_block, mrg_dst_block], dim=2), dim=2)
            # # x = x.reshape(n*x.shape[1], t, -1)
            # src, dst = split_temp_merge(x, level + 1)
            # n, t1, c = src.shape
            
            # unm_src_mrg_token = gather(src, dim=-2, index=unm_idx_mrg_token_list[level+1].expand(n, t1 - tr_mrg_list[level+1], c))
            # mrg_src_mrg_token = gather(src, dim=-2, index=src_idx_mrg_token_list[level+1].expand(n, tr_mrg_list[level+1], c))
            # dst_mrg_token = dst
            # if mode is not None:
            #     dst_mrg_token = dst_mrg_token.scatter_reduce(-2, dst_idx_mrg_token_list[level+1].expand(n, tr_mrg_list[level+1], c), mrg_src_mrg_token, reduce=mode)
            
            # out_mrg = torch.cat([unm_src_mrg_token, dst_mrg_token], dim=1)
            out_mrg = out_mrg.reshape(b, -1, out_mrg.shape[1]*out_mrg.shape[2])

            # Do token merging for unmerged blocks (TODO: use the entire dst for now)
            x = torch.cat([unm_src_block, dst_block], dim=1)
            x = x.reshape(x.shape[0]*x.shape[1], t, -1)
            src, dst = split_temp_unmerge(x, level)
            n, t1, c = src.shape
            
            unm_src_unm_token = gather(src, dim=-2, index=unm_idx_unm_token_list[level].expand(n, t1 - tr_unm_list[level], c))
            mrg_src_unm_token = gather(src, dim=-2, index=src_idx_unm_token_list[level].expand(n, tr_unm_list[level], c))
            dst_unm_token = dst
            if mode is not None:
                dst_unm_token = dst_unm_token.scatter_reduce(-2, dst_idx_unm_token_list[level].expand(n, tr_unm_list[level], c), mrg_src_unm_token, reduce=mode)
            
            out_unm = torch.cat([unm_src_unm_token, dst_unm_token], dim=1)
            out_unm = out_unm.reshape(b, -1, out_unm.shape[1]*out_unm.shape[2])
            out_unm_src = out_unm[:, :unm_src_idx_block_list[level].shape[1], :]
            out_unm_dst = out_unm[:, unm_src_idx_block_list[level].shape[1]:, :]

            out_unm_dst.scatter_(dim=-2, index=mrg_dst_idx_block_list[level].expand(out_mrg.shape), src=out_mrg)
            out = torch.cat([out_unm_src, out_unm_dst], dim=1)
            x = out
            
        out = out.reshape(B, GT, GT, GF, out.shape[1], -1, C).permute(0, 1, 2, 4, 3, 5, 6)
        out = out.reshape(-1, out.shape[-3] * out.shape[-2], C)

        return out

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(B, x.shape[0] // B, -1, C)
        x = x.reshape(B, GT * GT, x.shape[1] // (GT * GT), GF, -1, C).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B * GT * GT * GF, x.shape[3], x.shape[4] * x.shape[5])
        for level in range(L - 1, -1, -1):
            b = mrg_dst_idx_block_list[level].shape[0]
            t = x.shape[-1] // C
            # Split unmerged tokens and merged tokens first
            unm_token = x
            unm_token = unm_token.reshape(unm_token.shape[0] * unm_token.shape[1], t, -1)
            unm_block_len = unm_src_idx_block_list[level].shape[1]
            mrg_token = x[:, unm_block_len:, :]
            n, _, c = mrg_token.shape
            mrg_token = gather(mrg_token, dim=-2, index=mrg_dst_idx_block_list[level].expand(n, br_list[level], c))
            mrg_token = mrg_token.reshape(mrg_token.shape[0] * mrg_token.shape[1], t, -1)
            
            # Token unmerge: Split unm and dst for unmerged tokens
            unm_len = unm_idx_unm_token_list[level].shape[1]
            unm, dst = unm_token[..., :unm_len, :], unm_token[..., unm_len:, :]
            n, _, c = unm.shape

            src = gather(dst, dim=-2, index=dst_idx_unm_token_list[level].expand(n, tr_unm_list[level], c))

            # Combine back to the original shape
            t = unm_len + tr_unm_list[level] + num_dst_unm_token_list[level]
            out_unm = torch.zeros(n, t, c, device=x.device, dtype=x.dtype)
            out_unm.scatter_(dim=-2, index=b_idx_unm_token_list[level].expand(n, num_dst_unm_token_list[level], c), src=dst)
            out_unm.scatter_(dim=-2, index=gather(a_idx_unm_token_list[level].expand(n, a_idx_unm_token_list[level].shape[1], 1), dim=1, index=unm_idx_unm_token_list[level]).expand(n, unm_len, c), src=unm)
            out_unm.scatter_(dim=-2, index=gather(a_idx_unm_token_list[level].expand(n, a_idx_unm_token_list[level].shape[1], 1), dim=1, index=src_idx_unm_token_list[level]).expand(n, tr_unm_list[level], c), src=src)
            out_unm = out_unm.reshape(b, -1, t * c)
            unm_src = out_unm[:, :unm_src_idx_block_list[level].shape[1], :]
            unm_dst = out_unm[:, unm_src_idx_block_list[level].shape[1]:, :]

            # # Token unmerge: Split unm and dst for merged tokens -> first time
            # unm_len = unm_idx_mrg_token_list[level+1].shape[1]
            # unm, dst = mrg_token[..., :unm_len, :], mrg_token[..., unm_len:, :]
            # n, _, c = unm.shape

            # src = gather(dst, dim=-2, index=dst_idx_mrg_token_list[level+1].expand(n, tr_mrg_list[level+1], c))

            # # Combine back to the original shape
            # out_mrg = torch.zeros(n, t, c, device=x.device, dtype=x.dtype)
            # # out_mrg = torch.zeros(b, H*W, c, device=x.device, dtype=x.dtype)
            # out_mrg.scatter_(dim=-2, index=b_idx_mrg_token_list[level+1].expand(n, num_dst_mrg_token_list[level+1], c), src=dst)
            # out_mrg.scatter_(dim=-2, index=gather(a_idx_mrg_token_list[level+1].expand(n, a_idx_mrg_token_list[level+1].shape[1], 1), dim=1, index=unm_idx_mrg_token_list[level+1]).expand(n, unm_len, c), src=unm)
            # out_mrg.scatter_(dim=-2, index=gather(a_idx_mrg_token_list[level+1].expand(n, a_idx_mrg_token_list[level+1].shape[1], 1), dim=1, index=src_idx_mrg_token_list[level+1]).expand(n, tr_mrg_list[level+1], c), src=src)
            
            # Token unmerge: Split unm and dst for merged tokens -> second time
            # mrg_token = out_mrg
            unm_len = unm_idx_mrg_token_list[level].shape[1]
            unm, dst = mrg_token[..., :unm_len, :], mrg_token[..., unm_len:, :]
            n, _, c = unm.shape

            src = gather(dst, dim=-2, index=dst_idx_mrg_token_list[level].expand(n, tr_mrg_list[level], c))

            # Combine back to the original shape
            out_mrg = torch.zeros(n, 2 * t, c, device=x.device, dtype=x.dtype)
            # out_mrg = torch.zeros(b, H*W, c, device=x.device, dtype=x.dtype)
            out_mrg.scatter_(dim=-2, index=b_idx_mrg_token_list[level].expand(n, num_dst_mrg_token_list[level], c), src=dst)
            out_mrg.scatter_(dim=-2, index=gather(a_idx_mrg_token_list[level].expand(n, a_idx_mrg_token_list[level].shape[1], 1), dim=1, index=unm_idx_mrg_token_list[level]).expand(n, unm_len, c), src=unm)
            out_mrg.scatter_(dim=-2, index=gather(a_idx_mrg_token_list[level].expand(n, a_idx_mrg_token_list[level].shape[1], 1), dim=1, index=src_idx_mrg_token_list[level]).expand(n, tr_mrg_list[level], c), src=src)
            
            # Frame unmerge
            out_mrg = out_mrg.reshape(b, -1, 2, t * c)
            mrg_src = out_mrg[:, :, 0, :]
            mrg_dst = out_mrg[:, :, 1, :]

            # Combine back to the original shape
            _, s1, c = unm_src.shape
            _, s2, c = mrg_src.shape
            _, s3, c = unm_dst.shape
            out = torch.zeros(b, s1 + s2 + s3, c, device=x.device, dtype=x.dtype)
            # mrg_dst = out_mrg.reshape(B, -1, c)
            unm_dst.scatter_(dim=-2, index=mrg_dst_idx_block_list[level].expand(mrg_dst.shape), src=mrg_dst)
            # mrg_src = gather(unm_dst, dim=-2, index=mrg_dst_idx_block.expand(B, fr, c))
            out.scatter_(dim=-2, index=b_idx_block_list[level].expand(unm_dst.shape), src=unm_dst)
            out.scatter_(dim=-2, index=gather(a_idx_block_list[level].expand(b, a_idx_block_list[level].shape[1], 1), dim=1, index=unm_src_idx_block_list[level]).expand(unm_src.shape), src=unm_src)
            out.scatter_(dim=-2, index=gather(a_idx_block_list[level].expand(b, a_idx_block_list[level].shape[1], 1), dim=1, index=mrg_src_idx_block_list[level]).expand(mrg_src.shape), src=mrg_src)
            x = out
            
        out = out.reshape(B, GT, GT, GF, H // GT, W // GT, T // GF, C).permute(0, 1, 4, 2, 5, 3, 6, 7)
        out = out.reshape(B*H*W, T, C)

        return out
    
    return merge, unmerge