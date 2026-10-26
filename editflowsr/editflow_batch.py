import numpy as np
import torch


def sample_original_editflow_prior(
    x_1: torch.Tensor,
    pad_token: int,
    bos_token: int,
    vocab_size: int,
    rng=None,
):
    batch_size, seq_len = x_1.shape
    device = x_1.device
    np_rng = rng if rng is not None else np.random

    x_0_list = []
    for batch_idx in range(batch_size):
        x1_len = int((x_1[batch_idx] != pad_token).sum().item())
        rand_scale = float(np_rng.uniform(0.5, 1.5))
        rand_len = max(5, min(int(rand_scale * x1_len), seq_len))
        x0 = torch.randint(1, vocab_size, (rand_len,), dtype=x_1.dtype, device=device)
        x0[0] = bos_token
        if rand_len < seq_len:
            x0 = F.pad(x0, (0, seq_len - rand_len), value=pad_token)
        x_0_list.append(x0[:seq_len])

    return torch.stack(x_0_list, dim=0)


def prepare_editflow_alignment(
    x_1: torch.Tensor,
    pad_token: int,
    gap_token: int,
    bos_token: int,
    vocab_size: int,
    rng=None,
):
    x_0 = sample_original_editflow_prior(
        x_1,
        pad_token=pad_token,
        bos_token=bos_token,
        vocab_size=vocab_size,
        rng=rng,
    )
    z_0, z_1 = align_xs_to_zs(x_0, x_1, pad_token=pad_token, gap_token=gap_token)
    return x_0, z_0, z_1
