import argparse
import time
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from tqdm import tqdm

from editflowsr.model.transformer_editflow import (
    TransformerModel,
    cubic_scheduler,
    cubic_scheduler_derivative,
)
from editflowsr.model.embedders import LinearPointEmbedder
from editflowsr.envs.environment import FunctionEnvironment
from editflowsr.metrics import compute_metrics


def load_env_from_checkpoint(checkpoint_path):
    data = torch.load(checkpoint_path, map_location="cpu")
    params = argparse.Namespace(**data["params"])
    env = FunctionEnvironment(params)
    checkpoint_vocab_size = data["params"].get("n_words")
    for key in ("embeddings.weight", "module.embeddings.weight"):
        if key in data.get("decoder", {}):
            checkpoint_vocab_size = int(data["decoder"][key].shape[0])
            break
    env_vocab_size = len(env.equation_words)

    if checkpoint_vocab_size == env_vocab_size + 1 and "<GAP>" not in env.equation_word2id:
        gap_id = len(env.equation_words)
        env.equation_words.append("<GAP>")
        env.equation_id2word[gap_id] = "<GAP>"
        env.equation_word2id["<GAP>"] = gap_id
    elif checkpoint_vocab_size not in (None, env_vocab_size, len(env.equation_words)):
        raise ValueError(
            f"Checkpoint decoder vocab size is {checkpoint_vocab_size}, "
            f"but the environment exposes {env_vocab_size} tokens. "
            "Only a single extra <GAP> token can be auto-added."
        )
    params.n_words = len(env.equation_id2word)
    return env, params, data


def load_models_from_checkpoint(data, env):
    params = argparse.Namespace(**data["params"])
    embedder = LinearPointEmbedder(params, env)
    encoder = TransformerModel(
        params, env.equation_id2word,
        is_encoder=True,
        use_prior_embeddings=True,
        positional_embeddings=params.enc_positional_embeddings,
    )
    decoder = TransformerModel(
        params, env.equation_id2word,
        is_encoder=False,
        use_prior_embeddings=False,
        positional_embeddings=params.dec_positional_embeddings,
    )

    normalized_states = {}
    for module_name in ("embedder", "encoder", "decoder"):
        normalized_state = {}
        for key, value in data[module_name].items():
            normalized_key = key.replace("module.", "")
            if normalized_key.startswith("glob_attn_"):
                normalized_key = normalized_key.replace("glob_attn_", "glob_attn_module.")
            elif normalized_key.startswith("bottleneck_"):
                normalized_key = normalized_key.replace("bottleneck_", "bottleneck_module.")
            normalized_state[normalized_key] = value
        normalized_states[module_name] = normalized_state

    embedder.load_state_dict(normalized_states["embedder"])
    encoder.load_state_dict(normalized_states["encoder"])
    decoder_state = normalized_states["decoder"]
    if "null_condition" not in decoder_state:
        decoder_state["null_condition"] = torch.zeros(
            decoder.null_condition.shape,
            dtype=decoder.null_condition.dtype,
        )
    decoder.load_state_dict(decoder_state)

    return embedder, encoder, decoder


def encode_input_values(embedder, x_to_fit, y_to_fit, env, device):
    max_in = int(getattr(env.params, "max_input_dimension", x_to_fit.shape[1]))
    max_out = int(getattr(env.params, "max_output_dimension", y_to_fit.shape[1]))

    if x_to_fit.shape[1] > max_in:
        raise ValueError(
            f"x_to_fit has {x_to_fit.shape[1]} dims but checkpoint expects <= {max_in} "
            f"(max_input_dimension). Please use matching data/checkpoint."
        )
    if y_to_fit.shape[1] > max_out:
        raise ValueError(
            f"y_to_fit has {y_to_fit.shape[1]} dims but checkpoint expects <= {max_out} "
            f"(max_output_dimension). Please use matching data/checkpoint."
        )

    pairs = []
    for x_row, y_row in zip(x_to_fit, y_to_fit):
        pairs.append(
            (
                np.asarray(x_row, dtype=np.float32),
                np.asarray(y_row, dtype=np.float32),
            )
        )
    x_sequences = [pairs]
    x_embedded, lengths = embedder(x_sequences)
    x_embedded = x_embedded.to(device)
    lengths = lengths.to(device)
    return x_embedded, lengths


def apply_ins_del_operations(x_t, ins_mask, del_mask, sub_mask, ins_tokens, sub_tokens, pad_token, max_seq_len=None):
    batch_size, seq_len = x_t.shape
    device = x_t.device

    x_t[sub_mask] = sub_tokens[sub_mask]
    replace_mask = ins_mask & del_mask
    x_t_modified = x_t.clone()
    x_t_modified[replace_mask] = ins_tokens[replace_mask]
    eff_ins_mask = ins_mask & ~replace_mask
    eff_del_mask = del_mask & ~replace_mask
    xt_pad_mask = (x_t == pad_token)
    xt_seq_lens = (~xt_pad_mask).sum(dim=1)
    new_lengths = xt_seq_lens + eff_ins_mask.sum(dim=1) - eff_del_mask.sum(dim=1)
    max_new_len = int(new_lengths.max().item())
    if max_new_len <= 0:
        return torch.full((batch_size, 1), pad_token, dtype=x_t.dtype, device=device)
    x_new = torch.full((batch_size, max_new_len), pad_token, dtype=x_t.dtype, device=device)
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
    pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    cum_del = torch.cumsum(eff_del_mask.float(), dim=1)
    cum_ins = torch.cumsum(eff_ins_mask.float(), dim=1)
    cum_ins_before = F.pad(cum_ins[:, :-1], (1, 0), value=0)
    new_pos = pos_idx + cum_ins_before - cum_del
    keep_mask = ~eff_del_mask & (new_pos >= 0) & (new_pos < max_new_len)
    if keep_mask.any():
        x_new[batch_idx.expand(-1, seq_len)[keep_mask], new_pos[keep_mask].long()] = x_t_modified[keep_mask]
    if eff_ins_mask.any():
        ins_pos = new_pos + 1
        ins_valid = eff_ins_mask & (ins_pos >= 0) & (ins_pos < max_new_len)
        if ins_valid.any():
            x_new[batch_idx.expand(-1, seq_len)[ins_valid], ins_pos[ins_valid].long()] = ins_tokens[ins_valid]
    if max_seq_len is not None and max_new_len > max_seq_len:
        return x_new[:, :max_seq_len]
    return x_new


def decode_prediction(token_ids, env):
    raw_token_ids = [int(tid) for tid in token_ids.detach().cpu().tolist()]
    raw_tokens = []
    for tid in raw_token_ids:
        word = env.equation_id2word.get(tid)
        if word is not None:
            raw_tokens.append(word)

    cleaned_tokens = []
    started = False
    stopped_at_eos = False
    for word in raw_tokens:
        if word in ("<PAD>", "<GAP>"):
            continue
        if word == "<EOS>":
            if not started:
                started = True
                continue
            stopped_at_eos = True
            break
        started = True
        cleaned_tokens.append(word)

    if not cleaned_tokens:
        cleaned_tokens = [
            word for word in raw_tokens if word not in ("<PAD>", "<GAP>", "<EOS>")
        ]

    token_info = {
        "raw_token_ids": raw_token_ids,
        "raw_tokens": raw_tokens,
        "cleaned_tokens": cleaned_tokens,
        "stopped_at_eos": stopped_at_eos,
    }
    if not cleaned_tokens:
        return {
            **token_info,
            "predicted_tree": None,
            "prefix": None,
            "infix": None,
            "decode_failed": True,
            "error": "No decodable tokens after cleaning.",
        }

    predicted_tree = env.word_to_infix(
        cleaned_tokens,
        is_float=False,
        str_array=False,
    )
    if predicted_tree is None:
        return {
            **token_info,
            "predicted_tree": None,
            "prefix": None,
            "infix": None,
            "decode_failed": True,
            "error": "equation_encoder.decode returned None.",
        }

    return {
        **token_info,
        "predicted_tree": predicted_tree,
        "prefix": predicted_tree.prefix(),
        "infix": predicted_tree.infix(),
        "decode_failed": False,
        "error": None,
    }


def safe_decode_ground_truth_tree(tree_prefix, env):
    if not tree_prefix:
        return None
    return env.word_to_infix(tree_prefix.split(","), is_float=False, str_array=False)


@lru_cache(maxsize=None)
def load_worker_env(checkpoint_path):
    env, _, _ = load_env_from_checkpoint(checkpoint_path)
    return env


def score_candidate(metrics):
    for key in ("r2", "r2_zero"):
        value = metrics.get(key)
        if value is not None and np.isfinite(value):
            return (2, float(value))
    value = metrics.get("_mse")
    if value is not None and np.isfinite(value):
        return (1, -float(value))
    value = metrics.get("_rmse")
    if value is not None and np.isfinite(value):
        return (0, -float(value))
    return (-1, -float("inf"))


def evaluate_tree_candidate(
    name,
    tree,
    x_to_fit,
    y_to_fit,
    env,
    metrics,
    true_tree=None,
    error=None,
    status="ok",
):
    candidate = {
        "name": name,
        "status": status,
        "error": error,
        "tree": tree,
        "prefix": tree.prefix() if tree is not None else None,
        "infix": tree.infix() if tree is not None else None,
        "metrics": {},
        "y_pred": None,
    }

    if tree is None:
        return candidate

    numexpr_fn = env.simplifier.tree_to_numexpr_fn(tree)
    y_pred = numexpr_fn(np.asarray(x_to_fit))
    metric_inputs = {
        "true": [np.asarray(y_to_fit)],
        "predicted": [y_pred],
        "predicted_tree": [tree],
    }
    if true_tree is not None:
        metric_inputs["tree"] = [true_tree]
    flat_metrics = {}
    for key, values in compute_metrics(metric_inputs, metrics=metrics).items():
        if len(values) == 0:
            flat_metrics[key] = float("nan")
            continue
        value = values[0]
        if isinstance(value, np.generic):
            value = value.item()
        flat_metrics[key] = value
    candidate["metrics"] = flat_metrics
    candidate["y_pred"] = y_pred
    return candidate


def refine_tree_with_bfgs(tree, x_to_fit, y_to_fit, env, stop_after=30.0, downsample=1024):
    if tree is None:
        return None, "No tree to refine."

    skeleton_tree, candidate_constants = env.generator.function_to_skeleton(
        tree,
        constants_with_idx=True,
    )

    if len(candidate_constants) == 0:
        return None, "Tree has no constants to refine."

    coeffs0 = candidate_constants
    if "CONSTANT" in candidate_constants:
        coeffs0 = np.random.randn(len(candidate_constants))
        tree = env.wrap_equation_floats(skeleton_tree, coeffs0)
        skeleton_tree, coeffs0 = env.generator.function_to_skeleton(
            tree,
            constants_with_idx=True,
        )

    # BFGS constant fitting fails on degenerate expressions (sympytorch raising
    # KeyError on ComplexInfinity division by zero, wall-clock "Time is over."
    # ValueError, nan, etc.). On failure return None, and the caller falls back
    # to the original predicted_tree for scoring.
    try:
        objective_module = env.simplifier.tree_to_torch_module(skeleton_tree, dtype=torch.float64)
        features = np.asarray(x_to_fit)
        targets = np.asarray(y_to_fit)
        if downsample > 0:
            features = features[:downsample]
            targets = targets[:downsample]
        features = torch.tensor(features, dtype=torch.float64, requires_grad=False)
        targets = torch.tensor(targets, dtype=torch.float64, requires_grad=False)
        started = None
        best_fun_value = np.inf
        best_x = None

        def objective_torch(coeffs):
            if not isinstance(coeffs, torch.Tensor):
                coeffs = torch.tensor(coeffs, dtype=torch.float64, requires_grad=True)
            y_tilde = objective_module(features, coeffs)
            if y_tilde is None:
                return None
            return (targets.reshape(-1) - y_tilde.reshape(-1)).pow(2).mean().div(2)

        def objective_numpy(coeffs):
            nonlocal started, best_fun_value, best_x
            if started is None:
                started = time.time()
            elif abs(time.time() - started) >= stop_after:
                raise ValueError("Time is over.")
            fun_value = objective_torch(coeffs).item()
            if best_x is None or fun_value < best_fun_value:
                best_fun_value = fun_value
                best_x = coeffs
            return fun_value

        def gradient_numpy(coeffs):
            if not isinstance(coeffs, torch.Tensor):
                coeffs = torch.tensor(coeffs, dtype=torch.float64, requires_grad=True)
            grad_obj = torch.func.grad(objective_torch)(coeffs)
            return grad_obj.detach().numpy()

        minimize(objective_numpy, coeffs0, method="BFGS", jac=gradient_numpy, options={"disp": False})
        refined_tree = env.wrap_equation_floats(skeleton_tree, best_x)
        return refined_tree, None
    except Exception as e:
        return None, f"BFGS refine failed: {type(e).__name__}: {e}"


def sample_with_euler(
    embedder,
    encoder,
    decoder,
    x_to_fit,
    y_to_fit,
    env,
    params,
    scheduler_a=2.0,
    scheduler_b=0.5,
    n_steps=100,
    sample_idx=0,
    device="cuda",
    beam_size=1,
):
    batch_size = beam_size
    x_embedded, lengths = encode_input_values(embedder, x_to_fit, y_to_fit, env, device)
    encoded = encoder("fwd", x=x_embedded, lengths=lengths, causal=False)
    if batch_size > 1:
        encoded = encoded.repeat(batch_size, 1)

    vocab_size = len(env.equation_id2word)
    pad_token = env.equation_word2id["<PAD>"]
    eos_token = env.equation_word2id["<EOS>"]

    min_len = getattr(params, 'coupling_min_len', 10)
    max_len = getattr(params, 'coupling_max_len', 40)
    content_lengths = torch.randint(min_len, max_len + 1, (batch_size,), device=device)
    max_content_len = int(content_lengths.max().item())
    x_0 = torch.full((batch_size, max_content_len + 1), pad_token, dtype=torch.long, device=device)
    x_0[:, 0] = eos_token
    for i in range(batch_size):
        current_len = int(content_lengths[i].item())
        x_0[i, 1:1 + current_len] = torch.randint(
            0,
            vocab_size,
            size=(current_len,),
            dtype=torch.long,
            device=device,
        )

    x_t = x_0.clone()
    t = torch.zeros(batch_size, 1, device=device)

    print(f"\n=== Starting inference (sample {sample_idx}, {n_steps} steps, beam_size={beam_size}) ===")
    print(f"Input: x shape={x_to_fit.shape}, y shape={y_to_fit.shape}")

    if lengths.numel() > 0:
        print(f"embedder lengths[0]={int(lengths[0])} (expected {x_to_fit.shape[0]})")

    expressions = []

    h = 1.0 / n_steps
    with tqdm(total=n_steps, desc="EditFlow inference") as pbar:
        for _ in range(n_steps):
            if t.max() >= 1 - h:
                break

            x_pad_mask = (x_t == pad_token)
            lengths_t = (~x_pad_mask).sum(dim=1).long()
            _, ins_scores, sub_scores, raw_rate_logits = decoder(
                "fwd",
                x=x_t.t(),
                lengths=lengths_t,
                causal=False,
                src_enc=encoded,
                time_step=t,
                return_raw_rate_logits=True,
            )
            rates = F.softplus(raw_rate_logits)
            lambda_ins = rates[:, :, 0]
            lambda_sub = rates[:, :, 1]
            lambda_del = rates[:, :, 2]
            h_adapt = torch.minimum(
                h * torch.ones_like(t),
                (1 - cubic_scheduler(t, a=scheduler_a, b=scheduler_b))
                / cubic_scheduler_derivative(t, a=scheduler_a, b=scheduler_b),
            )
            ins_mask = torch.rand(lambda_ins.shape, device=device) < (1 - torch.exp(-h_adapt * lambda_ins))
            del_sub_mask = torch.rand(lambda_sub.shape, device=device) < (1 - torch.exp(-h_adapt * (lambda_sub + lambda_del)))
            prob_del = torch.where(del_sub_mask, lambda_del / (lambda_sub + lambda_del + 1e-10),
                                  torch.zeros_like(lambda_del))
            del_mask = torch.bernoulli(prob_del).bool()
            sub_mask = del_sub_mask & ~del_mask
            ins_tokens = torch.full(ins_scores.shape[:2], pad_token, dtype=torch.long, device=device)
            sub_tokens = torch.full(sub_scores.shape[:2], pad_token, dtype=torch.long, device=device)
            non_pad_mask = ~x_pad_mask
            if non_pad_mask.any():
                ins_probs = sampling_probs_from_scores(ins_scores[non_pad_mask], eos_token)
                sub_probs = sampling_probs_from_scores(sub_scores[non_pad_mask], eos_token)
                ins_sampled = torch.multinomial(ins_probs, num_samples=1, replacement=True).squeeze(-1)
                sub_sampled = torch.multinomial(sub_probs, num_samples=1, replacement=True).squeeze(-1)
                ins_tokens[non_pad_mask] = ins_sampled
                sub_tokens[non_pad_mask] = sub_sampled
            x_t = apply_ins_del_operations(
                x_t, ins_mask, del_mask, sub_mask, ins_tokens, sub_tokens,
                pad_token, max_seq_len=512
            )
            decoded = decode_prediction(x_t[0], env)
            if decoded["predicted_tree"] is not None:
                current_expr = decoded["prefix"]
            elif decoded["cleaned_tokens"]:
                current_expr = ",".join(decoded["cleaned_tokens"])
            else:
                current_expr = ",".join(decoded["raw_tokens"])
            expressions.append(current_expr)
            pbar.set_postfix({"expr": current_expr[:50] if len(current_expr) > 50 else current_expr})
            t = t + h_adapt
            pbar.update(1)
    return x_t, expressions


def sampling_probs_from_scores(scores, fallback_token):
    probs = F.softmax(scores.float(), dim=-1)
    finite_mask = torch.isfinite(probs).all(dim=-1)
    positive_mask = probs.sum(dim=-1) > 0
    valid_mask = finite_mask & positive_mask
    if valid_mask.all():
        return probs

    safe_probs = probs.clone()
    invalid_rows = ~valid_mask
    safe_probs[invalid_rows] = 0
    safe_probs[invalid_rows, fallback_token] = 1.0
    return safe_probs
