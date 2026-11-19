from logging import getLogger
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


N_MAX_POSITIONS = 4096


logger = getLogger()


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(n_pos)
        ]
    )
    out.detach_()
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))


def get_masks(slen, lengths, causal):
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]


    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask


    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, dim, src_dim, dropout, normalized_attention):
        super().__init__()
        self.dim = dim
        self.src_dim = src_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.normalized_attention = normalized_attention
        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(src_dim, dim)
        self.v_lin = nn.Linear(src_dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        if normalized_attention:
            self.attention_scale = nn.Parameter(
                torch.tensor(1.0 / math.sqrt(dim // n_heads))
            )

    def forward(self, input, mask=None, kv=None):
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim, "Dimensions do not match: %s input vs %s configured" % (
            dim,
            self.dim,
        )
        n_heads = self.n_heads
        dim_per_head = dim // n_heads

        def shape(x):
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(input))
        if kv is None:
            k = shape(self.k_lin(input))
            v = shape(self.v_lin(input))
        else:
            k = shape(self.k_lin(kv))
            v = shape(self.v_lin(kv))

        if self.normalized_attention:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            q = q * self.attention_scale
        else:
            q = q / math.sqrt(dim_per_head)

        scores = torch.matmul(q, k.transpose(2, 3))

        if mask is not None:
            mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)
            mask = (
                (mask == 0).view(mask_reshape).expand_as(scores)
            )
            scores.masked_fill_(mask, -float("inf"))

        weights = F.softmax(scores.float(), dim=-1).type_as(
            scores
        )
        weights = F.dropout(
            weights, p=self.dropout, training=self.training
        )
        context = torch.matmul(weights, v)
        context = unshape(context)

        return self.out_lin(context)

    def single_kv_forward(self, input, kv):
        bs, qlen, dim = input.size()
        assert dim == self.dim, "Dimensions do not match: %s input vs %s configured" % (
            dim,
            self.dim,
        )
        dim_per_head = dim // self.n_heads

        v = self.v_lin(kv).view(bs, 1, self.n_heads, dim_per_head).transpose(1, 2)
        weights = input.new_ones((bs, self.n_heads, qlen, 1))
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        context = weights * v.expand(-1, -1, qlen, -1)
        context = context.transpose(1, 2).contiguous().view(bs, qlen, self.n_heads * dim_per_head)

        return self.out_lin(context)


class TransformerFFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, hidden_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.midlin = nn.ModuleList()
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        for i in range(1, self.hidden_layers):
            self.midlin.append(nn.Linear(dim_hidden, dim_hidden))
        self.lin2 = nn.Linear(dim_hidden, out_dim)

    def forward(self, input):
        x = self.lin1(input)
        x = F.relu(x)
        for mlin in self.midlin:
            x = mlin(x)
            x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class BaseBottleneck(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(BaseBottleneck, self).__init__()

        self.fc1 = nn.Linear(input_dim, bottleneck_dim)

    def forward(self, h):

        z_rep = self.fc1(h)

        return z_rep


class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        device = t.device
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=t.dtype) * -emb)
        emb = t * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.hidden_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb


def cubic_scheduler(t, a=2.0, b=0.5):
    return (
        -2 * (t ** 3)
        + 3 * (t ** 2)
        + a * (t ** 3 - 2 * t ** 2 + t)
        + b * (t ** 3 - t ** 2)
    )


def cubic_scheduler_derivative(t, a=2.0, b=0.5):
    return (
        -6 * (t ** 2)
        + 6 * t
        + a * (3 * t ** 2 - 4 * t + 1)
        + b * (3 * t ** 2 - 2 * t)
    )


class TransformerModel(nn.Module):


    def __init__(
        self,
        params,
        id2word,
        is_encoder,
        use_prior_embeddings,
        positional_embeddings,
    ):
        super().__init__()


        self.dtype = torch.half if params.fp16 else torch.float
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder

        self.id2word = id2word
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.pad_index = self.word2id["<PAD>"]

        self.n_words = len(self.id2word)
        assert len(self.id2word) == self.n_words


        self.latent_dim = params.latent_dim
        self.dim = (
            params.enc_emb_dim if is_encoder else params.dec_emb_dim
        )
        self.src_dim = params.enc_emb_dim
        self.hidden_dim = self.dim * 4
        self.n_hidden_layers = (
            params.n_enc_hidden_layers if is_encoder else params.n_dec_hidden_layers
        )
        self.n_heads = (
            params.n_enc_heads if is_encoder else params.n_dec_heads
        )
        self.n_layers = params.n_enc_layers if is_encoder else params.n_dec_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        self.norm_attention = params.norm_attention
        assert (
            self.dim % self.n_heads == 0
        ), "transformer dim must be a multiple of n_heads"


        if positional_embeddings is None or positional_embeddings in ("none", "alibi"):
            self.position_embeddings = None
        elif positional_embeddings == "sinusoidal":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
            create_sinusoidal_embeddings(
                N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight
            )
        elif positional_embeddings == "learnable":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        else:
            raise NotImplementedError

        self.use_prior_embeddings = use_prior_embeddings
        if not use_prior_embeddings:
            self.embeddings = Embedding(
                self.n_words, self.dim, padding_idx=self.pad_index
            )
        else:
            self.embeddings = None
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)


        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        if self.is_decoder:
            self.layer_norm15 = nn.ModuleList()
            self.encoder_attn = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attentions.append(
                MultiHeadAttention(
                    self.n_heads,
                    self.dim,
                    self.dim,
                    dropout=self.attention_dropout,
                    normalized_attention=self.norm_attention,
                )
            )
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-12))
                self.encoder_attn.append(
                    MultiHeadAttention(
                        self.n_heads,
                        self.dim,
                        self.src_dim,
                        dropout=self.attention_dropout,
                        normalized_attention=self.norm_attention,
                    )
                )
            self.ffns.append(
                TransformerFFN(
                    self.dim,
                    self.hidden_dim,
                    self.dim,
                    self.n_hidden_layers,
                    dropout=self.dropout,
                )
            )
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))


        if self.is_decoder:
            self.time_embedding = nn.Sequential(
                SinusoidalTimeEmbedding(hidden_dim=self.dim),
                nn.Linear(self.dim, self.dim),
                nn.SiLU(),
                nn.Linear(self.dim, self.dim)
            )
            self.null_condition = nn.Parameter(torch.zeros(self.latent_dim))


            self.rates_out = nn.Sequential(
                nn.Linear(self.dim * 2, self.dim),
                nn.SiLU(),
                nn.Linear(self.dim, 3),
            )
            self.ins_logits_out = nn.Sequential(
                nn.Linear(self.dim * 2, self.dim),
                nn.SiLU(),
                nn.Linear(self.dim, self.n_words),
            )
            self.sub_logits_out = nn.Sequential(
                nn.Linear(self.dim * 2, self.dim),
                nn.SiLU(),
                nn.Linear(self.dim, self.n_words),
            )


            for module in [self.rates_out, self.ins_logits_out, self.sub_logits_out]:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.1)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)


        self.glob_attn_module = None
        self.bottleneck_module = None
        self.z_projection = None

        if self.is_encoder:
            self.glob_attn_module = nn.Sequential(
                nn.Linear(self.dim, 1), nn.Softmax(dim=1)
            )
            self.bottleneck_module = BaseBottleneck(self.dim, self.latent_dim)

        if self.is_decoder:
            self.z_projection = nn.Linear(self.latent_dim, self.dim)


        self.z_rep = None


    def forward(self, mode, **kwargs):
        if mode == "fwd":
            return self.fwd(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    @staticmethod
    def _apply_editflow_head(head, tensor, time_emb):
        first_linear = head[0]
        activation = head[1]
        second_linear = head[2]

        split = tensor.size(-1)
        hidden = F.linear(tensor, first_linear.weight[:, :split], first_linear.bias)
        hidden = hidden + F.linear(time_emb.unsqueeze(1), first_linear.weight[:, split:], None)
        hidden = activation(hidden)
        return second_linear(hidden)

    def get_null_condition(self, batch_size):
        return self.null_condition.unsqueeze(0).expand(batch_size, -1)

    def fwd(
        self,
        x,
        lengths,
        causal,
        src_enc=None,
        positions=None,
        time_step=None,
        return_raw_rate_logits=False,
    ):
        slen, bs = x.size()[:2]
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)
        if self.is_decoder:
            assert src_enc is not None
            assert src_enc.size(0) == bs
        elif src_enc is not None:
            raise ValueError("src_enc is only supported in decoder mode.")


        mask, attn_mask = get_masks(slen, lengths, causal)

        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)



        if not self.use_prior_embeddings:
            tensor = self.embeddings(x)
        else:
            tensor = x

        if self.position_embeddings is not None:
            tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        mask_expanded = mask.unsqueeze(-1).to(tensor.dtype)
        tensor *= mask_expanded

        src_enc_proj = None
        if self.is_decoder:

            src_enc_proj = self.z_projection(src_enc).unsqueeze(1)


        for i in range(self.n_layers):


            attn = self.attentions[i](tensor, attn_mask)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)


            if self.is_decoder:


                attn = self.encoder_attn[i].single_kv_forward(tensor, src_enc_proj)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)


            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)

            tensor *= mask_expanded


        if self.is_encoder:
            glob_attn = self.glob_attn_module(tensor)
            z_rep = torch.bmm(glob_attn.transpose(-1, 1), tensor).squeeze()


            if len(tensor) == 1:
                z_rep = z_rep.unsqueeze(0)

            z_rep = self.bottleneck_module(z_rep)
            return z_rep


        if self.is_decoder:


            time_emb = self.time_embedding(time_step)


            raw_rate_logits = self._apply_editflow_head(self.rates_out, tensor, time_emb)
            rates = F.softplus(raw_rate_logits)
            ins_logits = self._apply_editflow_head(self.ins_logits_out, tensor, time_emb)
            sub_logits = self._apply_editflow_head(self.sub_logits_out, tensor, time_emb)

            if return_raw_rate_logits:
                return rates, ins_logits, sub_logits, raw_rate_logits
            return rates, ins_logits, sub_logits
