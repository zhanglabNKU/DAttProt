import math
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def setup_seed(seed=1024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _get_clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class PosEmbedding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros((max_len, d_model), requires_grad=False)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0., d_model, 2) * (math.log(1e5) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TransformerEncoderLayer(nn.Module):
    """An alternative to nn.TransformerEncoderLayer if torch version < 1.2.0."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation='relu'):
        super().__init__()
        self.d_model = d_model  # F
        self.dropout = nn.Dropout(p=dropout)
        self.linears = _get_clones(nn.Linear(self.d_model, self.d_model), 4)
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, dim_feedforward),
            _get_activation_fn(activation), self.dropout,
            nn.Linear(dim_feedforward, self.d_model)
        )
        self.norm = _get_clones(nn.LayerNorm(self.d_model, eps=1e-9), 2)
        self.h = nhead  # H

    def multi_head_att(self, feats, mask):
        """
        :param feats: (B, L, F)
        :param mask: (B, L)
        :return: (B, L, F)
        """
        d = self.d_model // self.h
        # (B, H, L, F//H)
        query, key, value = [m(feats).view(feats.size(0), -1, self.h, d).transpose(1, 2)
                             for m in self.linears[:3]]
        # (B, H, L, L)
        scores = (d ** -0.5) * query @ key.transpose(-2, -1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1), -1e9)
        p_attn = F.softmax(scores, dim=-1)
        # (B, H, L, F//H)
        feats_ = self.dropout(p_attn) @ value
        feats_ = feats_.transpose(1, 2).contiguous().view(feats.size())
        # (B, L, F)
        feats = self.linears[-1](feats_) + feats
        return self.norm[0](feats)

    def forward(self, feats, src_key_padding_mask):
        feats = self.multi_head_att(feats, src_key_padding_mask)
        feats = feats + self.ff(feats)
        return self.norm[1](feats)


class TransformerSequenceEncoder(nn.Module):
    """Transformer encoder block"""

    def __init__(self, feature_dim, nhead, nlayer, dropout):
        super().__init__()
        import re
        ver = re.search(r"([0-9]+).([0-9]+)[^0-9]?.", torch.__version__)
        torch_version = [int(ver.group(1)), int(ver.group(2))]
        # if torch < 1.2.0, nn.TransformerEncoderLayer is not implemented,
        # then define and deploy TransformerEncoderLayer
        self.new_ver = torch_version[0] > 1 or (torch_version[0] == 1 and torch_version[1] >= 2)
        encoder_mod = nn.TransformerEncoderLayer if self.new_ver else TransformerEncoderLayer
        self.seq_embed = _get_clones(
            encoder_mod(feature_dim, nhead, feature_dim * 4, dropout=dropout, activation='gelu'), nlayer)

    def __len__(self):
        return len(self.seq_embed)

    def hidden_feats(self, x, mask=None):
        """
        Get outputs of all hidden layers
        :param x: (B, L, F)
        :param mask: (B, L)
        :return: [N_layer * (B, L, F)]
        """
        if self.new_ver:
            # (L, B, F)
            x = x.transpose(0, 1)

        hidden = []

        for mod in self.seq_embed:
            # (B, L, F) if self.new_ver else (L, B, F)
            x = mod(x, src_key_padding_mask=mask)
            hidden.append(x)

        if self.new_ver:
            hidden = [feat.transpose(0, 1).contiguous() for feat in hidden]
        return hidden

    def forward(self, x, mask=None, trn_ly=None):
        """
        :param x: (B, L, F)
        :param mask: (B, L)
        :param trn_ly:
            None: full layers
            int > 0: front layers
        :return: (B, L, F)
        """
        if self.new_ver:
            # (L, B, F)
            x = x.transpose(0, 1)

        if trn_ly is None:
            trn_ly = len(self.seq_embed)

        for mod in self.seq_embed[:trn_ly]:
            # (B, L, F) if self.new_ver else (L, B, F)
            x = mod(x, src_key_padding_mask=mask)

        if self.new_ver:
            # (B, L, F)
            x = x.transpose(0, 1).contiguous()

        if mask is not None:
            x[mask] = 0
        return x


def feat_agreement(feats, mask, feats_center=None):
    """
    double Attention block based on feature agreement algorithm
    :param feats: (B, N, L, F)
    :param feats_center: (B, 1, L, F)
    :param mask: (B, L)
    :return:
        feats: (B, L, F)
        pos_probs: (B, 1, L)
        scale_probs: (B, N, L)
    """
    if feats_center is None:
        # (B, 1, L, F)
        feats_center = feats.mean(1, keepdim=True)
    # (B, N, L)
    double_scores = (feats * feats_center).sum(-1) * feats.size(-1) ** -0.5
    double_scores = double_scores.masked_fill(mask.unsqueeze(1), -1e9)

    # Matrix {p(scale=i, timestep=j)}: (B, N, L)
    double_probs = F.softmax(double_scores.view(feats.size(0), -1), -1) \
        .view(double_scores.size())

    # Vector {p(timestep=j)}: (B, 1, L)
    pos_probs = torch.sum(double_probs, dim=1, keepdim=True)

    # Matrix {p(scale=i | timestep=j)}: (B, N, L)
    scale_probs = double_probs / (pos_probs + 1e-9)

    # (B, N, L, F)
    feats = feats * scale_probs.unsqueeze(-1)
    # (B, L, F)
    feats = feats.sum(1)
    return feats, pos_probs, scale_probs


class DAttProt(nn.Module):
    def __init__(self, seq_len, embedding_dim, feature_dim, vocab_size, encoding_dim=None,
                 key_padding_mask=True, nhead=8, nlayer=3, nclass=2, dropout=0.25):
        super().__init__()
        self.d_embed = embedding_dim  # F_embed
        self.d_feat = feature_dim  # F
        self.d_encode = self.d_feat // self.h if encoding_dim is None else encoding_dim  # F_out
        self.embedding = nn.Embedding(vocab_size + 1, self.d_embed, padding_idx=0)
        self.map = nn.Linear(self.d_embed, self.d_feat)
        # [0] for zero padding; [1:-1] for vocabulary: [-1] for MASK

        self.mask_flag = key_padding_mask
        self.l = seq_len  # L
        self.M = nclass  # M
        self.h = nhead  # H
        self.pos_embedding = PosEmbedding(self.d_embed)
        self.dropout = nn.Dropout(p=dropout)
        self.coder = TransformerSequenceEncoder(self.d_feat, self.h, nlayer, dropout)
        self.dim_reduction = nn.Linear(self.d_feat, self.d_encode)

    def _init_paras(self, module=None):
        if module is not None:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        elif isinstance(module, nn.Parameter):
            nn.init.xavier_uniform_(module)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def feat_embedding(self, x):
        # sequence and position embedding
        # (B, L, F_embed)
        seq_embed = self.embedding(x) * self.d_embed
        pos_embed = self.pos_embedding(x)
        # (B, L, F)
        embed = self.map(self.dropout(seq_embed + pos_embed))
        # (B, L)
        mask = x == 0
        return embed.masked_fill(mask.unsqueeze(-1), 0), mask

    def base_dict(self):
        base = {
            'embed': self.embedding.state_dict(),
            'pos_embed': self.pos_embedding.state_dict(),
            'coder': self.coder.state_dict(),
            'reduce': self.dim_reduction.state_dict()
        }
        return base

    def load_base_dict(self, base):
        self.embedding.load_state_dict(base['embed'])
        self.pos_embedding.load_state_dict(base['pos_embed'])
        self.coder.load_state_dict(base['coder'])
        self.dim_reduction.load_state_dict(base['reduce'])


class DAttProt_cls(DAttProt):
    """DAttProt for classification"""

    def __init__(self, scales, train_layer=None, **kwargs):
        super().__init__(**kwargs)
        if train_layer is None:
            self.trn_ly = len(self.coder)
        else:
            self.trn_ly = min(train_layer, len(self.coder))

        self.scales = tuple(scales)
        self.convs = nn.ModuleList()
        for scale in self.scales:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(self.d_feat, self.d_encode * scale, kernel_size=scale, padding=scale // 2),
                    nn.ReLU(),
                    nn.Conv1d(self.d_encode * scale, self.d_encode, 1)
                )
            )
        self.norm = nn.LayerNorm(self.d_encode, eps=1e-9)

        self.classifier = nn.Sequential(
            # nn.BatchNorm1d(self.d_encode),
            nn.BatchNorm1d(self.l),
            self.dropout,
            nn.Linear(self.d_encode, self.M)
        )
        self._init_paras()

    def forward(self, x):
        """
        :param x: (B, L)
        :return: (B, M)
        """
        # embed: (B, L, F)
        # mask: (B, L)
        embed, mask = self.feat_embedding(x)
        # (B, L, F)
        feats = self.coder(embed, mask if self.mask_flag else None, self.trn_ly)

        # (B, L, F_out)
        feats_local = self.norm(self.dim_reduction(feats))
        # [N * (B, L, F_out)]
        feats_multi = []
        for conv in self.convs:
            # (B, F_out, L)
            feat = conv(feats.transpose(-2, -1).contiguous())[:, :, :self.l]
            # (B, L, F_out)
            feats_multi.append(feat.transpose(-2, -1).contiguous())
        # (B, N, L, F_out)
        feats_multi = self.norm(torch.stack(feats_multi, dim=1))
        # (B, 1, L, F_out)
        feats_center = (feats_multi.sum(dim=1) + feats_local).unsqueeze(1) / (len(self.scales) + 1)

        # feats: (B, L, F_out)
        # pos_probs: (B, 1, L)
        feats, pos_probs, _ = feat_agreement(feats_multi, mask, feats_center)

        # (B, M, L)
        probs = self.classifier(feats).transpose(-2, -1)
        # (B, M, L)
        probs = probs * pos_probs
        # (B, M)
        probs = probs.sum(-1)
        return probs


class DAttProt_lm(DAttProt):
    """DAttProt for language model pre-training"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.word_classes = kwargs["vocab_size"]  # V
        self.to_word = nn.Linear(self.d_encode, self.word_classes)
        self._init_paras()

    def pretrain(self, x, pred_pos):
        """
        :param x: (B, L)
        :param pred_pos: (N_mask)
        :return: (N_mask, V)
        """
        # embed: (B, L, F)
        # mask: (B, L)
        embed, mask = self.feat_embedding(x)
        # (B, L, F)
        feats = self.coder(embed, mask if self.mask_flag else None)
        # feats: (B, L, F_out)
        feats = self.dim_reduction(feats)
        # (N_mask, F_out)
        feats = feats.view(-1, self.d_encode)[pred_pos]
        # (N_mask, V)
        preds = self.to_word(feats)
        return preds


class RandMaskBuffer:
    """A buffer for random mask module"""

    def __init__(self, buffer_size, min_ix, mask_ix, mask_probs=(0.8, 0.1, 0.1)):
        self.buffer_size = buffer_size
        # normal tokens: [min_ix, mask_ix)
        # MASK token: mask_ix
        self.min_ix = min_ix
        self.mask_ix = mask_ix
        # mask_probs: [
        #     Prob(change token to MASK),
        #     Prob(randomly switch to another token),
        #     Prob(retain original token)
        # ]
        self.mask_prob = mask_probs[0]
        # rand ixs buffer may contain identical ix with prob = 1 / vocab_size
        self.switch_prob = self.mask_prob + mask_probs[1] / (1 - 1 / (mask_ix - min_ix))
        self.unused_addr = 0
        self.__ix_buffer = None
        self.__mask_buffer = None
        self.generate()

    def generate(self):
        rand_buffer = torch.rand(self.buffer_size)
        # If mask, ix = MASK. If switch, ix = rand(different ix). If retain, ix = 0
        ix_buffer = torch.randint(self.min_ix, self.mask_ix, [self.buffer_size])
        ix_buffer = ix_buffer.masked_fill(rand_buffer < self.mask_prob, self.mask_ix)
        self.__ix_buffer = ix_buffer.masked_fill(rand_buffer >= self.switch_prob, 0)
        # If mask or switch, mask = 0. If retain, mask = 1
        self.__mask_buffer = (rand_buffer >= self.switch_prob).long()
        # For each token x, employ transformation: x_mask = x * mask + ix

    def get(self, l):
        # if the pointer reach the end of the buffer queue, generate new random numbers
        if self.unused_addr + l > self.buffer_size:
            self.generate()
            self.unused_addr = 0
        masks = self.__mask_buffer[self.unused_addr: self.unused_addr + l]
        ixs = self.__ix_buffer[self.unused_addr: self.unused_addr + l]
        self.unused_addr += l
        return masks, ixs


class RandMask(nn.Module):
    """Random mask module for pre-training"""

    def __init__(self, vocab_size, pred_rate=0.15, buffer_size=100000):
        super().__init__()
        assert 0 < pred_rate < 1
        self.buffer = RandMaskBuffer(buffer_size, 1, vocab_size)
        self.pred_rate = pred_rate  # p

    def forward(self, x, lens):
        """
        :param x: input sequences with paddings (B, L)
        :param lens: a list of each sequence's real length
        :return:
            masked_x: masked_input (B, L)
            pred_pos: 1D masked positions (N_mask)
            pred_ixs: 1D masked tokens (N_mask)
        """
        pred_pos = []
        batch_sz, seq_len = x.size()
        for i in range(batch_sz):
            real_len = lens[i].item()  # L_real
            # (L_real * p,)
            input_pred_pos = torch.LongTensor(random.sample(range(real_len), k=int(real_len * self.pred_rate)))
            pred_pos.append(input_pred_pos + seq_len * i)
        # (N_mask,)
        pred_pos = torch.cat(pred_pos, dim=-1)
        # flatten x into 1D tensor (B * L,)
        x = x.view(-1)
        # (N_mask,)
        pred_ixs = x[pred_pos] - 1
        mask, ixs = self.buffer.get(pred_pos.size(0))
        x[pred_pos] = x[pred_pos] * mask + ixs
        return x.view(batch_sz, -1), pred_pos, pred_ixs
