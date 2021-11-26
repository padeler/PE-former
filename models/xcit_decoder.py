# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implementation of Cross-Covariance Image Transformer (XCiT)
Based on timm and DeiT code bases
https://github.com/rwightman/pytorch-image-models/tree/master/timm
https://github.com/facebookresearch/deit/
"""
import math

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import _cfg, Mlp
from timm.models.registry import register_model
from timm.models.layers import DropPath, drop, trunc_normal_, to_2tuple

import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Callable, List, Optional, Tuple
from torch import nn, Tensor
from models.xcit import LPI
from models.transformer import _get_activation_fn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_

class DecXCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qk_scale=None, attn_drop=0., proj_drop=0., qkv_bias=True, Nq=100, Nt=197):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.proj = nn.Linear(dim, dim)
        self.qW = nn.Linear(dim, dim, bias=qkv_bias)
        self.kW = nn.Linear(dim, dim, bias=qkv_bias)
        self.vW = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_mapper = nn.Parameter(torch.ones(1, 1, Nt, Nq))
        self.x_mapper = nn.Parameter(torch.ones(1, Nq, Nt))
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qW.weight)
        nn.init.xavier_uniform_(self.kW.weight)
        nn.init.xavier_uniform_(self.vW.weight)
        self.qW.bias.data.fill_(0)
        self.kW.bias.data.fill_(0)
        self.vW.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.proj.weight)
        self.proj.bias.data.fill_(0)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        Nq, B, C = query.shape
        Nt = key.shape[0]
        assert Nt == value.shape[0], "K and V must have the same number of tokens"

        #
        # Add qkv linear projection
        #
        # qkv = self.qkv(torch.cat((query, key, value), dim=2))
        # qkv = qkv.reshape(N, B, 3, self.num_heads, C // self.num_heads)
        # qkv = qkv.permute(2, 1, 3, 0, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]

        query = self.qW(query)
        key = self.kW(key)
        value = self.vW(value)

        # WAS
        # B, N, C
        # B, N , 1, H, HC
        # 1, B, H, N, HC
        # q = query.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        # k = key.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        # v = value.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]

        # Converted to:
        # 1, B, H, N, HC
        d_k = C // self.num_heads
        q = query.reshape(Nq, B, 1, self.num_heads, d_k).permute(2, 1, 3, 0, 4)[0]
        k = key.reshape(Nt, B, 1, self.num_heads, d_k).permute(2, 1, 3, 0, 4)[0]
        v = value.reshape(Nt, B, 1, self.num_heads, d_k).permute(2, 1, 3, 0, 4)[0]

        # print("DECXCA qkv", q.shape, k.shape, v.shape)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        q_m = torch.nn.functional.normalize(self.q_mapper, dim=-1).transpose(-2, -1)
        x_m = torch.nn.functional.normalize(self.x_mapper, dim=-1)
        
        q = q @ q_m
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # x = (attn @ v)
        x = (attn @ v)
        x = x @ x_m.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # x = x_m @ x

        x = x.permute(1, 0, 2)
        
        # print("X", x.shape)
        return x, None

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class MHA(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., Nt=197, Nq=100):
        super(MHA, self).__init__()

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.num_heads = num_heads
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q_mapper = nn.Parameter(torch.ones(1, Nt, Nq))
        self.x_mapper = nn.Parameter(torch.ones(1, Nq, Nt))

        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        # xavier_uniform_(self.temperature)
        # xavier_uniform_(self.q_mapper)
        # xavier_uniform_(self.x_mapper)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                attn_mask=None, key_padding_mask=None):

        attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                None, None, False,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=None, need_weights=False,
                attn_mask=None,
                temperature=self.temperature,
                q_mapper=self.q_mapper,
                x_mapper=self.x_mapper,
                )

        return attn_output, attn_output_weights


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    temperature: Optional[Tensor] = None,
    q_mapper: Optional[Tensor] = None,
    x_mapper: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

    #
    # compute in-projection
    #
    q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    # attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output, attn_output_weights = _xc_attention(q, k, v, temperature, q_mapper, x_mapper, attn_mask, dropout_p)
    # attn_output is (B, Nq, E)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    return attn_output, None


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:

    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def _xc_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    temperature: Tensor,
    q_mapper: Tensor,
    x_mapper: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:

    B, Nq, E = q.shape
    Nt = k.shape[1]
    assert Nt == v.shape[1], "k and v must have the same number of tokens"

    q = q.transpose(-2, -1)
    k = k.transpose(-2, -1)
    v = v.transpose(-2, -1)

    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    q_mapper = torch.nn.functional.normalize(q_mapper, dim=-1).transpose(-2, -1)
    x_mapper = torch.nn.functional.normalize(x_mapper, dim=-1)
    
    nh = temperature.shape[0]
    bsz = B//nh
    temperature = temperature.unsqueeze(0).repeat(bsz, 1, 1, 1)
    temperature = temperature.reshape(B, 1, 1)

    # (B, E, Nq) x (B, Nq, Nt) -> (B, E, Nt)
    qm = torch.bmm(q, q_mapper.repeat(B, 1, 1))
    # (B, E, Nt) x (B, Nt, E) -> (B, E, E)
    attn = torch.bmm(qm, k.transpose(-2, -1)) * temperature
    attn = attn.softmax(dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)

    # (B, E, E) x (B, E, Nt) -> (B, E, Nt)
    x = torch.bmm(attn, v)
    # (B, E, Nt) x (B, Nt, Nq) -> (B, E, Nq)
    output = torch.bmm(x, x_mapper.repeat(B, 1, 1).transpose(-2, -1))
    # transpose output to (B, Nq, E) same as input
    output = output.transpose(-2, -1)
    return output, attn


class XCiTDecoderLayer2(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, eta=1.0):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.Nq = 100
        self.Nt = 197

        # BUG NOTE:
        # DecXCA "works" (not good but you get something) as an XCA decoder
        # MHA should be the same but it does not work at all.
        # 
        # - It is not the initialization of the weights
        # - It is not the torch.bmm vs torch.matmul
        #
        # Maybe some issue with the use of view and transpose?
        # https://jdhao.github.io/2019/07/10/pytorch_view_reshape_transpose_permute/

        # self.self_attn = DecXCA(d_model, nhead, attn_drop=dropout, proj_drop=dropout, Nq=self.Nq, Nt=self.Nq)
        # self.multihead_attn = DecXCA(d_model, nhead, attn_drop=dropout, proj_drop=dropout, Nq=self.Nq, Nt=self.Nt)
        self.self_attn = MHA(d_model, nhead, Nq=self.Nq, Nt=self.Nq)
        self.multihead_attn = MHA(d_model, nhead, Nq=self.Nq, Nt=self.Nt)

        self.local_mp = LPI(in_features=d_model, act_layer=nn.GELU)

        self.gamma0 = nn.Parameter(eta * torch.ones(d_model), requires_grad=True)
        self.gamma1 = nn.Parameter(eta * torch.ones(d_model), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(d_model), requires_grad=True)
        self.gamma3 = nn.Parameter(eta * torch.ones(d_model), requires_grad=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):

        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(self.gamma0 * tgt2)

        # # LPI (FIXME assuming num_queries==100)
        # tgt2 = self.norm3(tgt)
        # tgt2 = self.local_mp(tgt2.permute(1, 0, 2), 10, 10).permute(1, 0, 2)
        # tgt = tgt + self.dropout3(self.gamma2 * tgt2)

        # LPI on memory (FIXME assuming 224x224 input, patch_size 16)
        # memory[1:] = self.local_mp(memory[1:].permute(1, 0, 2), 14, 14).permute(1, 0, 2)
        # mem = tgt + self.dropout3(self.gamma2 * tgt2)

        # Multihead attention with memory
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(self.gamma1 * tgt2)

        # MLP
        tgt2 = self.norm4(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout4(self.gamma3 * tgt2)

        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class XCiTDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0., eta=1.0, dim_patches=(14, 14), with_lpi=True, num_queries=100):
        super().__init__()

        self.Nq = num_queries
        self.Nt = dim_patches[0] * dim_patches[1] + 1
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.self_attn = DecXCA(d_model, nhead, attn_drop=dropout, proj_drop=dropout, Nq=self.Nq, Nt=self.Nq)
        self.multihead_attn = DecXCA(d_model, nhead, attn_drop=dropout, proj_drop=dropout, Nq=self.Nq, Nt=self.Nt)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.gelu
        self.Wp, self.Hp = dim_patches
        self.with_lpi = with_lpi
        if self.with_lpi:
            self.local_mp = LPI(in_features=d_model, act_layer=nn.GELU)

        self.gamma1 = nn.Parameter(eta * torch.ones(d_model), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(d_model), requires_grad=True)
        self.gamma3 = nn.Parameter(eta * torch.ones(d_model), requires_grad=True)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        
        # memory is Nt B C
        key_mem = self.with_pos_embed(memory, pos)
        val_mem = memory
        
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=key_mem,
                                   value=val_mem)[0]
        
        tgt = tgt + self.gamma1 * self.dropout2(tgt2)
        # add LPI
        if self.with_lpi:
            tgt2 = self.norm3(tgt)
            tok = tgt2[1:].permute(1, 0, 2)  # get only the patch tokens
            tok = self.local_mp(tok, self.Hp, self.Wp).permute(1, 0, 2)
            tgt2[1:] = tok  # replace tokens
            tgt = tgt + self.gamma3 * tgt2

        # FC
        tgt2 = self.norm4(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.gamma2 * self.dropout3(tgt2)

        return tgt


class XCiTDec(nn.Module):

    def __init__(self, embed_dim=768, depth=12, num_heads=12, dim_feedforward=2048,
                 attn_drop_rate=0., norm_layer=None, return_intermediate=False,
                 dim_patches=(14, 14), with_lpi=True, num_queries=100):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.return_intermediate = return_intermediate

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.Nq = num_queries
        self.Nt = dim_patches[0] * dim_patches[1] + 1

        self.blocks = nn.ModuleList([
            XCiTDecoderLayer2(embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=attn_drop_rate,
                              activation='gelu', normalize_before=False)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # self.apply(self._init_weights)
        # self._reset_parameters()

    # def _reset_parameters(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)


    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward(self,  tgt, memory, memory_key_padding_mask=None, pos=None, query_pos=None):

        output = tgt
        intermediate = []

        # memory is Nt B C
        # permute to B Nt C
        # matmul with mem_mapper (mapper is 1 Nq Nt)
        # val_mem = (self.mem_mapper @ memory.permute(1, 0, 2)).permute(1, 0, 2)

        for blk in self.blocks:
            output = blk(output, memory, pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.return_intermediate:
            return torch.stack(intermediate)
        
        # else
        output = self.norm(output)
        return output.unsqueeze(0)

