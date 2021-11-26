# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from re import M
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from models import position_encoding

from models.xcit_decoder import XCiTDec
from models.transformer import TransformerDecoder, TransformerDecoderLayer
from models import vision_transformer
from util.misc import NestedTensor, load_pretrained_weights_vit
from models.vision_transformer import trunc_normal_
from models.position_encoding import build_position_encoding

from timm.models import create_model
from models import xcit
import math


class TransformerVit(nn.Module):

    def __init__(self, encoder, decoder, vit_dim=512, d_model=512, nhead=8,  pos_embed_version='sine'):
        super().__init__()

        self.encoder = encoder  # use deit as an encoder
        self.decoder = decoder
    
        self.lin_proj_mem = nn.Linear(vit_dim, d_model, bias=False)
        # self.lin_proj_pos = nn.Linear(vit_dim, d_model, bias=False)

        self.vit_norm = nn.LayerNorm(d_model)
        num_patches = math.prod(self.encoder.dim_patches)
        self.pos_embed_version = pos_embed_version
        if pos_embed_version in ["enc_learned", "enc_sine"]:
            self.pos_encoder = build_position_encoding(d_model, version=pos_embed_version)
        elif pos_embed_version == "enc_xcit":
            # same as enc_sine but using the xcit code 
            # (which should be identical anyway but i am lazy right now)
            from models.xcit import PositionalEncodingFourier
            self.pos_encoder = PositionalEncodingFourier(dim=d_model)

        elif pos_embed_version == "learned_cls":
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
            trunc_normal_(self.pos_embed, std=.02)
        elif pos_embed_version == "learned_nocls":
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
            trunc_normal_(self.pos_embed, std=.02)
        elif pos_embed_version == "none":  # no positional embedding for keys of the decoder
            self.pos_embed = None

        # torch.nn.init.zeros_(self.lin_proj_mem.bias)
        # torch.nn.init.zeros_(self.lin_proj_pos.bias)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for n, p in self.named_parameters():
            # PPP Do not reinitialize the VIT encoder. 
            # Pretrained weights are used
            if p.dim() > 1 and "encoder" not in n:
                # print("Initializing ", n, p.dim())
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, query_embed):
        # flatten NxCxHxW to HWxNxC

        # Src should be Bs, 3, 224, 224
        # print("SRC SHAPE ", src.shape)
        bs, c, h, w = src.shape
        if type(self.encoder) == torch.nn.Sequential:
            memory = self.encoder(src)
            memory = memory.permute(0, 2, 1)  # send channels last
        else:
            memory = self.encoder(src, cls_only=False)

        # print("Memory shape ", memory.shape)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # print("Query embed shape ", query_embed.shape)

        tgt = torch.zeros_like(query_embed)

        if self.pos_embed_version in ["enc_learned", "enc_sine"]:
            pW, pH = self.encoder.dim_patches
            pos_embed = self.pos_encoder(bs, pH, pW, src.device).flatten(2)
            memory = memory[:, 1:, :]  # drop CLS token
            # XXX DeTR Decoder expects the dim order of pos embed to be 
            # the same as the memory (see note on memory permutation)
            pos_embed = pos_embed.permute(2, 0, 1)
        elif self.pos_embed_version == "enc_xcit":
            pW, pH = self.encoder.dim_patches
            pos_embed = self.pos_encoder(bs, pH, pW).flatten(2)
            # memory = memory[:, 1:, :]  # drop CLS token
            # XXX DeTR Decoder expects the dim order of pos embed to be 
            # the same as the memory (see note on memory permutation)
            cls_pos_shape = (pos_embed.shape[0], pos_embed.shape[1], 1)
            cls_pos = torch.zeros(cls_pos_shape, dtype=pos_embed.dtype, device=pos_embed.device)
            pos_embed = torch.cat((cls_pos, pos_embed), dim=2)
            pos_embed = pos_embed.permute(2, 0, 1)

        elif self.pos_embed_version in ["learned_cls", "learned_nocls"]:
            pos_embed = self.pos_embed.repeat(bs, 1, 1)
            # XXX DeTR Decoder expects the dim order of pos embed to be
            # the same as the memory (see note on memory permutation)
            # note: permute here is different because construction is different.
            pos_embed = pos_embed.permute(1, 0, 2)
        elif self.pos_embed_version == "none":
            pos_embed = None

        # print("POS Embed: ", pos_embed.shape)
        memory = self.vit_norm(self.lin_proj_mem(memory))
        # pos_embed = self.vit_norm(self.lin_proj_pos(pos_embed))


        # Memory should be BS, 197, 384
        # XXX DeTR Decoder expects the shape to be 197, BS, 384
        memory = memory.permute(1, 0, 2)

        hs = self.decoder(tgt, memory, memory_key_padding_mask=None,
                          pos=pos_embed, query_pos=query_embed)

        # XXX memory is expected permuted by the decoder
        return hs.transpose(1, 2), memory.permute(1, 0, 2)


def create_resnet_encoder(args):
    model = create_model("resnet50", num_classes=0, pretrained=True)

    # Remove global_pool. We need all the features.
    model.global_pool = torch.nn.Identity()

    # Following DETR code: Only train layers 2,3,4.
    # TODO Test with frozenBN
    # for name, parameter in model.named_parameters():
    #     if name not in ['layer2', 'layer3', 'layer4']:
    #         parameter.requires_grad_(False)

    # Apply a 1x1 conv to reduce the channels if needed
    input_proj = nn.Conv2d(model.num_features, args.vit_dim, kernel_size=1)

    encoder = torch.nn.Sequential(model, input_proj, torch.nn.Flatten(2))
    encoder.num_channels = args.vit_dim

    print(f"Encoder.num_channels {encoder.num_channels}")

    return encoder


def build_transformer_vit(args):

    patch_size = args.patch_size
    if "xcit_" not in args.vit_arch and args.vit_arch != "resnet50":
    # Initialize VIT
        if "_tiny" in args.vit_arch:
            encoder = vision_transformer.deit_tiny(patch_size=patch_size, img_size=args.input_size)
        else:
            encoder = vision_transformer.deit_small(patch_size=patch_size, img_size=args.input_size)

        encoder.dim_patches = [encoder.patch_embed.img_size[0] // encoder.patch_embed.patch_size[0],
                               encoder.patch_embed.img_size[1] // encoder.patch_embed.patch_size[1]]

        load_pretrained_weights_vit(encoder, args.vit_weights,
                                    checkpoint_key="teacher", model_name=args.vit_arch, patch_size=patch_size)

    elif "xcit_" in args.vit_arch:
        encoder = create_model(args.vit_arch,
                               pretrained=False,
                               num_classes=0,
                               img_size=args.input_size,
                               drop_rate=args.vit_dropout,
                               drop_block_rate=None)

        encoder.dim_patches = [encoder.patch_embed.img_size[0] // encoder.patch_embed.patch_size[0],
                               encoder.patch_embed.img_size[1] // encoder.patch_embed.patch_size[1]]

        # Load weights here
        if args.vit_weights is not None:
            if args.vit_weights.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.vit_weights, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.vit_weights, map_location='cpu')

            checkpoint_model = checkpoint['model']
            msg = encoder.load_state_dict(checkpoint_model, strict=False)
            print(f"Loaded encoder weights from {args.vit_weights}. Message: {msg}")

    elif "resnet50" == args.vit_arch:
        encoder = create_resnet_encoder(args)
        # Load weights here
        if args.vit_weights is not None:
            if args.vit_weights.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.vit_weights, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.vit_weights, map_location='cpu')

            msg = encoder[0].load_state_dict(checkpoint, strict=False)
            print(f"Loaded resnet50 encoder weights from {args.vit_weights}. Message: {msg}")


        # XXX This is the output features size for
        # resnet50. Resnet50 Downsamples the input by 32
        encoder.dim_patches = [args.input_size[0] // 32,
                               args.input_size[1] // 32]

    #  Prepare Decoder
    if args.dec_arch == "detr":
        decoder_layer = TransformerDecoderLayer(args.hidden_dim, args.nheads, args.dim_feedforward,
                                                args.dropout, args.activation, args.pre_norm)
        decoder_norm = nn.LayerNorm(args.hidden_dim)
        decoder = TransformerDecoder(decoder_layer, args.dec_layers, decoder_norm,
                                     return_intermediate=True)
    elif args.dec_arch == "xcit":
        decoder = XCiTDec(args.hidden_dim, args.dec_layers, args.nheads, dim_feedforward=args.dim_feedforward,
                          attn_drop_rate=args.dropout, return_intermediate=True,
                          dim_patches=encoder.dim_patches, with_lpi=args.with_lpi,
                          num_queries=args.num_queries)

    return TransformerVit(
        encoder=encoder,
        decoder=decoder,
        vit_dim=args.vit_dim,
        d_model=args.hidden_dim,
        nhead=args.nheads,
        pos_embed_version=args.position_embedding
    )
