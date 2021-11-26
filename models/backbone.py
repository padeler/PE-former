# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process, load_pretrained_weights_vit

from .position_encoding import build_position_encoding
from models import vision_transformer
import math

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    # def forward(self, tensor_list: NestedTensor):
    #     xs = self.body(tensor_list.tensors)
    #     out: Dict[str, NestedTensor] = {}
    #     for name, x in xs.items():
    #         m = tensor_list.mask
    #         assert m is not None
    #         mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
    #         out[name] = NestedTensor(x, mask)
    #     return out

    def forward(self, tensor_list):
        xs = self.body(tensor_list)
        return xs


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class BackboneVit(nn.Module):
    """Vit backbone."""
    def __init__(self, args):
        patch_size = args.patch_size
        # Initialize VIT
        if args.vit_arch != "resnet50":
            if "_tiny" in args.vit_arch:
                backbone = vision_transformer.deit_tiny(patch_size=patch_size, img_size=args.input_size,
                                                        drop_rate=args.vit_dropout, attn_drop_rate=args.vit_dropout,
                                                        drop_path_rate=args.vit_dropout)
            else:
                backbone = vision_transformer.deit_small(patch_size=patch_size, img_size=args.input_size,
                                                         drop_rate=args.vit_dropout, attn_drop_rate=args.vit_dropout,
                                                         drop_path_rate=args.vit_dropout)

            backbone.dim_patches = [backbone.patch_embed.img_size[0] // backbone.patch_embed.patch_size[0],
                                    backbone.patch_embed.img_size[1] // backbone.patch_embed.patch_size[1]]

            self.is_vit_backbone = True
            load_pretrained_weights_vit(backbone, args.vit_weights,
                                        checkpoint_key="teacher", model_name=args.vit_arch, patch_size=patch_size)
        elif args.vit_arch == "resnet50":
            from models.transformer_vit import create_resnet_encoder
            backbone = create_resnet_encoder(args)
            self.is_vit_backbone = False

            # XXX This is the output features size for
            # resnet50. Resnet50 Downsamples the input by 32
            backbone.dim_patches = [args.input_size[0] // 32,
                                    args.input_size[1] // 32]

        super().__init__()

        num_patches = backbone.dim_patches[0] * backbone.dim_patches[1]
        # number of patches in the image
        # i.e the image has patches_dim x patches_dim
        self.num_patches = num_patches
        # self.patches_dim = int(num_patches ** 0.5)

        # Dont train the backbone ViT as a first test
        # for name, parameter in backbone.named_parameters():
        #     parameter.requires_grad_(False)
    
        # return_layers = {'layer4': "0"}
        self.body = backbone
        # number of ViT "channels"
        # depends on input res and number of boxes.
        # I.e 224x224 input res with 16x16 box size gives 14x14 outputs 
        # 16X16 patch size for an RGB image (3channels) is 16*16*3 = 764
        # Deit_small halfs this input to 384 and that is its embedding dim size

        # I concatenate the vector from CLS with each patch vector (of size args.vim_dim)
        # This gives a total number of channels 2 * args.vit_dim
        self.num_channels = args.vit_dim

    def forward(self, tensor_list):
        if self.is_vit_backbone:  # vit version
            out = self.body.forward(tensor_list, cls_only=False)
        else:  # resnet version
            out = self.body.forward(tensor_list)
            out = out.permute(0, 2, 1)  # send channels last

        # Shape is B, N, C where N is patches+1 (i.e 14*14+1 for 16 patch size and 224 input)
        return out
        
        # convert to xs
        # XXX The following was part of early testing with concatenating CLS with
        # each tokern to increase the channels.
        # Turned out to make no positive difference
        # Keeping it here in comments for now.

        # cls_out = vit_out[:, 0:1, :]
        # vit_out = vit_out[:, 1:, :]
        # B, N, C = vit_out.shape
        
        # cls_out = cls_out.expand(-1, N, -1)
        # # cls_out
        # vit_out = torch.cat((cls_out, vit_out), 2)
        # C = 2 * C  # number of channels after concat

        # D = self.patches_dim
        # xs = OrderedDict()
        # xs["0"] = vit_out.reshape(B, D, D, C).permute(0, 3, 1, 2)
        # return xs


# class Joiner(nn.Sequential):
#     def __init__(self, backbone, position_embedding):
#         super().__init__(backbone, position_embedding)

#     def forward(self, tensor_list):
#         xs = self[0](tensor_list)
#         out = []
#         pos = []
#         for name, x in xs.items():
#             out.append(x)
#             # position encoding
#             pos.append(self[1](x).to(x[0].dtype))

#         return out, pos


# def build_backbone(args):
#     position_embedding = build_position_encoding(args)
#     train_backbone = args.lr_backbone > 0
#     return_interm_layers = args.masks
#     backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
#     model = Joiner(backbone, position_embedding)
#     model.num_channels = backbone.num_channels
#     return model


def build_backbone_vit(args):
    # position_embedding = build_position_encoding(args)
    backbone = BackboneVit(args)
    # backbone = Backbone("resnet50", True, False, False)

    # model = Joiner(backbone, position_embedding)
    # model.num_channels = backbone.num_channels
    return backbone
