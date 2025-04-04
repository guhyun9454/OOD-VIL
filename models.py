# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Joint-FT
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import torch.nn as nn
from timm.models._registry import register_model
from engines.custom_vit.vit_with_adapter import _create_vision_transformer
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

__all__ = [
    'vit_base_patch16_224_ICON',
]

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


@register_model
def vit_base_patch16_224_ICON(pretrained=False, **kwargs):
    # print("vit_base_patch16_224")
    # print(kwargs)
    # {'pretrained_cfg': None, 'pretrained_cfg_overlay': None, 'num_classes': 10, 'drop_rate': 0.0, 'drop_path_rate': 0.0, 'adapt_blocks': [0, 1, 2, 3, 4]}
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    default_cfg = _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
        custom_load=True,)
    kwargs.update(pretrained_cfg=default_cfg)

    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)

    model = _create_vision_transformer(
        'vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
