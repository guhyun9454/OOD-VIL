"""
Minimal VisionTransformer implementation for RainbowPrompt (vendored).

This is a trimmed variant based on RainbowPrompt's `vision_transformer.py`,
kept intentionally small:
- Only the pieces required by RainbowPrompt in this OOD-VIL integration are included.
- No timm model registry registration is performed.
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn

from timm.models.helpers import checkpoint_seq
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_

from .prompt import RainbowPrompt
from .attention import PreT_Attention


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, *args):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_layer=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, prompt=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), prompt)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=None,
        class_token=True,
        no_embed_class=False,
        fc_norm=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        block_fn=Block,
        # RainbowPrompt args
        prompt_length=None,
        embedding_key="cls",
        prompt_init="uniform",
        prompt_pool=False,
        prompt_key=False,
        pool_size=None,
        top_k=None,
        batchwise_prompt=False,
        prompt_key_init="uniform",
        head_type="token",
        use_prompt_mask=False,
        use_g_prompt=False,
        g_prompt_length=None,
        g_prompt_layer_idx=None,
        use_prefix_tune_for_g_prompt=False,
        use_e_prompt=False,
        e_prompt_layer_idx=None,
        use_prefix_tune_for_e_prompt=False,
        same_key_value=False,
        n_tasks=None,
        D1=None,
        relation_type=None,
        use_linear=None,
        warm_up=None,
        KI_iter=None,
        self_attn_idx=None,
        D2=None,
    ):
        super().__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"

        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.warmup = warm_up if warm_up is not None else 0
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.class_token = class_token
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + (1 if class_token else 0)
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.prompt_pool = prompt_pool
        self.head_type = head_type
        self.use_prompt_mask = use_prompt_mask
        self.use_e_prompt = use_e_prompt
        self.e_prompt_layer_idx = e_prompt_layer_idx or []
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt

        # We only implement E-Prompt + prefix tuning path used by RainbowPrompt.
        if use_e_prompt and len(self.e_prompt_layer_idx) > 0:
            self.rainbow_prompt = RainbowPrompt(
                length=prompt_length,
                embed_dim=embed_dim,
                embedding_key=embedding_key,
                prompt_init=prompt_init,
                prompt_pool=prompt_pool,
                prompt_key=prompt_key,
                pool_size=pool_size,
                top_k=top_k,
                batchwise_prompt=batchwise_prompt,
                prompt_key_init=prompt_key_init,
                num_layers=len(self.e_prompt_layer_idx),
                use_prefix_tune_for_e_prompt=use_prefix_tune_for_e_prompt,
                num_heads=num_heads,
                same_key_value=same_key_value,
                prompt_tune_idx=self.e_prompt_layer_idx,
                n_tasks=n_tasks,
                D1=D1,
                relation_type=relation_type,
                use_linear=use_linear,
                KI_iter=KI_iter,
                self_attn_idx=self_attn_idx,
                D2=D2,
            )
        else:
            self.rainbow_prompt = None

        if self.use_e_prompt and self.use_prefix_tune_for_e_prompt:
            attn_layer = PreT_Attention
        else:
            attn_layer = Attention

        self.total_prompt_len = 0
        if self.prompt_pool and prompt_length is not None and top_k is not None:
            self.total_prompt_len += int(prompt_length) * int(top_k) * len(self.e_prompt_layer_idx)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    init_values=init_values,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    attn_layer=attn_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != "skip":
            self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if isinstance(self.head, nn.Linear):
            nn.init.zeros_(self.head.bias)

    def forward_features(self, x, task_id=-1, learned_id=-1, cls_features=None, train=False, epoch_info=None):
        if train:
            prompt_type = "Rainbow" if (epoch_info is not None and epoch_info >= int(self.warmup)) else "Unique"
        else:
            prompt_type = "Rainbow"

        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
            res = {}
        else:
            res = {}
            if self.use_e_prompt and self.rainbow_prompt is not None:
                for i, block in enumerate(self.blocks):
                    if i in self.e_prompt_layer_idx:
                        pr = self.rainbow_prompt(
                            x,
                            layer=i,
                            previous_mask=None,
                            cls_features=cls_features,
                            task_id=task_id,
                            cur_id=learned_id,
                            train=train,
                            p_type=prompt_type,
                        )
                        res["sim_loss"] = pr.get("sim_loss", res.get("sim_loss", None))
                        e_prompt = pr["batched_prompt"]
                        x = block(x, prompt=e_prompt)
                    else:
                        x = block(x)
            else:
                x = self.blocks(x)

        x = self.norm(x)
        res["x"] = x
        return res

    def forward_head(self, res):
        x = res["x"]
        if self.class_token and self.head_type == "token":
            x = x[:, 0]
        elif self.head_type == "gap":
            x = x.mean(dim=1)
        elif self.head_type == "prompt" and self.prompt_pool:
            x = x[:, 1 : (1 + self.total_prompt_len)] if self.class_token else x[:, 0 : self.total_prompt_len]
            x = x.mean(dim=1)
        elif self.head_type == "token+prompt" and self.prompt_pool and self.class_token:
            x = x[:, 0 : self.total_prompt_len + 1]
            x = x.mean(dim=1)
        else:
            raise ValueError(f"Invalid head_type={self.head_type}")

        res["pre_logits"] = x
        x = self.fc_norm(x)
        res["logits"] = self.head(x)
        return res

    def forward(self, x, task_id=-1, learned_id=-1, cls_features=None, train=False, epoch_info=None):
        res = self.forward_features(x, task_id=task_id, learned_id=learned_id, cls_features=cls_features, train=train, epoch_info=epoch_info)
        res = self.forward_head(res)
        return res


def vit_base_patch16_224(pretrained=False, **kwargs):
    # pretrained is handled by the engine via timm state_dict loading
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    return VisionTransformer(**model_kwargs)

