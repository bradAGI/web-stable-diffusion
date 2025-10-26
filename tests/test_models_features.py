import math
import pathlib
import sys

import importlib
import types

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "web_stable_diffusion" / "models"


pkg = types.ModuleType("web_stable_diffusion")
pkg.__path__ = [str(ROOT / "web_stable_diffusion")]
sys.modules.setdefault("web_stable_diffusion", pkg)

models_pkg = types.ModuleType("web_stable_diffusion.models")
models_pkg.__path__ = [str(MODELS_DIR)]
sys.modules.setdefault("web_stable_diffusion.models", models_pkg)

attention_processor = importlib.import_module("web_stable_diffusion.models.attention_processor")
attention_module = importlib.import_module("web_stable_diffusion.models.attention")
transformer_module = importlib.import_module("web_stable_diffusion.models.transformer_2d")
resnet_module = importlib.import_module("web_stable_diffusion.models.resnet")

Attention = attention_processor.Attention
CrossAttention = attention_module.CrossAttention
Transformer2DModel = transformer_module.Transformer2DModel
Upsample2D = resnet_module.Upsample2D
Downsample2D = resnet_module.Downsample2D


def test_prepare_attention_mask_padding_and_truncation():
    attn = Attention(query_dim=4, heads=2, dim_head=2)
    mask = torch.zeros(1, 1, 3)
    padded = attn.prepare_attention_mask(mask, target_length=5, batch_size=1)
    assert padded.shape == (2, 1, 5)
    assert torch.allclose(padded[:, :, 3:], torch.zeros_like(padded[:, :, 3:]))

    long_mask = torch.ones(1, 1, 8)
    truncated = attn.prepare_attention_mask(long_mask, target_length=4, batch_size=1)
    assert truncated.shape == (2, 1, 4)
    assert torch.all(truncated == 1)


def test_attn_processor_respects_custom_scale():
    torch.manual_seed(0)
    attn = Attention(query_dim=4, heads=1, dim_head=4, scale_qk=False)
    hidden_states = torch.randn(2, 3, 4)

    output = attn(hidden_states)

    query = attn.to_q(hidden_states).view(2, 3, 1, 4).transpose(1, 2)
    key = attn.to_k(hidden_states).view(2, 3, 1, 4).transpose(1, 2)
    value = attn.to_v(hidden_states).view(2, 3, 1, 4).transpose(1, 2)

    default_scale = 1 / math.sqrt(query.size(-1))
    manual_query = query.clone()
    if attn.scale != default_scale:
        manual_query = manual_query * (attn.scale / default_scale)

    manual = torch.nn.functional.scaled_dot_product_attention(
        manual_query, key, value, dropout_p=0.0, is_causal=False
    )
    manual = manual.transpose(1, 2).reshape(2, 3, 4)
    manual = attn.to_out[1](attn.to_out[0](manual))

    assert torch.allclose(output, manual, atol=1e-5)


def test_cross_attention_applies_boolean_mask():
    attn = CrossAttention(query_dim=2, context_dim=2, heads=1, dim_head=2)
    torch.nn.init.eye_(attn.to_q.weight)
    torch.nn.init.eye_(attn.to_k.weight)
    torch.nn.init.eye_(attn.to_v.weight)
    torch.nn.init.eye_(attn.to_out[0].weight)
    attn.to_out[0].bias.data.zero_()

    hidden_states = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    context = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    mask = torch.tensor([[True, False]])

    output = attn(hidden_states, context=context, mask=mask)
    expected = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    assert torch.allclose(output, expected, atol=1e-6)


def test_transformer_continuous_out_channels_and_residual():
    torch.manual_seed(0)
    model = Transformer2DModel(
        num_attention_heads=1,
        attention_head_dim=4,
        in_channels=4,
        out_channels=2,
        num_layers=0,
        sample_size=2,
        norm_num_groups=4,
    )
    hidden = torch.randn(1, 4, 2, 2)
    out = model(hidden)
    assert out.shape == (1, 2, 2, 2)

    model_same = Transformer2DModel(
        num_attention_heads=1,
        attention_head_dim=4,
        in_channels=4,
        out_channels=4,
        num_layers=0,
        sample_size=2,
        norm_num_groups=4,
    )
    hidden_same = torch.randn(1, 4, 2, 2)
    out_same = model_same(hidden_same)
    assert out_same.shape == hidden_same.shape


def test_resnet_layers_accept_legacy_weights():
    up = Upsample2D(4, use_conv=True, name="conv")
    legacy_state = {
        "Conv2d_0.weight": torch.ones_like(up.conv.weight),
        "Conv2d_0.bias": torch.zeros_like(up.conv.bias),
    }
    up.load_state_dict(legacy_state, strict=False)
    assert torch.allclose(up.conv.weight, torch.ones_like(up.conv.weight))

    down = Downsample2D(4, use_conv=True, name="Conv2d_0")
    legacy_down_state = {
        "Conv2d_0.weight": torch.ones_like(down.conv.weight),
        "Conv2d_0.bias": torch.zeros_like(down.conv.bias),
    }
    down.load_state_dict(legacy_down_state, strict=False)
    assert torch.allclose(down.conv.weight, torch.ones_like(down.conv.weight))
