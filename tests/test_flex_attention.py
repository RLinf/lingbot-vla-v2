# Licensed under Apache-2.0; see LICENSE.
"""Cached attention preserves the supplied prefix, causal, and padding mask."""

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask

from lingbotvla.models.vla.lingbot_vla.flex_attention import (
    build_block_mask,
    flex_attention_with_block_mask,
)


CASES = [(128, 193, False), (7, 193, False), (321, 0, False), (128, 193, True), (321, 0, "block")]


def visibility(query_length, prefix_length, padding, device):
    keys = torch.arange(prefix_length + query_length, device=device)
    query = torch.arange(query_length, device=device)
    allowed = keys[None, :] <= prefix_length + query[:, None]
    if padding == "block":
        allowed = (keys[None, :] < 193) | (query[:, None] >= 193)
    elif padding:
        allowed[:, :3] = False
    return allowed[None]


@pytest.mark.parametrize("query_length,prefix_length,padding", CASES)
def test_cached_mask_visibility(query_length, prefix_length, padding):
    allowed = visibility(query_length, prefix_length, padding, "cpu")
    kv_length = prefix_length + query_length
    block = build_block_mask(allowed, 4, query_length, kv_length)
    actual = create_mask(block.mask_mod, B=1, H=4, Q_LEN=block.shape[-2], KV_LEN=block.shape[-1], device="cpu")
    expected = torch.zeros(block.shape, dtype=torch.bool)
    expected[:, :, :query_length, :kv_length] = allowed[:, None]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if prefix_length:
        assert actual[0, 0, 0, prefix_length - 1]
        assert actual[0, 0, query_length - 1, prefix_length - 1]
    assert actual[0, 0, 0, prefix_length]
    if padding == "block":
        assert not actual[0, 0, 192, 193]
        assert actual[0, 0, 193, 320]
    elif query_length > 1:
        assert not actual[0, 0, 0, prefix_length + 1]
        assert actual[0, 0, 1, prefix_length + 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Flex kernel requires CUDA")
@pytest.mark.parametrize("query_length,prefix_length,padding", CASES)
def test_cached_attention_matches_dense_reference(query_length, prefix_length, padding):
    generator = torch.Generator().manual_seed(1729)
    kv_length = prefix_length + query_length
    q = torch.randn(1, query_length, 4, 32, generator=generator).cuda()
    k = torch.randn(1, kv_length, 2, 32, generator=generator).cuda()
    v = torch.randn(1, kv_length, 2, 32, generator=generator).cuda()
    allowed = visibility(query_length, prefix_length, padding, "cuda")
    block = build_block_mask(allowed, 4, query_length, kv_length)
    actual = flex_attention_with_block_mask(q, k, v, block, query_length)
    keys = k.transpose(1, 2).repeat_interleave(2, dim=1)
    values = v.transpose(1, 2).repeat_interleave(2, dim=1)
    scores = q.transpose(1, 2) @ keys.transpose(-1, -2) / 32**0.5
    scores = scores.masked_fill(~allowed[:, None], float("-inf"))
    reference = (scores.softmax(-1) @ values).transpose(1, 2).reshape(1, query_length, 128)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cached attention regression uses CUDA")
@pytest.mark.parametrize("prefix_length", [0, 193])
def test_v2_forward_decodes_with_prefix_cache(prefix_length):
    from types import SimpleNamespace

    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import QwenvlWithExpertV2Model

    class ProjectionLayer(torch.nn.Module):
        def forward(
            self, hidden, attention=None, start=None, end=None, compute_kqv=False, output_atten=False, ada_cond=None
        ):
            if compute_kqv:
                q = hidden.reshape(1, 128, 4, 32)
                return q, q[:, :, :2], q[:, :, :2] * 0.5
            return attention[:, start:end], None

    # Isolate the real V2 forward/cache dispatch from pretrained projection weights and RoPE.
    model = QwenvlWithExpertV2Model.__new__(QwenvlWithExpertV2Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        attention_implementation="flex_cached",
        qwen_expert_config=SimpleNamespace(num_hidden_layers=1),
        final_norm_adanorm=False,
    )
    decoder = SimpleNamespace(layers=[ProjectionLayer()], norm=torch.nn.Identity())
    model.qwenvl = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=1, num_attention_heads=4)),
        model=SimpleNamespace(language_model=decoder),
    )
    model.qwen_expert = SimpleNamespace(model=decoder)
    model.apply_mrope = lambda q, k, positions: (q, k)
    generator = torch.Generator().manual_seed(1729)
    hidden = torch.randn(1, 128, 128, generator=generator).cuda()
    prefix_k = torch.randn(1, prefix_length, 2, 32, generator=generator).cuda()
    prefix_v = torch.randn(1, prefix_length, 2, 32, generator=generator).cuda()
    cache = {0: {"key_states": prefix_k, "value_states": prefix_v}}
    mask = visibility(128, prefix_length, False, "cuda")
    outputs, returned_cache, _ = model.forward(
        attention_mask=mask, inputs_embeds=[None, hidden], past_key_values=cache, use_cache=True, fill_kv_cache=False
    )
    query = hidden.reshape(1, 128, 4, 32).transpose(1, 2)
    keys = torch.cat((prefix_k, hidden.reshape(1, 128, 4, 32)[:, :, :2]), dim=1)
    values = torch.cat((prefix_v, hidden.reshape(1, 128, 4, 32)[:, :, :2] * 0.5), dim=1)
    keys = keys.transpose(1, 2).repeat_interleave(2, dim=1)
    values = values.transpose(1, 2).repeat_interleave(2, dim=1)
    scores = (query @ keys.transpose(-1, -2) / 32**0.5).masked_fill(~mask[:, None], float("-inf"))
    expected = (scores.softmax(-1) @ values).transpose(1, 2).reshape(1, 128, 128)
    assert returned_cache is cache
    assert outputs[0] is None
    assert torch.isfinite(outputs[1]).all()
    torch.testing.assert_close(outputs[1], expected, rtol=1e-4, atol=1e-5)
