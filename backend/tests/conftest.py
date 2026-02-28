"""Shared test fixtures for Neural-MRI backend tests."""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest
import torch

from neural_mri.core.model_manager import ModelManager
from neural_mri.core.scan_cache import ScanCache
from neural_mri.schemas.model import LayerConfig, ModelInfo

N_LAYERS = 2
N_HEADS = 2
D_MODEL = 8
D_HEAD = 4
D_MLP = 16
D_VOCAB = 50
SEQ_LEN = 4


def _make_cfg(**overrides):
    defaults = dict(
        n_layers=N_LAYERS, n_heads=N_HEADS, d_model=D_MODEL,
        d_head=D_HEAD, d_mlp=D_MLP, d_vocab=D_VOCAB,
        n_ctx=64, model_name="mock-gpt2", device="cpu", dtype="float32",
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _build_mock_cache(cfg, seq_len=SEQ_LEN):
    cache = {}
    cache["hook_embed"] = torch.randn(1, seq_len, cfg.d_model)
    for i in range(cfg.n_layers):
        cache[f"blocks.{i}.hook_resid_post"] = torch.randn(1, seq_len, cfg.d_model)
        cache[f"blocks.{i}.hook_resid_pre"] = torch.randn(1, seq_len, cfg.d_model)
        cache[f"blocks.{i}.attn.hook_z"] = torch.randn(1, seq_len, cfg.n_heads, cfg.d_head)
        cache[f"blocks.{i}.hook_attn_out"] = torch.randn(1, seq_len, cfg.d_model)
        cache[f"blocks.{i}.hook_mlp_out"] = torch.randn(1, seq_len, cfg.d_model)
        cache[f"blocks.{i}.attn.hook_pattern"] = torch.softmax(
            torch.randn(1, cfg.n_heads, seq_len, seq_len), dim=-1,
        )
    return cache


@pytest.fixture
def mock_cfg():
    return _make_cfg()


@pytest.fixture
def mock_model(mock_cfg):
    cfg = mock_cfg
    model = MagicMock()
    model.cfg = cfg

    # to_tokens(prompt) -> [1, seq_len]; to_tokens(tok, prepend_bos=False) -> [1, 1]
    def _to_tokens(prompt, prepend_bos=True):
        if not prepend_bos:
            return torch.randint(0, cfg.d_vocab, (1, 1))
        return torch.randint(0, cfg.d_vocab, (1, SEQ_LEN))
    model.to_tokens = MagicMock(side_effect=_to_tokens)

    model.to_str_tokens = MagicMock(return_value=["<bos>", "The", " capital", " of"])
    model.to_string = MagicMock(side_effect=lambda ids: f"tok_{ids[0]}")

    model.tokenizer = MagicMock()
    model.tokenizer.decode = MagicMock(side_effect=lambda ids: f"tok_{ids[0]}")

    cache = _build_mock_cache(cfg)
    logits = torch.randn(1, SEQ_LEN, cfg.d_vocab)
    model.run_with_cache = MagicMock(return_value=(logits, cache))
    model.return_value = logits
    model.run_with_hooks = MagicMock(return_value=logits)

    sd = {}
    for i in range(cfg.n_layers):
        sd[f"blocks.{i}.attn.W_Q"] = torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head)
        sd[f"blocks.{i}.attn.W_K"] = torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head)
        sd[f"blocks.{i}.mlp.W_in"] = torch.randn(cfg.d_model, cfg.d_mlp)
        sd[f"blocks.{i}.mlp.W_out"] = torch.randn(cfg.d_mlp, cfg.d_model)
    sd["embed.W_E"] = torch.randn(cfg.d_vocab, cfg.d_model)
    sd["unembed.W_U"] = torch.randn(cfg.d_model, cfg.d_vocab)
    model.state_dict = MagicMock(return_value=sd)

    params = list(sd.values())
    model.parameters = MagicMock(return_value=iter(params))

    model.W_U = torch.randn(cfg.d_model, cfg.d_vocab)
    model.b_U = torch.randn(cfg.d_vocab)
    model.ln_final = lambda x: x

    return model


@pytest.fixture
def mock_model_manager(mock_model):
    mm = MagicMock(spec=ModelManager)
    mm._model = mock_model
    mm._model_id = "gpt2"
    mm.model_id = "gpt2"
    mm.is_loaded = True
    mm.get_model.return_value = mock_model
    mm.get_model_info.return_value = ModelInfo(
        model_id="gpt2",
        model_name="mock-gpt2",
        n_params=124_000_000,
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        d_vocab=D_VOCAB,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        d_mlp=D_MLP,
        max_seq_len=64,
        device="cpu",
        layers=[
            LayerConfig(
                layer_index=i, layer_type="transformer_block",
                components=["attn", "mlp"], num_heads=N_HEADS,
                d_model=D_MODEL, d_head=D_HEAD, d_mlp=D_MLP,
            )
            for i in range(N_LAYERS)
        ],
        dtype="float32",
    )
    return mm


@pytest.fixture
def mock_sae():
    sae = MagicMock()
    sae.cfg = types.SimpleNamespace(
        d_sae=32,
        metadata={"hook_name": "blocks.1.hook_resid_pre"},
    )

    def _encode(x):
        batch_shape = x.shape[:-1]
        return torch.relu(torch.randn(*batch_shape, 32))
    sae.encode = MagicMock(side_effect=_encode)

    def _decode(x):
        return torch.randn(*x.shape[:-1], D_MODEL)
    sae.decode = MagicMock(side_effect=_decode)
    return sae


@pytest.fixture
def mock_sae_manager(mock_sae):
    mgr = MagicMock()
    mgr.get_sae.return_value = mock_sae
    return mgr
