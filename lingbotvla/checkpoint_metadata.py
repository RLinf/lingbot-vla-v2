# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Public metadata reader for exported and official-release VLA2 checkpoints.

This module does not load weights, start inference, or own a training loop.
The official release stores architecture fields in lingbotvla_cli.yaml and
only a family marker in hf_ckpt/config.json.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def read_checkpoint_config(checkpoint: str | Path) -> dict[str, Any]:
    """Read full exported metadata, or the pinned release's sibling CLI config."""
    checkpoint = Path(checkpoint).expanduser()
    with (checkpoint / "config.json").open() as source:
        config = json.load(source)
    if "max_action_dim" in config and "chunk_size" in config:
        return config
    release_config = checkpoint.parent.parent.parent / "lingbotvla_cli.yaml"
    if not release_config.is_file():
        raise ValueError(
            f"{checkpoint}/config.json is incomplete; expected a full exported "
            f"model config or official release metadata at {release_config}"
        )
    with release_config.open() as source:
        release = yaml.safe_load(source)
    config = {**release["model"], **release["train"]}
    for field in ("max_action_dim", "max_state_dim", "chunk_size"):
        if not isinstance(config.get(field), int) or config[field] <= 0:
            raise ValueError(f"Official checkpoint requires positive {field}")
    return config


def build_checkpoint_config(values: dict[str, Any], base_model: str | Path):
    """Reconstruct the same VLA/Qwen configuration as the release evaluator.

    A caller chooses attention/cache/training flags after construction. Never
    discard depth/video query heads: their tokens also affect action inference.
    """
    from transformers import AutoConfig

    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
        LingbotVLAV2Config,
    )

    values = deepcopy(values)
    config = LingbotVLAV2Config(**values)
    for key, value in values.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    qwen = AutoConfig.from_pretrained(str(base_model), local_files_only=True)
    if qwen.model_type != "qwen3_vl":
        raise ValueError(f"VLA2 requires qwen3_vl, got {qwen.model_type!r}")
    qwen_dict = qwen.to_dict()
    text = qwen_dict["text_config"]
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "max_position_embeddings",
        "hidden_act",
        "tie_word_embeddings",
    ):
        if key in text:
            setattr(config, key, text[key])
        elif key in qwen_dict:
            setattr(config, key, qwen_dict[key])
    config.vision_config = qwen.vision_config
    if values.get("vocab_size", 0):
        config.vocab_size = values["vocab_size"]
    config.tokenizer_path = str(Path(base_model).expanduser())
    return config
