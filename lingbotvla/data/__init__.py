# Modified for RLinf: lazy dataset imports and PPO-compatible MoE replay.
# Licensed under Apache-2.0; see LICENSE.

# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from importlib import import_module


# Model preprocessing and data.constants must not require offline LeRobot datasets.
# Keep the training-facing exports, importing each implementation when requested.
_EXPORTS = {
    "build_chat_template": ".chat_template",
    "build_dataloader": ".data_loader",
    "build_iterative_dataset": ".dataset",
    "build_mapping_dataset": ".dataset",
    "build_vla_dataset": ".dataset",
    "build_multimodal_chat_template": ".multimodal.multimodal_chat_template",
    **{name: ".data_collator" for name in (
        "CollatePipeline", "DataCollatorWithPacking", "DataCollatorWithPadding",
        "DataCollatorWithPositionIDs", "MakeMicroBatchCollator",
        "TextSequenceShardCollator", "UnpackDataCollator",
    )},
    **{name: ".multimodal.data_collator" for name in (
        "OmniDataCollatorWithPacking", "OmniDataCollatorWithPadding",
        "OmniSequenceShardCollator", "VLADataCollatorWithPacking",
    )},
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
