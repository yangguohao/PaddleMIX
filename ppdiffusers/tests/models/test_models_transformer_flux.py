# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle
from ppdiffusers import FluxTransformer2DModel
from ppdiffusers.utils.testing_utils import (
    enable_full_determinism,
    # paddle_device,
)
from .test_modeling_common import ModelTesterMixin

enable_full_determinism()


class FluxTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = FluxTransformer2DModel
    main_input_name = "hidden_states"
    @property
    def dummy_input(self):
        batch_size = 2
        embedding_dim = 32
        pooled_embedding_dim = embedding_dim * 2
        sequence_length = 128
        hidden_states = paddle.randn((batch_size, 32, 32))
        encoder_hidden_states = paddle.randn((batch_size, sequence_length, embedding_dim))
        pooled_prompt_embeds = paddle.randn((batch_size, pooled_embedding_dim))
        timestep = paddle.randint(0, 1000, shape=(batch_size,))
        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_prompt_embeds,
            "timestep": timestep,
            "txt_ids": paddle.randn((80, 3)),
            "img_ids": paddle.randn((80, 3)),
        }
    @property
    def input_shape(self):
        return (32, 32)
    @property
    def output_shape(self):
        return (32, 32)
    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "patch_size": 1,
            "in_channels": 32,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 128,
            "num_attention_heads": 4,
            "joint_attention_dim": 32,
            "pooled_projection_dim": 64,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    @unittest.skip("FluxTransformer2DModel uses a dedicated attention processor. This test doesn't apply")
    def test_from_save_pretrained(self):
        pass

    @unittest.skip("FluxTransformer2DModel uses a dedicated attention processor. This test doesn't apply")
    def test_outputs_equivalence(self):
        pass

    @unittest.skip("FluxTransformer2DModel uses a dedicated attention processor. This test doesn't apply")
    def test_set_attn_processor_for_determinism(self):
        pass
