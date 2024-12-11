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

from __future__ import annotations

import sys

import paddle

import paddlemix.models.diffsinger.modules.compat as compat
from paddlemix.models.diffsinger.modules.core.ddpm import MultiVarianceDiffusion
from paddlemix.models.diffsinger.utils import filter_kwargs
from paddlemix.models.diffsinger.utils.hparams import hparams

VARIANCE_CHECKLIST = ["energy", "breathiness", "voicing", "tension"]


class ParameterAdaptorModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.variance_prediction_list = []
        self.predict_energy = hparams.get("predict_energy", False)
        self.predict_breathiness = hparams.get("predict_breathiness", False)
        self.predict_voicing = hparams.get("predict_voicing", False)
        self.predict_tension = hparams.get("predict_tension", False)
        if self.predict_energy:
            self.variance_prediction_list.append("energy")
        if self.predict_breathiness:
            self.variance_prediction_list.append("breathiness")
        if self.predict_voicing:
            self.variance_prediction_list.append("voicing")
        if self.predict_tension:
            self.variance_prediction_list.append("tension")
        self.predict_variances = len(self.variance_prediction_list) > 0

    def build_adaptor(self, cls=MultiVarianceDiffusion):
        ranges = []
        clamps = []
        if self.predict_energy:
            ranges.append((hparams["energy_db_min"], hparams["energy_db_max"]))
            clamps.append((hparams["energy_db_min"], 0.0))
        if self.predict_breathiness:
            ranges.append((hparams["breathiness_db_min"], hparams["breathiness_db_max"]))
            clamps.append((hparams["breathiness_db_min"], 0.0))
        if self.predict_voicing:
            ranges.append((hparams["voicing_db_min"], hparams["voicing_db_max"]))
            clamps.append((hparams["voicing_db_min"], 0.0))
        if self.predict_tension:
            ranges.append((hparams["tension_logit_min"], hparams["tension_logit_max"]))
            clamps.append((hparams["tension_logit_min"], hparams["tension_logit_max"]))
        variances_hparams = hparams["variances_prediction_args"]
        total_repeat_bins = variances_hparams["total_repeat_bins"]
        assert (
            total_repeat_bins % len(self.variance_prediction_list) == 0
        ), f"Total number of repeat bins must be divisible by number of variance parameters ({len(self.variance_prediction_list)})."
        repeat_bins = total_repeat_bins // len(self.variance_prediction_list)
        backbone_type = compat.get_backbone_type(hparams, nested_config=variances_hparams)
        backbone_args = compat.get_backbone_args(variances_hparams, backbone_type=backbone_type)
        kwargs = filter_kwargs(
            {
                "ranges": ranges,
                "clamps": clamps,
                "repeat_bins": repeat_bins,
                "timesteps": hparams.get("timesteps"),
                "time_scale_factor": hparams.get("time_scale_factor"),
                "backbone_type": backbone_type,
                "backbone_args": backbone_args,
            },
            cls,
        )
        return cls(**kwargs)

    def collect_variance_inputs(self, **kwargs) -> list:
        return [kwargs.get(name) for name in self.variance_prediction_list]

    def collect_variance_outputs(self, variances: (list | tuple)) -> dict:
        return {name: pred for name, pred in zip(self.variance_prediction_list, variances)}
