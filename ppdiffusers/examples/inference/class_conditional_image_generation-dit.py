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

import os
import paddle
from paddlenlp.trainer import set_seed
from ppdiffusers import DDIMScheduler, DiTPipeline

os.environ["Inference_Optimize"] = "True"
    
dtype = paddle.float16
pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", paddle_dtype=dtype)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
set_seed(42)

words = ["golden retriever"]  # class_ids [207]
class_ids = pipe.get_label_ids(words)


image = pipe(class_labels=class_ids, num_inference_steps=25).images[0]
image = pipe(class_labels=class_ids, num_inference_steps=25).images[0]
image = pipe(class_labels=class_ids, num_inference_steps=25).images[0]

    
import datetime
import time
repeat_times = 10
sum_time = 0.

for i in range(repeat_times):
    paddle.device.synchronize()
    starttime = datetime.datetime.now()
    
    image = pipe(class_labels=class_ids, num_inference_steps=25).images[0]
    
    paddle.device.synchronize()
    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    
    time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
    sum_time+=time_ms
print("The ave end to end time : ", sum_time / repeat_times, "ms")
image.save("class_conditional_image_generation-dit_last_result.png")

