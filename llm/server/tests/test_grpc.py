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

import json
import os
import queue
import sys
import uuid
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *


class OutputData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def triton_callback(output_data, result, error):
    if error:
        output_data._completed_requests.put(error)
    else:
        output_data._completed_requests.put(result)

def test_base(grpc_url, input_data, test_iters=1, log_level="simple"):
    if log_level not in ["simple", "verbose"]:
        raise ValueError("log_level must be simple or verbose")

    model_name = "model"
    inputs = [grpcclient.InferInput("IN", [1], np_to_triton_dtype(np.object_))]
    outputs = [grpcclient.InferRequestedOutput("OUT")]
    output_data = OutputData()

    with grpcclient.InferenceServerClient(url=grpc_url, verbose=False) as triton_client:
        triton_client.start_stream(callback=partial(triton_callback, output_data))
        for i in range(test_iters):
            input_data = json.dumps([input_data])
            inputs[0].set_data_from_numpy(np.array([input_data], dtype=np.object_))

            triton_client.async_stream_infer(model_name=model_name,
                                             inputs=inputs,
                                             request_id="{}".format(i),
                                             outputs=outputs)

            print("output_data:")
            while True:
                output_item = output_data._completed_requests.get(timeout=10)
                if type(output_item) == InferenceServerException:
                    print(f"Exception: status is {output_item.status()}, msg is {output_item.message()}")
                    break
                else:
                    result = json.loads(output_item.as_numpy("OUT")[0])
                    result = result[0] if isinstance(result, list) else result
                    if result.get("is_end") == 1 or result.get("error_msg"):
                        print(f"\n {result} \n")
                        break
                    else:
                        if log_level == "simple":
                            print(result['token'] if 'token' in result else result['token_ids'][0], end="")
                        else:
                            print(result)

if __name__ == "__main__":
    input_data = {
                "req_id": 0,
                "text": "hello",
                "seq_len": 1024,
                "min_dec_len": 2,
                "penalty_score": 1.0,
                "temperature": 0.8,
                "topp": 0.8,
                "frequency_score": 0.1,
                "presence_score": 0.0
                }
    grpc_url = "0.0.0.0:8891"
    test_base(grpc_url=grpc_url, input_data=input_data)
