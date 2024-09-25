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
import argparse
import json
import os
import sys
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial

import httpx
import numpy as np

import tritonclient.grpc as grpcclient
from tritonclient.utils import *

from logger import get_logger

def _http_send_worker(args, req_dict, result_queue):
    is_error_resp = False
    headers = {'Content-Type': 'application/json'}
    with httpx.stream("POST", args.url, headers=headers, timeout=args.timeout, json=req_dict) as r:
        for chunk in r.iter_lines():
            resp = json.loads(chunk)
            if resp.get("error_msg") or resp.get("error_code"):
                is_error_resp = True
                content = {"error_msg": resp.get("error_msg"), "req_id": req_dict.get("req_id")}
                result_queue.put({"type": "error", "now_time": str(datetime.now()), "content": content})
            else:
                result_queue.put({"type": "response", "now_time": str(datetime.now()), "content": resp})
    return is_error_resp

def _grpc_send_worker(args, req_dict, result_queue):
    class OutputData:
        def __init__(self):
            self._completed_requests = queue.Queue()

    def triton_callback(output_data, result, error):
        if error:
            output_data._completed_requests.put(error)
        else:
            output_data._completed_requests.put(result)

    model_name = "model"
    inputs = [grpcclient.InferInput("IN", [1], np_to_triton_dtype(np.object_))]
    outputs = [grpcclient.InferRequestedOutput("OUT")]
    output_data = OutputData()
    is_error_resp = False

    with grpcclient.InferenceServerClient(url=args.url, verbose=False) as triton_client:
        triton_client.start_stream(callback=partial(triton_callback, output_data))

        input_data = json.dumps([req_dict])
        inputs[0].set_data_from_numpy(np.array([input_data], dtype=np.object_))

        triton_client.async_stream_infer(model_name=model_name,
                                            inputs=inputs,
                                            request_id=str(uuid.uuid4()),
                                            outputs=outputs)

        while True:
            output_item = output_data._completed_requests.get(timeout=args.timeout)
            if type(output_item) == InferenceServerException:
                is_error_resp = True
                error_msg = f"Exception: status is {output_item.status()}, msg is {output_item.message()}"
                content = {"error_msg": error_msg, "req_id": req_dict.get("req_id")}
                result_queue.put({"type": "error", "now_time": str(datetime.now()), "content": content})
            else:
                result = json.loads(output_item.as_numpy("OUT")[0])
                result = result[0] if isinstance(result, list) else result
                result_queue.put({"type": "response", "now_time": str(datetime.now()), "content": result})
                if result.get("is_end") == 1:
                    break
    return is_error_resp

def send_worker(args, data_queue, result_queue, worker_idx, logger):
    """
    send requests and put response into result_queue
    """
    logger.info(f"[send_worker {worker_idx}] start...")

    cur_idx = 0
    exception_num = 0
    exception_threshold = 10
    error_resp_num = 0
    log_step = 10

    while not data_queue.empty():
        # read data
        try:
            input_data = data_queue.get(timeout=3)
            remaining_num = data_queue.qsize()
            cur_idx += 1
        except queue.Empty:
            logger.info(f"[send_worker {worker_idx}] data queue is empty")
            break
        except Exception as e:
            exception_num += 1
            logger.error(f"[send_worker {worker_idx}][fd_error] fetch data error: {e}")
            continue

        result_queue.put({"type": "request", "now_time": str(datetime.now()), "content": input_data})

        # send request
        try:
            if args.api_type == 'http':
                is_error_resp = _http_send_worker(args, input_data, result_queue)
            elif args.api_type == 'grpc':
                is_error_resp = _grpc_send_worker(args, input_data, result_queue)
            error_resp_num += 1 if is_error_resp else 0
        except Exception as e:
            exception_num += 1
            content = {"exception_msg": str(e), "req_id": input_data.get("req_id")}
            result_queue.put({"type": "exception", "now_time": str(datetime.now()), "content": content})
            if exception_num > exception_threshold:
                logger.error(f"[send_worker {worker_idx}] exception num ({exception_num}) exceeds "
                             f"threshold, exit")
                break

        # log
        if cur_idx % log_step == 1:
            logger.info(f"[send_worker {worker_idx}] processed_num: {cur_idx}, exception_num: {exception_num}, "
                        f"error_resp_num: {error_resp_num}, data queue remaining ({remaining_num}) tasks")

    logger.info(f"[send_worker {worker_idx}] exit, processed_num: {cur_idx}, exception_num: {exception_num}, "
                f"error_resp_num: {error_resp_num}")

def save_worker(result_path, result_queue, logger, timeout=50, log_step=10000):
    """
    save the result to file
    """
    logger.info("[save_worker] start...")
    num = 0
    with open(result_path, "w", encoding='utf-8') as out_file:
        while True:
            try:
                res_chunk = result_queue.get(timeout=timeout)
            except queue.Empty:
                logger.info("[save_worker] result queue is empty")
                break
            except Exception as e:
                logger.error(f"[save_worker] Error retrieving data from queue: {e}")
                break

            json_str = json.dumps(res_chunk, ensure_ascii=False)
            out_file.write(json_str + "\n")
            num += 1
            if num % log_step == 0:
                logger.info(f"[save_worker] process {num} response chunks")

    logger.info("[save_worker] exit")

def prepare_data(data_path, data_num, benchmark=True, stream=True, timeout=180):
    """
    prepare data
    """
    '''
    data_queue = queue.Queue()
    with open(data_path, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            raw_data = json.loads(line.rstrip('\n'))
            input_data = {
                "text": raw_data['text_before_process'],
                "max_dec_len": raw_data["max_dec_len"],
                "min_dec_len": raw_data["min_dec_len"],
                "topp": raw_data["topp"],
                "temperature": raw_data["temperature"],
                "frequency_score": raw_data["frequency_score"],
                "penalty_score": raw_data["penalty_score"],
                "presence_score": raw_data["presence_score"],
                "req_id": str(uuid.uuid4()),
                "stream": stream,
                "benchmark": benchmark,
                "timeout": timeout,
            }
            if raw_data["history_QA"] != []:
                input_data["history_qa"] = raw_data["history_QA"]

            data_queue.put(input_data)
            if data_num > 0 and idx + 1 >= data_num:
                break
    return data_queue
    '''
    data_queue = queue.Queue()
    with open(data_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
        dataset = [data for data in dataset if len(data['conversations']) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [(data['conversations'][0]['value'],
                    data['conversations'][1]['value']) for data in dataset]
        prompts = [prompt for prompt, _ in dataset]

        for idx, text in enumerate(prompts):
            input_data = {
                "text": text,
                "max_dec_len": 1024,
                "min_dec_len": 1,
                "topp": 0,
                "temperature": 1,
                "req_id": str(uuid.uuid4()),
                "stream": stream,
                "benchmark": benchmark,
                "timeout": timeout,
            }
            data_queue.put(input_data)
            if data_num > 0 and idx + 1 >= data_num:
                break
    return data_queue


def parse_args():
    """
    parse the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_type", default="http", type=str, help="grpc or http api")
    parser.add_argument("--url", default="http://0.0.0.0:8894/v1/chat/completions", type=str, help="the url for model server")
    parser.add_argument("--data_path", default="data.jsonl", type=str, help="the path of data with jsonl format")
    parser.add_argument("--data_num", default=-1, type=int, help="-1 means all data")
    parser.add_argument("--timeout", default=180, type=int, help="timeout for waiting repsonse")
    parser.add_argument("--worker_num", default=1, type=int, help="the number of worker_num for sending requests")
    parser.add_argument("--tag", default="test", type=str, help="identify the test case")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # prepare
    data_queue = prepare_data(args.data_path, args.data_num, benchmark=False, timeout=args.timeout)
    if args.data_num < 0:
        args.data_num = data_queue.qsize()
    print(f"data_queue size: {data_queue.qsize()}")

    test_tag = f"{args.tag}-{args.api_type}-wk{args.worker_num}-dn{args.data_num}"
    logger = get_logger('benchmark', f'{test_tag}.log')
    logger.info(f"args: {args}")
    logger.info(f"test_tag: {test_tag}")

    result_path = f"output/{test_tag}.jsonl"
    if os.path.exists(result_path):
        logger.error(f"result file ({result_path}) already exists, overwrite it")
    if not os.path.exists("output/"):
        os.makedirs("output/")
    logger.info(f"result_path: {result_path}")

    # save worker
    worker_list = []
    result_queue = queue.Queue()
    worker = threading.Thread(target=save_worker, args=(result_path, result_queue, logger, 20))
    worker.start()
    worker_list.append(worker)

    # send worker
    tic = time.time()
    for idx in range(args.worker_num):
        worker = threading.Thread(target=send_worker, args=(args, data_queue, result_queue, idx, logger))
        worker.start()
        worker_list.append(worker)
    for worker in worker_list:
        worker.join()

    toc = time.time()
    logger.info(f'Done, cost time: {round(toc - tic, 2)}s')
