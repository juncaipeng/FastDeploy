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
import sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class Resp:
    is_valid: bool = False

    req_id: str = None
    max_dec_len: int = None
    min_dec_len: int = None
    max_send_idx: int = None

    input_token_num: int = None
    output_token_num: int = None
    is_end: bool = None

    send_req_time: float = None
    first_token_end2end_time: float = None
    all_token_end2end_time: float = None
    first_token_infer_time: float = None
    all_token_infer_time: float = None

    http_received_cost_time: float = 0
    infer_received_cost_time: float = 0
    tokenizer_encode_cost_time: float = 0
    tokenizer_decode_cost_time: float = 0
    preprocess_cost_time: float = 0
    pending_cost_time: float = 0
    get_image_cost_time: float = 0
    process_image_cost_time: float = 0

    input_text: str = None
    output_list: list = field(default_factory=list)

    error_msg: str = ""
    exception_msg: str = ""

    def auto_set_valid(self):
        self.is_valid = True
        names = ["req_id", "max_dec_len", "min_dec_len", "max_send_idx", "is_end",
                 "output_token_num", "send_req_time", "first_token_end2end_time",
                 "all_token_end2end_time", "first_token_infer_time", "all_token_infer_time"]
        for name in names:
            if getattr(self, name) is None:
                self.is_valid = False
        if self.error_msg != "" or self.exception_msg != "":
            self.is_valid = False

    def is_error(self) -> bool:
        return self.error_msg != ""

    def is_exception(self) -> bool:
        return self.exception_msg != ""


def str_to_datetime(date_string):
    if "." in date_string:
        return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S.%f")
    else:
        return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")

def datetime_diff(datetime_start, datetime_end):
    if isinstance(datetime_start, str):
        datetime_start = str_to_datetime(datetime_start)
    if isinstance(datetime_end, str):
        datetime_end = str_to_datetime(datetime_end)
    if datetime_end > datetime_start:
        cost = datetime_end - datetime_start
    else:
        cost = datetime_start - datetime_end
    return cost.total_seconds()

def pp_print(name, input_list):
    out_str = f"{name:<35}"
    for item in input_list:
        out_str += f"{item:<15}"
    print(out_str)

def pp_print_md(name, lst):
    info = f"| {name:<35} |"
    for i in lst:
        info += f" {i:<15} |"

    info = f"| {name:<35} | "
    print(info)


def collect_response(input_path):
    result_dict = {}
    start_time = None
    end_time = None
    log_step = 100000

    print("\nstart read and collect response...")
    with open(input_path, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            try:
                item = json.loads(line.rstrip('\n'))
            except Exception as e:
                print(f"error when parse line. idx: {idx}, line: {line}  error:{e}")

            item_type = item['type']
            assert item_type in ["request", "response", "error", "exception"]

            req_id = item['content']['req_id']
            if req_id in result_dict:
                resp = result_dict[req_id]
                if resp.is_valid:
                    print("error: the req_id is already in result_dict")
                    continue
            else:
                resp = Resp(req_id=req_id)
                result_dict[req_id] = resp

            if item_type == "request":
                resp.max_dec_len = item['content']["max_dec_len"]
                resp.min_dec_len = item['content']["min_dec_len"]
                resp.input_text = item['content']["text"]
                resp.send_req_time = str_to_datetime(item["now_time"])
            elif item_type == "response":
                content = item['content']
                if content["send_idx"] == 0:
                    resp.input_token_num = content.get("input_ids_len", 0)
                    if content.get("http_received_time"):
                        resp.http_received_cost_time = datetime_diff(resp.send_req_time, content.get("http_received_time"))
                    if content.get("preprocess_start_time"):
                        resp.infer_received_cost_time = datetime_diff(resp.send_req_time, content.get("preprocess_start_time"))

                    resp.first_token_infer_time = content["inference_time_cost"]
                    if content.get("preprocess_start_time") and content.get("preprocess_end_time"):
                        resp.preprocess_cost_time = datetime_diff(content.get("preprocess_start_time"),
                                                                    content.get("preprocess_end_time"))
                    if content.get("preprocess_end_time") and content.get("schedule_start_time"):
                        resp.pending_cost_time = datetime_diff(content.get("preprocess_end_time"),
                                                                    content.get("schedule_start_time"))
                    resp.get_image_cost_time = content.get("get_image_cost_time", 0)
                    resp.process_image_cost_time = content.get("process_image_cost_time", 0)
                    resp.tokenizer_encode_cost_time = content.get("tokenizer_encode_cost_time", 0)
                    resp.first_token_end2end_time = datetime_diff(resp.send_req_time, item["now_time"])
                if content["is_end"] == 1:
                    resp.is_end = True
                    resp.max_send_idx = content["send_idx"]
                    resp.output_token_num = content["tokens_all_num"]
                    resp.all_token_end2end_time = datetime_diff(resp.send_req_time, item["now_time"])
                    resp.all_token_infer_time = content["inference_time_cost"]
                    resp.auto_set_valid()
                resp.output_list.append({'idx': int(content['send_idx']), 'token':content['token']})
                resp.tokenizer_decode_cost_time += content.get("tokenizer_decode_cost_time", 0)
            elif item_type == "error":
                resp.error_msg += item['content']["error_msg"]
            elif item_type == "exception":
                resp.exception_msg += item['content']["exception_msg"]

            now_time = str_to_datetime(item["now_time"])
            if start_time is None:
                start_time = resp.send_req_time
            if end_time is None:
                end_time = now_time
            elif end_time < now_time:
                end_time = now_time

            if idx % log_step == 0:
                print(f"read {idx+1} chunks", end=', ', flush=True)

    result_list = result_dict.values()
    cost_time = datetime_diff(start_time, end_time)
    print(f"\nstart_time: {start_time}, end_time: {end_time}, "
          f"cost_time: {cost_time}, result_list_num: {len(result_list)}")
    return result_list, cost_time

def save_output_text(result_list, input_path):
    output_path = input_path.replace(".jsonl", "-out_msg.jsonl")
    with open(output_path, "w", encoding='utf-8') as out_file:
        for result in result_list:
            if result.is_valid:
                output_list = sorted(result.output_list, key=lambda d: d['idx'])
                output_text = ""
                for i in output_list:
                    output_text += i['token']
                #dict_obj = {'req_id': result.req_id, 'input_text': result.input_text, 'output_text': output_text}
                dict_obj = {'input_text': result.input_text, 'output_text': output_text}
                out_file.write(json.dumps(dict_obj, ensure_ascii=False)   + "\n")
    print(f"output save in {output_path}")


def stats_and_percentiles(lst, round_bit=3, multi=1):
    lst = [item * multi for item in lst]
    num = len(lst)
    max_val = round(max(lst), round_bit)
    min_val = round(min(lst), round_bit)
    avg_val = round(sum(lst) / len(lst), round_bit)

    pct_50, pct_80, pct_95, pct_99 = np.percentile(lst, [50, 80, 95, 99])
    pct_50 = round(pct_50, round_bit)
    pct_80 = round(pct_80, round_bit)
    pct_95 = round(pct_95, round_bit)
    pct_99 = round(pct_99, round_bit)

    return {"num": num, "max": max_val, "min": min_val, "avg": avg_val,
        "pct_50": pct_50, "pct_80": pct_80, "pct_95": pct_95, "pct_99": pct_99}

def analyse_single_key(result_list, key_name, round_bit=2, multi=1):
    key_list = []
    for resp in result_list:
        if not resp.is_valid:
            continue
        key_list.append(resp.__dict__[key_name])

    return stats_and_percentiles(key_list, round_bit, multi)

def analyse_response(result_list, cost_time):
    print("\nstart anaylse response...")
    valid_resp_num = 0
    error_num = 0
    exception_num = 0
    for resp in result_list:
        if resp.is_valid:
            valid_resp_num += 1
        elif resp.is_error():
            error_num += 1
            print(f"error resp: {resp}")
        elif resp.is_exception():
            exception_num += 1
            print(f"exception resp: {resp}")

    print(f"total response num: {len(result_list)}, valid response num: {valid_resp_num}, "
          f"error_num: {error_num}, exception_num: {exception_num}")
    print(f"qps: {round(valid_resp_num / cost_time, 2)} \n")

    info_list = [{'key': 'output_token_num', 'multi': 1, 'msg': '生成token数'},
                 {'key': 'first_token_infer_time', 'multi': 1000, 'msg': '首token推理耗时(ms)'},
                 {'key': 'all_token_infer_time', 'multi': 1000, 'msg': '整句推理耗时(ms)'},
                 {'key': 'first_token_end2end_time', 'multi': 1000, 'msg': '首token用户侧耗时(ms)'},
                 {'key': 'all_token_end2end_time', 'multi': 1000, 'msg': '整句用户侧耗时(ms)'},
                 {'key': 'infer_received_cost_time', 'multi': 1000, 'msg': '推理收到请求耗时(ms)'},
                 {'key': 'http_received_cost_time', 'multi': 1000, 'msg': 'http收到请求耗时(ms)'},
                 {'key': 'preprocess_cost_time', 'multi': 1000, 'msg': '预处理耗时(ms)'},
                 {'key': 'pending_cost_time', 'multi': 1000, 'msg': '缓存等待推理耗时(ms)'},
                ]
    print("| 指标  | 样本数 | 最大 | 最小 | 平均 | 50% | 80% | 95% | 99% |")
    print("| ---- | ---- | ---- | ----| ---- | ---- | ---- | ---- | ---- |")
    for info in info_list:
        out = analyse_single_key(result_list, info['key'], multi=info['multi'])
        print(f"| {info['msg']:<35} | {out['num']:<15} | {out['max']:<15} | {out['min']:<15} | {out['avg']:<15} "
              f"| {out['pct_50']:<15} | {out['pct_80']:<15} | {out['pct_95']:<15} | {out['pct_99']:<15} |")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="the jsonl result file generated by run_benchmark_xx.py")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(f"input_path: {args.input_path}")

    result_list, cost_time = collect_response(args.input_path)
    analyse_response(result_list, cost_time)
    save_output_text(result_list, args.input_path)
