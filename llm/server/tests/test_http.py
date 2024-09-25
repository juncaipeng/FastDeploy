import argparse
import json
import uuid
from datetime import datetime

import httpx
import requests


def http_no_stream(url, data):
    print("--http_no_stream--")
    headers = {'Content-Type': 'application/json'}
    #resp = httpx.post(url=url, headers=headers, timeout=300, json=data)
    resp = requests.post(url, headers=headers, json=data)
    print(resp.text)

def http_stream(url, data, show_chunk=False):
    print("--http_stream--")
    headers = {'Content-Type': 'application/json'}
    data = data.copy()
    data["stream"] = True
    #with httpx.stream("POST", url, headers=headers, timeout=300,json=data) as r:
    with requests.post(url, json=data, headers=headers, timeout=300, stream=True) as r:
        result = ""
        for chunk in r.iter_lines():
            if chunk:
                resp = json.loads(chunk)
                if resp["error_msg"] != "" or resp["error_code"] != 0:
                    print(resp)
                    return
                else:
                    result += resp.get("token", "")
                    if show_chunk:
                        print(resp)
        print(f"Result: {result}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--http_host", default="10.95.147.146", type=str, help="host to the http server")
    parser.add_argument("--http_port", default=8894, type=int, help="port to the http server")
    parser.add_argument("-o", "--open_source_model", action="store_true", help="test eb_model or open_source_model")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    url = f"http://{args.http_host}:{args.http_port}/v1/chat/completions"
    print(f"url: {url}")

    print("\n\n=====single round test=====")
    data = {
        "req_id": str(uuid.uuid4()),
        "text": "hello",
        "max_dec_len": 1024,
        "min_dec_len": 2,
        "penalty_score": 1.0,
        "temperature": 0.8,
        "topp": 0,
        "frequency_score": 0.1,
        "presence_score": 0.0,
        "timeout": 600,
        "benchmark": True,
        }
    http_no_stream(url, data)
    http_stream(url, data)

    print("\n\n=====single round test with default params=====")
    data = {"text": "hello"}
    http_no_stream(url, data)
    http_stream(url, data)


    print("\n\n=====test error case=====")
    data = {
        "req_id": str(uuid.uuid4()),
        "text": "hello",
        "max_dec_len": 1024,
        "min_dec_len": 2,
        "penalty_score": 1.0,
        "temperature": 0.8,
        "topp": 2,           # topp should be in [0, 1]
        "frequency_score": 0.1,
        "presence_score": 0.0,
        "history_QA": [],
        "benchmark": True,
        "timeout": 600}
    http_no_stream(url, data)
    http_stream(url, data)
