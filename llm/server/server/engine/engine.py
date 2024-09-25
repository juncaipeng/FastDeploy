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

import multiprocessing
import os
import signal
import subprocess
import time
import uuid
import weakref
from datetime import datetime
from multiprocessing import shared_memory

import numpy as np
from server.engine.resource_manager import ResourceManager
from server.engine.task_queue_manager import (TaskQueueManager,
                                              launch_task_queue_manager)
from server.engine.out_processor import OutProcessor
from server.utils import model_server_logger


class Engine(object):
    """
    Engine Class
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.resource_manager = ResourceManager(self.cfg)
        self.out_processor = OutProcessor(self.cfg)
        self.out_processor.set_resource_manager(self.resource_manager)

        self._init_engine_flags()

        self.tqm_proc = self._start_task_queue_manager()
        self.task_queue_manager = TaskQueueManager(mp_num=self.cfg.mp_num, port=self.cfg.infer_queue_port)

        start_time = time.time()
        self.infer_proc = self._start_infer_process()
        model_server_logger.info("Waitting infer processes ready...")
        while not self._infer_processes_ready():
            time.sleep(1)
        model_server_logger.info("Infer processes are launched with {} seconds.".format(time.time() - start_time))

        self._finalizer = weakref.finalize(self, self._exit_sub_services)

    def insert_tasks(self, tasks):
        """
        insert tasks to the engine

        Args:
            tasks: list of tasks

        Returns:
            return: True if success, False otherwise
        """
        if not isinstance(tasks, list):
            tasks = [tasks]

        for item in tasks:
            item["schedule_start_time"] = datetime.now()

        available_batch = np.sum(self.resource_manager.stop_flags)
        if len(tasks) > available_batch:
            model_server_logger.error("Inserting batch:{} exceeds the available batch:{}.".format(
                len(tasks), available_batch))
            model_server_logger.error("The exceeded part will be ignored!")
            tasks = tasks[:available_batch]

        for i in range(len(tasks)):
            req_id = tasks[i]["req_id"]
            input_token_num = len(tasks[i]["input_ids"])
            if input_token_num >= self.cfg.max_seq_len - 1:
                model_server_logger.warning(f"{req_id}: Input length:{input_token_num}, exceed the limits.")
                tasks[i]["input_ids"] = tasks[i]["input_ids"][:self.cfg.max_seq_len - 1]
            if "seq_len" in tasks[i] and "max_dec_len" not in tasks[i]:
                tasks[i]["max_dec_len"] = tasks[i]["seq_len"]

            # max_dec_len + input_token_num > MAX_SEQ_LEN
            if input_token_num + tasks[i]["max_dec_len"] > self.cfg.max_seq_len:
                tasks[i]["max_dec_len"] = self.cfg.max_seq_len - input_token_num
                model_server_logger.warning("Force max_dec_len to be {} for req_id={}.".format(
                    tasks[i]["max_dec_len"], tasks[i]["req_id"]))

            # min_dec_len + input_token_num > MAX_SEQ_LEN
            if input_token_num + tasks[i]["min_dec_len"] > self.cfg.max_seq_len:
                tasks[i]["min_dec_len"] = self.cfg.max_seq_len - input_token_num
                model_server_logger.warning("Force min_dec_len to be {} for req_id={}.".format(
                    tasks[i]["min_dec_len"], tasks[i]["req_id"]))

        tasks = self.resource_manager.allocate_resources_for_new_tasks(tasks)
        if not tasks:
            return False

        req_ids = [t["req_id"] for t in tasks]
        model_server_logger.info(f"Tasks are sent to engine, req_ids={req_ids}")
        self.task_queue_manager.put((tasks, self.resource_manager.real_bsz))
        return True

    def task_is_finished(self, index):
        """
        judge if the task is finished

        Args:
            index: task index

        Returns:
            return: True if finished, False otherwise
        """
        assert index < len(self.resource_manager.stop_flags)
        return self.resource_manager.stop_flags[index]

    def is_queue_empty(self):
        """
        judge if the queue is empty

        Returns:
            return: True if empty, False otherwise
        """
        return self.task_queue_manager.empty()

    def is_resource_sufficient(self, input_token_num):
        """
        judge if the resource is sufficient

        Args:
            input_token_num: input token number

        Returns:
            return: True if sufficient, False otherwise
        """
        return self.resource_manager.is_resource_sufficient(input_token_num)

    def all_tasks_finished(self):
        """
        judge if all tasks are finished

        Returns:
            return: True if all finished, False otherwise
        """
        return np.sum(self.resource_manager.stop_flags) == len(self.resource_manager.stop_flags)

    def available_batch(self):
        """
        available batch size of the engine

        Returns:
            return: available batch size
        """
        return self.resource_manager.available_batch()

    def available_block_num(self):
        """
        available block number of the engine

        Returns:
            return: available block number
        """
        return self.resource_manager.availabel_block_num()

    def _infer_processes_ready(self):
        """
        judge if all infer processes are ready

        Returns:
            return: True if all ready, False otherwise
        """
        if np.sum(self.flag_ready_array) == self.cfg.mp_num:
            return True
        return False

    def _clear_engine_flags(self):
        """
        clear engine flags
        """
        try:
            self.shm_flag_ready.close()
            self.shm_flag_ready.unlink()
            self.shm_flag_has_block_step.close()
            self.shm_flag_has_block_step.unlink()
        except:
            pass

    def _init_engine_flags(self):
        """
        Initialize shared memory to indicate engine status
        """
        flag_array = np.zeros([self.cfg.mp_num], dtype=np.int32)
        try:
            tmp = shared_memory.SharedMemory(
                create=False, size=flag_array.nbytes, name=self.cfg.get_unique_name("shm_flag_infer_ready")
            )
            tmp.close()
            tmp.unlink()
        except:
            pass
        self.shm_flag_ready = shared_memory.SharedMemory(
            create=True, size=flag_array.nbytes, name=self.cfg.get_unique_name("shm_flag_infer_ready")
        )
        self.flag_ready_array = np.ndarray(
            flag_array.shape, dtype=flag_array.dtype, buffer=self.shm_flag_ready.buf
        )
        self.flag_ready_array[:] = 0

        # broadcast flag for engine
        broadcast_flag_array = np.zeros([1], dtype=np.int32)
        try:
            tmp = shared_memory.SharedMemory(
                create=False,
                size=broadcast_flag_array.nbytes,
                name=self.cfg.get_unique_name("shm_pd_infer_flag_broadcast"),
            )
            tmp.close()
            tmp.unlink()
        except:
            pass
        self.shm_flag_broadcast = shared_memory.SharedMemory(
            create=True, size=broadcast_flag_array.nbytes, name=self.cfg.get_unique_name("shm_pd_infer_flag_broadcast")
        )
        self.flag_broadcast_array = np.ndarray(
            broadcast_flag_array.shape,
            dtype=broadcast_flag_array.dtype,
            buffer=self.shm_flag_broadcast.buf,
        )
        self.flag_broadcast_array[0] = 0

        has_block_step_flag_array = np.zeros([1], dtype=np.int32)
        try:
            tmp = shared_memory.SharedMemory(
                create=False,
                size=has_block_step_flag_array.nbytes,
                name=self.cfg.get_unique_name("shm_flag_has_block_step"))
            tmp.close()
            tmp.unlink()
        except:
            pass
        self.shm_flag_has_block_step = shared_memory.SharedMemory(
            create=True,
            size=has_block_step_flag_array.nbytes,
            name=self.cfg.get_unique_name("shm_flag_has_block_step"))
        self.flag_has_block_step_array = np.ndarray(
            has_block_step_flag_array.shape,
            dtype=has_block_step_flag_array.dtype,
            buffer=self.shm_flag_has_block_step.buf)
        self.flag_has_block_step_array[:] = 0

    def _exit_sub_services(self):
        """
        exit sub services
        """
        if hasattr(self, "tqm_proc") and self.tqm_proc is not None:
            self.tqm_proc.terminate()
            self.tqm_proc.join()
        if hasattr(self, "infer_proc") and self.infer_proc is not None:
            os.killpg(self.infer_proc.pid, signal.SIGTERM)

    def _start_task_queue_manager(self):
        """
        start tasks queue service

        Returns:
            p: process handle
        """
        p = multiprocessing.Process(target=launch_task_queue_manager, args=(self.cfg.infer_queue_port, self.cfg.mp_num))
        p.start()
        if p.is_alive():
            model_server_logger.info("start tasks queue service successfully")
        else:
            error_msg = "Failed to start task queue manager, please check " \
                        "the log/task_queue_manager.log for details"
            model_server_logger.info(error_msg)
            raise Exception(error_msg)
        return p

    def _start_gpu_infer_process(self):
        """
        start gpu infer process

        Returns:
            p: process handle
        """
        current_file_path = os.path.abspath(__file__)
        current_dir_path = os.path.split(current_file_path)[0]
        pd_cmd = "python3 -m paddle.distributed.launch "
        py_script = os.path.join(current_dir_path, "infer.py")

        arguments = (f" --devices {self.cfg.device_ids} {py_script} --model_dir {self.cfg.model_dir}"
                    f" --max_batch_size {self.cfg.max_batch_size} --max_seq_len {self.cfg.max_seq_len}"
                    f" --max_dec_len {self.cfg.max_dec_len}"
                    f" --max_block_num {self.cfg.total_block_num} --block_size {self.cfg.block_size}"
                    f" --use_cache_kv_int8 {self.cfg.use_cache_kv_int8}"
                    f" --enc_dec_block_num {self.cfg.enc_dec_block_num}"
                    f" --block_ratio {self.cfg.block_ratio} --dtype {self.cfg.dtype}")
        pd_cmd = pd_cmd + arguments + " >log/launch_infer.log 2>&1"
        model_server_logger.info("Launch infer service command: {}".format(pd_cmd))
        p = subprocess.Popen(
            pd_cmd,
            shell=True,
            preexec_fn=os.setsid,
        )
        return p

    def _start_infer_process(self):
        """
        start infer process
        """
        return self._start_gpu_infer_process()
