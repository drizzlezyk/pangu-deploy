# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import json
from multiprocessing import Process
from threading import Thread
from flask import Flask, request

from gevent import monkey
from gevent.pywsgi import WSGIServer

from pcl_pangu.context import set_context
from pcl_pangu.model import alpha

APP = Flask(__name__)
monkey.patch_all(thread=False)


class MyThread(Thread):
    def __init__(self, func, args=()):
        super().__init__()
        self.func = func
        self.args = args
        self.output = None

    def run(self):
        self.output = self.func(*self.args)

    def get_result(self):
        try:
            return self.output
        except RuntimeError:
            print('No output!')


set_context(backend='mindspore')
config = alpha.model_config_npu(model='2B6')
model, config_model = alpha.load_model(config, input='四川的省会是?')


def process_result(result):
    if result.isspace():
        return "我不能理解，换个方式提问呢~"
    if result.startswith("上联") and len(result.split('\n')) > 1:
        return result.split('\n')[1]
    return result


# health check
@APP.route('/health', methods=['GET'])
def health_func():
    return json.dumps({'health': 'true'}, indent=4)


@APP.route('/infer/text', methods=['POST'])
def inference_text():
    inputs = request.json
    question = inputs['question']
    result = alpha.inference_with_model(config,
                                        model,
                                        config_model,
                                        input=question)
    result = process_result(result)
    res_data = {
        "result": result
    }
    return json.dumps(res_data, indent=4)


def run(use_multi_process):
    if not use_multi_process:
        WSGIServer(('0.0.0.0', 8080), APP).serve_forever()
    else:
        multi_server = WSGIServer(('0.0.0.0', 8080), APP)
        multi_server.start()

        def server_forever():
            multi_server.start_accepting()
            multi_server._stop_event.wait()

        for _ in range(1):
            process = Process(target=server_forever)
            process.start()


if __name__ == "__main__":
    run(True)
