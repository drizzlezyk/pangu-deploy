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
from flask import Flask, request

from pcl_pangu.context import set_context
from pcl_pangu.model import alpha


APP = Flask(__name__)
set_context(backend='mindspore')

config = alpha.model_config_npu(model='2B6')
model, config_model = alpha.load_model(config, input='四川的省会是?')


@APP.route('/health', methods=['GET'])
def health_func():
    return json.dumps({'health': 'true'}, indent=4)


@APP.route('/infer/text', methods=['POST'])
def inference_text():
    inputs = request.json
    question = inputs['question']
    result = alpha.inference_with_model(config, model, config_model, input=question)
    res_data = {
        "result": result
    }
    return json.dumps(res_data, indent=4)


if __name__ == "__main__":
    APP.run(host='0.0.0.0', port='8080')
