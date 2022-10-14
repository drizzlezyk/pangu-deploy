from flask import Flask, request
import json

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
    input = request.json
    question = input['question']
    result = alpha.inference_with_model(config, model, config_model, input=question)
    res_data = {
        "result": result
    }
    return json.dumps(res_data, indent=4)


if __name__ == "__main__":
    APP.run(host='0.0.0.0', port='8080')
