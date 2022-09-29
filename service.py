from flask import Flask, request
import json
from pcl_pangu.context import set_context
from pcl_pangu.model import alpha


APP = Flask(__name__)
MODEL_PATH = '/home/model/checkpoint_file'
OUTPUT_FILE = '/home/model/output/result.txt'
STRATEGY_CKPT_FILE = '/home/model/strategy/pangu_alpha_2.6B_512_ckpt_strategy.zip'
VOCAB_FILE = '/home/model/tokenizer'

config = alpha.model_config_npu(model='2B6',
                                load_ckpt_local_path=MODEL_PATH,
                                tokenizer_path=VOCAB_FILE,
                                strategy_load_ckpt_path=STRATEGY_CKPT_FILE,
                                inputs='',
                                output_file=OUTPUT_FILE,
                                oneCardInference=True)
set_context(backend='mindspore')
model, config_load = alpha.get_model(config)


@APP.route('/health', methods=['GET'])
def health_func():
    return json.dumps({'health': 'true'}, indent=4)


@APP.route('/infer/text', methods=['POST'])
def inference_text():
    input = request.json
    question = input['question']

    config = alpha.model_config_npu(model='2B6',
                                    load_ckpt_local_path=MODEL_PATH,
                                    tokenizer_path=VOCAB_FILE,
                                    strategy_load_ckpt_path=STRATEGY_CKPT_FILE,
                                    inputs=question,
                                    output_file=OUTPUT_FILE,
                                    oneCardInference=True)
    print('start inference')
    result = alpha.inference(config, model, config_load)
    res_data = {
        "result": result
    }
    print(result)
    return json.dumps(res_data, indent=4)


if __name__ == "__main__":
    APP.run(host='0.0.0.0', port='8080')
