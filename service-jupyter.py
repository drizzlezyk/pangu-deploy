import json
from pcl_pangu.context import set_context
from pcl_pangu.model import alpha


MODEL_PATH = '/home/model/checkpoint_file'
OUTPUT_FILE = '/home/model/output/result.txt'
STRATEGY_CKPT_FILE = '/home/model/strategy/pangu_alpha_2.6B_512_ckpt_strategy.zip'
VOCAB_FILE = '/home/model/tokenizer'

input = {
    "question": '四川的省会是哪里'
}
question = input['question']

config = alpha.model_config_npu(model='2B6',
                                load_ckpt_local_path=MODEL_PATH,
                                tokenizer_path=VOCAB_FILE,
                                strategy_load_ckpt_path=STRATEGY_CKPT_FILE,
                                inputs=question,
                                output_file=OUTPUT_FILE,
                                oneCardInference=True)
set_context(backend='mindspore')
print('start inference')
result = alpha.inference(config)

print(result)
