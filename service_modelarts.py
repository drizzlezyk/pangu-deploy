import json
from pcl_pangu.context import set_context
from pcl_pangu.model import alpha
import os

# MODEL_PATH = '/home/model/checkpoint_file'
# OUTPUT_FILE = '/home/model/output/result.txt'
# STRATEGY_CKPT_FILE = '/home/model/strategy/pangu_alpha_2.6B_512_ckpt_strategy.zip'
# VOCAB_FILE = '/home/model/tokenizer'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='obs://pangu/models/26b/checkpiont_file/',
                    help="the path of model path")
parser.add_argument("--strategy_path", type=str, default='四川的省会是哪里',
                    help="question")
parser.add_argument("--model", type=str, default='2B6',
                    help="model")
parser.add_argument("--vocab_file", type=str, default='obs://pangu/models/26b/output/',
                    help="mindir path")
parser.add_argument("--output_file", type=str, default='obs://pangu/models/26b/output/result.txt',
                    help="output_file path")
parser.add_argument("--ckpt_output_path", type=str, default='obs://pangu/models/26b/output/result.txt',
                    help="output_file path")
args_opt = parser.parse_args()

MODEL_PATH = args_opt.model_path
OUTPUT_FILE = args_opt.output_file
STRATEGY_CKPT_FILE = args_opt.strategy_path
VOCAB_FILE = args_opt.vocab_file

ckpt_output_file = os.path.join(args_opt.ckpt_output_path, 'pangu_2B6.ckpt')


config = alpha.model_config_npu(model='2B6',
                                load_ckpt_local_path=MODEL_PATH,
                                tokenizer_path=VOCAB_FILE,
                                strategy_load_ckpt_path=STRATEGY_CKPT_FILE,
                                inputs='',
                                output_file=OUTPUT_FILE,
                                oneCardInference=True,
                                use_past="true",
                                seq_length=13000,
                                batch_size=4)
set_context(backend='mindspore')
model, config_load = alpha.get_model(config)

from mindspore import save_checkpoint
save_checkpoint(model, ckpt_output_file)



# def inference_text():
#     question = "四川的省会是哪里"
#
#     config = alpha.model_config_npu(model='2B6',
#                                     load_ckpt_local_path=MODEL_PATH,
#                                     tokenizer_path=VOCAB_FILE,
#                                     strategy_load_ckpt_path=STRATEGY_CKPT_FILE,
#                                     inputs=question,
#                                     output_file=OUTPUT_FILE,
#                                     oneCardInference=True)
#     print('start inference')
#     result = alpha.inference(config, model, config_load)
#     res_data = {
#         "result": result
#     }
#     print(result)
#     return json.dumps(res_data, indent=4)


# inference_text()
