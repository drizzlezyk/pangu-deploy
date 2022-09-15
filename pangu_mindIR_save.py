import os
os.system('pip install loguru')
from pcl_pangu.context import set_context
from pcl_pangu.model import alpha

# parser = argparse.ArgumentParser()
# parser.add_argument("--model_path", type=str, default='obs://pangu/models/26b/checkpiont_file/',
#                     help="the path of model path")
# parser.add_argument("--input_str", type=str, default='四川的省会是哪里',
#                     help="question")
# parser.add_argument("--model", type=str, default='2B6',
#                     help="model")
# parser.add_argument("--mindir_path", type=str, default='obs://pangu/models/26b/output/',
#                     help="mindir path")
# parser.add_argument("--output_file", type=str, default='obs://pangu/models/26b/output/result.txt',
#                     help="output_file path")
# args_opt = parser.parse_args()


set_context(backend='mindspore')

config = alpha.model_config_npu(model='2B6')

print('start inference')


# output_file = os.path.join(args_opt.output_file, 'result.txt')
alpha.inference(config,
                input="四川的省会是哪里",
                oneCardInference=True)
