from pcl_pangu.context import set_context
from pcl_pangu.model import alpha


set_context(backend='mindspore')
config = alpha.model_config_npu(model='2B6')
model, config_model = alpha.load_model(config, input='四川的省会是?')

result = alpha.inference_with_model(model, config_model)
print(result)