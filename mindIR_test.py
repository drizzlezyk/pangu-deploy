from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
from mindspore import Tensor
import numpy as np
from mindvision.classification.models import resnet50

resnet_50 = resnet50()
# 返回模型的参数字典
param_dict = load_checkpoint("./models/resnet50_224.ckpt")
# 加载参数到网络
load_param_into_net(resnet_50, param_dict)
input = np.random.uniform(0.0, 1.0, size=[32, 1, 32, 32]).astype(np.float32)

# 以指定的名称和格式导出文件
export(resnet_50, Tensor(input), file_name='./models/resnet50_224.mindir', file_format='MINDIR')