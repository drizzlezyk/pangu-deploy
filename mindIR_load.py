import mindspore as ms
from mindvision.classification.models import resnet50
from mindspore import Tensor
import numpy as np
import mindspore.nn as nn

graph = ms.load('./models/resnet50_224.mindir')

# net = resnet50()
input = Tensor(np.random.uniform(0.0, 1.0, size=[32, 1, 32, 32]).astype(np.float32))
# ms.export(net, input, file_name="net", file_format="MINDIR")
net = nn.GraphCell(graph)
output = net(input)
print(output)