# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
from paddle.vision.models.resnet import ResNet, BasicBlock, BottleneckBlock
from paddle.vision.models.resnet import model_urls,get_weights_path_from_url

class ResNet_PaDiM(nn.Layer):

    def __init__(self, depth=18, pretrained=True):
        super(ResNet_PaDiM, self).__init__()
        arch = 'resnet{}'.format(depth)
        Block = BottleneckBlock
        if depth < 50:
            Block = BasicBlock
        self.model = ResNet(Block, depth)
        self.distribution = None
        self.init_weight(arch, pretrained)

    def init_weight(self, arch, pretrained):
        if pretrained:
            assert arch in model_urls, "{} model do not have a pretrained model now," \
                                       " you should set pretrained=False".format(arch)
            weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                    model_urls[arch][1])

            self.model.set_dict(paddle.load(weight_path))

    def forward(self, x):
        res = []
        with paddle.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            res.append(x)
            x = self.model.layer2(x)
            res.append(x)
            x = self.model.layer3(x)
            res.append(x)
        return res