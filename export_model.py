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

import os
import argparse
import pickle

import paddle

from models.resnet import ResNet_PaDiM

def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    parser.add_argument("--depth", type=int, default=18, help="resnet depth")
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the exported model',
        type=str,
        default='./output')
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for export',
        type=str,
        default=None)
    parser.add_argument(
        '--img_size',
        help='The img size for export',
        type=int,
        default=224)

    return parser.parse_args()


def main(args):

    # build model
    model = ResNet_PaDiM(depth=args.depth, pretrained=True)

    if args.model_path is not None:
        state = paddle.load(args.model_path)
        model.model.set_dict(state["params"])
        model.distribution = state["distribution"]

    save_path = os.path.join(args.save_dir, 'distribution')
    with open(save_path, 'wb') as f:
            pickle.dump(model.distribution, f)

    shape = [-1, 3, args.img_size, args.img_size]
    new_net = model
    new_net.eval()
    new_net = paddle.jit.to_static(
        new_net,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    save_path = os.path.join(args.save_dir, 'model')
    paddle.jit.save(new_net, save_path)
    print(f'Model is saved in {args.save_dir}.')


if __name__ == '__main__':
    args = parse_args()
    main(args)