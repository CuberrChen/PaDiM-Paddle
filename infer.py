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

import cv2
import os
from os import path as osp
import argparse
import pickle
import numpy as np
import datetime
import random
from PIL import Image
from random import sample
from collections import OrderedDict
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib
import matplotlib.pyplot as plt

import paddle
import paddle.nn.functional as F
from paddle.vision import transforms as T

from paddle import inference
from paddle.inference import Config, create_predictor

def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # general params
    parser = argparse.ArgumentParser("PaddleVideo Inference model script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument("-i", "--input_file", type=str, help="input file path")
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--params_file", type=str)

    # params for predict
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=None)
    parser.add_argument("--save_path", type=str, default='./test_tipc/output/')
    parser.add_argument("--category", type=str, default='capsule')
    parser.add_argument("--distribution", type=str, default='./test_tipc/output/distribution')
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()




def create_paddle_predictor(args):
    config = Config(args.model_file, args.params_file)
    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.cpu_threads:
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()

    # config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        # choose precision
        if args.precision == "fp16":
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32

        # calculate real max batch size during inference when tenrotRT enabled
        num_seg = 1
        num_views = 1
        max_batch_size = args.batch_size * num_views * num_seg
        config.enable_tensorrt_engine(precision_mode=precision,
                                      max_batch_size=max_batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

    return config, predictor


def crop(img, top, left, height, width):
    """Crops the given image.

    Args:
        img (np.array): Image to be cropped. (0,0) denotes the top left
            corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        np.array: Cropped image.

    """

    return img[top:top + height, left:left + width, :]


def center_crop(img, output_size):
    """Crops the given image and resize it to desired size.

        Args:
            img (np.array): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
            backend (str, optional): The image proccess backend type. Options are `pil`, `cv2`. Default: 'pil'.

        Returns:
            np.array: Cropped image.

        """

    h, w = img.shape[0:2]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)

def preprocess(img):
    transform_x = T.Compose([T.Resize(256),
                             T.CenterCrop(224),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
    x = Image.open(img).convert('RGB')
    x = transform_x(x).unsqueeze(0)
    return x.numpy()


def parse_file_paths(input_path: str) -> list:
    if osp.isfile(input_path):
        files = [
            input_path,
        ]
    else:
        files = os.listdir(input_path)
        files = [
            file for file in files
            if (file.endswith(".png"))
        ]
        files = [osp.join(input_path, file) for file in files]
    return files


def postprocess(args, test_imgs, class_name, outputs, distribution):
    outputs = [paddle.to_tensor(i) for i in outputs]
    idx = paddle.to_tensor(sample(range(0, 448), 100))
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    # get intermediate layer outputs
    for k, v in zip(test_outputs.keys(), outputs):
        test_outputs[k].append(v.detach())
    for k, v in test_outputs.items():
        test_outputs[k] = paddle.concat(v, 0)

    # Embedding concat
    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        layer_embedding = test_outputs[layer_name]
        layer_embedding = F.interpolate(layer_embedding, size=embedding_vectors.shape[-2:], mode="nearest")
        embedding_vectors = paddle.concat((embedding_vectors, layer_embedding), 1)

    # randomly select d dimension
    embedding_vectors = paddle.index_select(embedding_vectors, idx, 1)

    # calculate distance matrix
    B, C, H, W = embedding_vectors.shape
    embedding = embedding_vectors.reshape((B, C, H * W))
    # calculate mahalanobis distances
    mean, covariance = paddle.to_tensor(distribution[0]), paddle.to_tensor(distribution[1])
    inv_covariance = paddle.linalg.inv(covariance.transpose((2, 0, 1)))

    delta = (embedding - mean).transpose((2, 0, 1))

    distances = (paddle.matmul(delta, inv_covariance) * delta).sum(2).transpose((1, 0))
    distances = distances.reshape((B, H, W))
    distances = paddle.sqrt(distances)
    score_map = F.interpolate(distances.unsqueeze(1), size=(224,224), mode='bilinear',
                            align_corners=False).squeeze(1).numpy()

    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)
    save_name = args.save_path
    plot_fig(test_imgs, scores, 0.5, save_name, class_name)


def plot_fig(test_img, scores, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax = ax_img[1].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[1].imshow(img, cmap='gray', interpolation='none')
        ax_img[1].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[1].title.set_text('Predicted heat map')
        ax_img[2].imshow(mask, cmap='gray')
        ax_img[2].title.set_text('Predicted mask')
        ax_img[3].imshow(vis_img)
        ax_img[3].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)
        if i < 1:  # save one result
            fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x

def main():
    args = parse_args()

    random.seed(args.seed)
    paddle.seed(args.seed)

    model_name = 'PaDiM'
    print(f"Inference model({model_name})...")
    # InferenceHelper = build_inference_helper(cfg.INFERENCE)

    print('load train set feature from: %s' % args.distribution)
    with open(args.distribution, 'rb') as f:
        distribution = pickle.load(f)

    inference_config, predictor = create_paddle_predictor(args)

    # get input_tensor and output_tensor
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_tensor_list = []
    output_tensor_list = []
    for item in input_names:
        input_tensor_list.append(predictor.get_input_handle(item))
    for item in output_names:
        output_tensor_list.append(predictor.get_output_handle(item))

    # get the absolute file path(s) to be processed
    files = parse_file_paths(args.input_file)

    if args.enable_benchmark:
        num_warmup = 0

        # instantiate auto log
        import auto_log
        pid = os.getpid()
        autolog = auto_log.AutoLogger(
            model_name="PaDiM",
            model_precision=args.precision,
            batch_size=args.batch_size,
            data_shape="dynamic",
            save_path="./output/auto_log.lpg",
            inference_config=inference_config,
            pids=pid,
            process_name=None,
            gpu_ids=0 if args.use_gpu else None,
            time_keys=['preprocess_time', 'inference_time', 'postprocess_time'],
            warmup=num_warmup)

    # Inferencing process
    batch_num = args.batch_size
    for st_idx in range(0, len(files), batch_num):
        ed_idx = min(st_idx + batch_num, len(files))

        # auto log start
        if args.enable_benchmark:
            autolog.times.start()

        # Pre process batched input
        batched_inputs = [files[st_idx:ed_idx]]
        imgs = []
        test_imgs = []
        for inp in batched_inputs[0]:
            img = preprocess(inp)
            imgs.append(img)
            test_imgs.extend(img)

        imgs = np.concatenate(imgs)
        batched_inputs = [imgs]
        # get pre process time cost
        if args.enable_benchmark:
            autolog.times.stamp()

        # run inference
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(batched_inputs[i].shape)
            input_tensor.copy_from_cpu(batched_inputs[i].copy())

        # do the inference
        predictor.run()

        # get inference process time cost
        if args.enable_benchmark:
            autolog.times.stamp()

        # get out data from output tensor
        results = []
        # get out data from output tensor
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)

        postprocess(args, test_imgs, args.category, results, distribution) # 可视化

        # get post process time cost
        if args.enable_benchmark:
            autolog.times.end(stamp=True)

        # time.sleep(0.01)  # sleep for T4 GPU

    # report benchmark log if enabled
    if args.enable_benchmark:
        autolog.report()


if __name__ == "__main__":
    main()