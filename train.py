import os
import time
import random
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from random import sample
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib
import matplotlib.pyplot as plt

import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader

import datasets.mvtec as mvtec
from models.resnet import ResNet_PaDiM

CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default='data/Mvtec')
    parser.add_argument('--save_path', type=str, default='./output')
    parser.add_argument("--category", type=str , default='leather', help="category name for MvTec AD dataset")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--depth", type=int, default=18, help="resnet depth")
    parser.add_argument("--save_picture", type=bool, default=True)
    parser.add_argument("--val", type=bool, default=True)
    parser.add_argument("--print_freq", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():

    args = parse_args()

    random.seed(args.seed)
    paddle.seed(args.seed)

    # build model
    model = ResNet_PaDiM(depth=args.depth, pretrained=True)
    model.eval()

    t_d, d = 448, 100 # "resnet18": {"orig_dims": 448, "reduced_dims": 100, "emb_scale": 4},
    class_name = args.category
    assert class_name in mvtec.CLASS_NAMES
    print("Training model for {}".format(class_name))

    # build datasets
    train_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    if args.val:
        test_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    idx = paddle.to_tensor(sample(range(0, t_d), d))

    train(args, model, train_dataloader, idx)

    if args.val:
        val(args, model, test_dataloader, class_name, idx)



def train(args, model, train_dataloader, idx):
    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    # extract train set features
    epoch_begin = time.time()
    end_time = time.time()

    for index, item in enumerate(train_dataloader):
        start_time = time.time()
        data_time = start_time - end_time
        x = item[0]

        # model prediction
        with paddle.no_grad():
            outputs = model(x)

        # get intermediate layer outputs
        for k, v in zip(train_outputs.keys(), outputs):
            train_outputs[k].append(v.cpu().detach())

        end_time = time.time()
        batch_time = end_time - start_time

        if index % args.print_freq == 0:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
                  "Epoch {}[{}/{}]: loss:{:.5f}, lr:{:.5f}, batch time:{:.4f}, data time:{:.4f}".format(
                      0,
                      index + 1,
                      len(train_dataloader),
                      0,
                      float(0),
                      float(batch_time),
                      float(data_time)
                  ))

    for k, v in train_outputs.items():
        train_outputs[k] = paddle.concat(v, 0)

    # Embedding concat
    embedding_vectors = train_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        layer_embedding = train_outputs[layer_name]
        layer_embedding = F.interpolate(layer_embedding, size=embedding_vectors.shape[-2:], mode="nearest")
        embedding_vectors = paddle.concat((embedding_vectors, layer_embedding), 1)

    # randomly select d dimension
    embedding_vectors = paddle.index_select(embedding_vectors,  idx, 1)
    # calculate multivariate Gaussian distribution
    B, C, H, W = embedding_vectors.shape
    embedding_vectors = embedding_vectors.reshape((B, C, H * W))
    mean = paddle.mean(embedding_vectors, axis=0).numpy()
    cov = paddle.zeros((C, C, H * W)).numpy()
    I = np.identity(C)
    for i in range(H * W):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
    # save learned distribution
    train_outputs = [mean, cov]
    model.distribution = train_outputs
    t = time.time() - epoch_begin
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          "Train ends, total {:.2f}s".format(0, t))
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Saving model...")
    save_name = os.path.join(args.save_path, args.category, 'best.pdparams')
    dir_name = os.path.dirname(save_name)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    state_dict = {
        "params":model.model.state_dict(),
        "distribution":model.distribution,
    }
    paddle.save(state_dict, save_name)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Save model in {}".format(str(save_name)))


def val(args, model, test_dataloader, class_name, idx):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Starting eval model...")
    total_roc_auc = []
    total_pixel_roc_auc = []

    gt_list = []
    gt_mask_list = []
    test_imgs = []

    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    # extract test set features
    for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):

        test_imgs.extend(x.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        gt_mask_list.extend(mask.cpu().detach().numpy())
        # model prediction
        with paddle.no_grad():
            outputs = model(x)
        # get intermediate layer outputs
        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v.cpu().detach())
    for k, v in test_outputs.items():
        test_outputs[k] = paddle.concat(v, 0)

    # Embedding concat
    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        layer_embedding = test_outputs[layer_name]
        layer_embedding = F.interpolate(layer_embedding, size=embedding_vectors.shape[-2:], mode="nearest")
        embedding_vectors = paddle.concat((embedding_vectors, layer_embedding), 1)

    # randomly select d dimension
    embedding_vectors = paddle.index_select(embedding_vectors,  idx, 1)

    # calculate distance matrix
    B, C, H, W = embedding_vectors.shape
    embedding_vectors = embedding_vectors.reshape((B, C, H * W)).numpy()
    dist_list = []
    for i in range(H * W):
        mean = model.distribution[0][:, i]
        conv_inv = np.linalg.inv(model.distribution[1][:, :, i])
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)

    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)


    # upsample
    dist_list = paddle.to_tensor(dist_list)
    score_map = F.interpolate(dist_list.unsqueeze(1), size=x.shape[2:], mode='bilinear',
                              align_corners=False).squeeze().numpy()

    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)

    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    total_roc_auc.append(img_roc_auc)

    # get optimal threshold
    gt_mask = np.asarray(gt_mask_list, dtype=np.int64)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    # calculate per-pixel level ROCAUC
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    total_pixel_roc_auc.append(per_pixel_rocauc)
    if args.save_picture:
        save_name = os.path.join(args.save_path, args.category)
        dir_name = os.path.dirname(save_name)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_name, class_name)

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          'Class:{}'.format(class_name) +':\t'+ 'Image AUC: %.3f' % np.mean(total_roc_auc))
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          'Class:{}'.format(class_name) +':\t'+ 'Pixel AUC: %.3f' % np.mean(total_pixel_roc_auc))


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
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
        if i < 1: # save one result
            fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


if __name__ == '__main__':
    main()
