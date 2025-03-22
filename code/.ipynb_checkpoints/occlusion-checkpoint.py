#!/usr/bin/env python
"""
occlusion.py – MindSpore 版本

使用训练好的模型对输入图像进行遮挡分析，系统地在图像上移动遮挡块（occluding filter），
计算目标类别（通常为原图预测的最高置信度类别）在不同遮挡位置下的预测概率变化，
并生成遮挡热力图，最后将热力图与原图融合保存。
    
示例用法:
  python occlusion.py \
    --image_file=/path/to/image.jpg \
    --ckpt=/path/to/retrained_model.ckpt \
    --labels=/path/to/output_labels.txt \
    --roi_size=32 \
    --stride=32
"""

import argparse
import os
import time
from math import floor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import mindspore as ms
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net

# 假设 train.py 和 utils.py 与本文件在同一目录下
from net import train
from net import utils

def create_occlusion(im, roi_size, stride_size):
    """
    对给定 PIL 图像 im（尺寸应为 512x512）创建遮挡图像批次，
    遮挡块大小为 roi_size，步长为 stride_size。
    返回遮挡图像列表以及 grid 的行数（列数相同）。
    """
    # 根据 512 像素大小计算步数
    iters = int(floor((512 - roi_size + 1) / stride_size))
    batch = []
    for r in range(iters):
        for c in range(iters):
            imOc = im.copy()
            draw = ImageDraw.Draw(imOc)
            # 用灰色（150,150,150）填充遮挡区域
            left = c * stride_size
            top = r * stride_size
            right = left + roi_size - 1
            bottom = top + roi_size - 1
            draw.rectangle([left, top, right, bottom], fill=(150, 150, 150))
            batch.append(imOc)
    return batch, iters

def process_image_for_model(im, model_config):
    """
    将 PIL 图像转换为模型输入格式：
    - 调整大小到模型所需尺寸（例如 299x299）
    - 转为 numpy 数组、归一化并转置为 (C,H,W)
    """
    input_width = model_config["input_width"]
    input_height = model_config["input_height"]
    input_mean = model_config["input_mean"]
    input_std = model_config["input_std"]
    
    im_resized = im.resize((input_width, input_height))
    img_array = np.array(im_resized).astype(np.float32)
    img_array = (img_array - input_mean) / input_std
    # 转置为 (C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))
    return img_array

def build_model(ckpt_path, num_classes):
    """
    根据 train.py 中的逻辑构造模型：
      - 加载 MindCV 中预训练的 InceptionV3 模型，并构造瓶颈提取器（bottleneck_extractor）
      - 创建最终分类层（final_layer）
      - 加载 checkpoint 参数到 final_layer
    返回：(bottleneck_extractor, final_layer, model_config)
    """
    # 获取模型配置信息（例如输入尺寸、瓶颈尺寸等）
    model_config = train.get_model_config()
    # 这里的 model_dir 可传一个占位路径，因为 create_model_graph 会加载预训练模型
    dummy_model_dir = "./dummy_model_dir"
    _, bottleneck_extractor, _ = train.create_model_graph(model_config, dummy_model_dir)
    
    # 创建最终层，类别数由 labels 文件决定
    final_tensor_name = "final_result"  # 与训练时保持一致
    final_layer, _, _ = train.train_final_layer(num_classes, final_tensor_name,
                                                bottleneck_extractor,
                                                model_config["bottleneck_tensor_size"],
                                                learning_rate=0.001)
    # 加载训练好的参数到最终层
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(final_layer, param_dict)
    
    # 设置网络为评估模式
    final_layer.set_train(False)
    bottleneck_extractor.set_train(False)
    
    return bottleneck_extractor, final_layer, model_config

def predict(im, bottleneck_extractor, final_layer, model_config):
    """
    对单张图像（PIL格式）进行预测：
      - 预处理图像为模型输入（大小、归一化、转置）
      - 将图像转为 Tensor，并通过 bottleneck_extractor 提取特征
      - 通过 final_layer 得到 logits，再用 softmax 计算概率
    返回一个 numpy 数组，形状为 (num_classes,)
    """
    img_array = process_image_for_model(im, model_config)
    # 扩展 batch 维度
    tensor_img = ms.Tensor(np.expand_dims(img_array, axis=0), ms.float32)
    # 获取瓶颞特征（形状为 [1, bottleneck_tensor_size]）
    bottleneck = bottleneck_extractor(tensor_img)
    logits = final_layer(bottleneck)
    softmax_op = nn.Softmax()
    probs = softmax_op(logits)
    return probs.asnumpy()[0]

def blend_occlusion_map(orig_im, occlusion_map, blend_alpha=0.4):
    """
    将生成的遮挡热力图与原图融合：
      - 将 occlusion_map 数值归一化至 0~255，并转换为灰度 PIL 图像（先转换为 RGB）
      - 调整遮挡图大小与原图一致，融合后返回融合图像
    """
    # 将 occlusion_map 归一化为 [0, 255]
    norm_map = (occlusion_map - np.min(occlusion_map)) / (np.max(occlusion_map) - np.min(occlusion_map) + 1e-8)
    norm_map = (norm_map * 255).astype(np.uint8)
    # 使用 matplotlib 的 colormap（例如 magma）将灰度图转换为 RGB
    cmap = plt.get_cmap("magma")
    colored_map = cmap(norm_map)
    # colored_map 为 (H,W,4) RGBA，转换为 RGB
    colored_map = (colored_map[:, :, :3] * 255).astype(np.uint8)
    occlusion_im = Image.fromarray(colored_map)
    # 调整 occlusion_im 与原图尺寸一致
    occlusion_im = occlusion_im.resize(orig_im.size)
    # 融合
    blended = Image.blend(orig_im, occlusion_im, blend_alpha)
    return blended

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_file", type=str, required=True, default="OCT2017/test/DME/DME-1081406-1.jpeg",
                        help="绝对路径，待分析图像文件（例如 JPEG）")
    parser.add_argument("--ckpt", type=str, required=True, default="MS_Demo/retrained_model.ckpt",
                        help="训练好的 checkpoint 文件路径（.ckpt）")
    parser.add_argument("--labels", type=str, required=True,
                        help="包含类别名称的文本文件路径")
    parser.add_argument("--roi_size", type=int, default=32,
                        help="遮挡块大小（像素）")
    parser.add_argument("--stride", type=int, default=32,
                        help="遮挡块移动步长（像素）")
    parser.add_argument("--output_dir", type=str, default="MS_Demo/Occlusion",
                        help="融合图像保存目录")
    args = parser.parse_args()

    # 加载类别标签
    with open(args.labels, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    num_classes = len(labels)

    # 构建模型：加载瓶颈提取器与最终层
    bottleneck_extractor, final_layer, model_config = build_model(args.ckpt, num_classes)

    # 加载图像，并调整为 512x512 用于 occlusion 分析
    orig_im = Image.open(args.image_file).convert("RGB")
    im_512 = orig_im.resize((512, 512))

    # 获取原图预测，确定目标类别（取置信度最高的类别）
    baseline_probs = predict(im_512, bottleneck_extractor, final_layer, model_config)
    target_class = int(np.argmax(baseline_probs))
    baseline_target_prob = baseline_probs[target_class]
    print("原图预测类别: {} 置信度: {:.4f}".format(labels[target_class], baseline_target_prob))

    # 生成遮挡图像批次
    roi_size = args.roi_size
    stride_size = args.stride
    occlusion_batch, grid_size = create_occlusion(im_512, roi_size, stride_size)
    num_occlusions = len(occlusion_batch)
    print("生成 {} 张遮挡图像，grid 大小: {}x{}".format(num_occlusions, grid_size, grid_size))

    # 对每个遮挡图像进行预测，提取目标类别的预测概率
    occlusion_probs = []
    for idx, oc_im in enumerate(occlusion_batch):
        probs = predict(oc_im, bottleneck_extractor, final_layer, model_config)
        occlusion_probs.append(probs[target_class])
        if (idx + 1) % 50 == 0:
            print("处理 {}/{} 张遮挡图".format(idx + 1, num_occlusions))
    occlusion_probs = np.array(occlusion_probs)

    # 构造遮挡热力图：这里用 (baseline_target_prob - occluded_prob) 表示预测下降幅度
    occlusion_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    i = 0
    for r in range(grid_size):
        for c in range(grid_size):
            occlusion_map[r, c] = baseline_target_prob - occlusion_probs[i]
            i += 1

    # 绘制遮挡热力图
    plt.figure()
    plt.matshow(occlusion_map, interpolation="lanczos", cmap="magma")
    plt.title("Occlusion Map for class '{}'".format(labels[target_class]))
    plt.colorbar()
    # 临时保存热力图
    heatmap_path = os.path.join("/home/mindspore/work/MS_Demo", "occlusion_map.png")
    plt.savefig(heatmap_path, bbox_inches="tight", pad_inches=0)
    plt.close("all")
    print("热力图保存至:", heatmap_path)

    # 融合原图与热力图
    blended = blend_occlusion_map(orig_im, occlusion_map, blend_alpha=0.4)

    # 保存融合图像至指定目录
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_name = os.path.basename(args.image_file)
    output_path = os.path.join(output_dir, base_name)
    blended.save(output_path)
    print("融合图像已保存至:", output_path)

    elapsed = time.time()
    print("Occlusion 分析完成。")

if __name__ == "__main__":
    main()
