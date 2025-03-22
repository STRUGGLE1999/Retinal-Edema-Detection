#!/usr/bin/env python
"""
train.py - MindSpore version
包含模型配置、瓶颈特征计算、最终层训练、评估等功能。
"""

import os
import random
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
# import mindspore.common.dtype as mstype
import mindspore.common.dtype as mstype

def get_model_config():
    # 此处将 inception_url 置为空字符串，因为我们使用 mindcv 的 InceptionV3 模型
    inception_url = ""
    bottleneck_tensor_size = 2048
    input_width = 299 
    input_height = 299
    input_depth = 3
    model_file_name = "inception_v3.ms"  # 使用 mindcv 提供的 InceptionV3 模型文件名
    input_mean = 128
    input_std = 128
    return {
        "inception_url": inception_url,
        "bottleneck_tensor_size": bottleneck_tensor_size,
        "input_width": input_width,
        "input_height": input_height,
        "input_depth": input_depth,
        "model_file_name": model_file_name,
        "input_mean": input_mean,
        "input_std": input_std,
    }

def run_bottleneck_on_image(model, image_data):
    """
    使用预训练模型计算图像的瓶颈特征。
    image_data 为 MindSpore Tensor。
    """
    features = model(image_data)
    features = features.asnumpy().squeeze()
    return features

def create_bottleneck(bottleneck_path, image_path, model):
    """
    对指定图像计算瓶颈特征，并存储到 bottleneck_path 中。
    """
    from net import utils
    print("Creating Bottleneck at {}".format(bottleneck_path))
    # 使用 utils.decode_image 解码图像
    img = utils.decode_image(image_path, target_size=(get_model_config()["input_height"], get_model_config()["input_width"]))
    bottleneck_values = run_bottleneck_on_image(model, Tensor(np.expand_dims(img, axis=0).astype(np.float32)))
    bottleneck_string = ",".join(str(x) for x in bottleneck_values)
    bottleneck_dir = os.path.dirname(bottleneck_path)
    if not os.path.exists(bottleneck_dir):
        os.makedirs(bottleneck_dir)
    with open(bottleneck_path, "w") as f:
        f.write(bottleneck_string)

def get_bottleneck(model, image_lists, label_name, index, image_dir, category, bottleneck_dir):
    """
    获取指定图像的瓶颈特征，如果不存在则计算后保存。
    """
    from net import utils
    bottleneck_path = utils.get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category)
    if not os.path.exists(bottleneck_path):
        image_path = utils.get_image_path(image_lists, label_name, index, image_dir, category)
        create_bottleneck(bottleneck_path, image_path, model)
    with open(bottleneck_path, "r") as f:
        bottleneck_string = f.read()
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(",")]
    except ValueError:
        print("Error reading bottleneck, recreating bottleneck")
        image_path = utils.get_image_path(image_lists, label_name, index, image_dir, category)
        create_bottleneck(bottleneck_path, image_path, model)
        with open(bottleneck_path, "r") as f:
            bottleneck_string = f.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(",")]
    return bottleneck_values

def store_bottlenecks(model, image_lists, image_dir, bottleneck_dir, jpeg_decoder, decoded_image_func, resized_image_processor):
    num_bottlenecks = 0
    from net import utils
    if not os.path.exists(bottleneck_dir):
        os.makedirs(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ["training", "testing", "validation"]:
            category_list = label_lists[category]
            for index in range(len(category_list)):
                get_bottleneck(model, image_lists, label_name, index, image_dir, category, bottleneck_dir)
                num_bottlenecks += 1
                if num_bottlenecks % 100 == 0:
                    print("{} bottleneck files created.".format(num_bottlenecks))

def train_final_layer(class_count, final_tensor_name, bottleneck_model, bottleneck_tensor_size, learning_rate):
    """
    新增一个全连接层，用于在瓶颈特征上进行分类训练
    """
    # print('bottleneck_tensor_size',bottleneck_tensor_size)
    class FinalLayer(nn.Cell):
        def __init__(self, bottleneck_tensor_size, class_count):
            super(FinalLayer, self).__init__()
            self.fc = nn.Dense(
                in_channels=bottleneck_tensor_size,
                out_channels=class_count,
                dtype=mstype.float32 # 设置为 float32 与输入数据类型一致
            )
        def construct(self, x):
            x = x.astype(np.float32)  # 转换输入为 float32
            # x = x.view(x.shape[0], -1)  # 展平操作，将输入转换为二维张量
            x = x.view(-1, self.fc.in_channels)  # 使用固定维度展平
            return self.fc(x)
    final_layer = FinalLayer(bottleneck_tensor_size, class_count)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction="mean")
    optimizer = nn.Adam(final_layer.trainable_params(), learning_rate=learning_rate)
    return final_layer, loss_fn, optimizer

def create_evaluation_graph(final_layer, ground_truth_tensor):
    """
    创建评估模块，这里直接返回 Accuracy 计算单元
    """
    evaluation_metric = nn.Accuracy()
    return evaluation_metric

def create_model_graph(model_config, model_dir):
    """
    加载预训练模型，并构造瓶颈特征提取器与图像预处理函数
    """
    from mindcv.models import inception_v3
    # 加载预训练 InceptionV3 模型
    pretrained_model = inception_v3(num_classes=1000, pretrained=True)
    # 构造瓶颈提取器：截取 InceptionV3 从前向处理到全局平均池化和 dropout 之后的输出
    class BottleneckExtractor(nn.Cell):
        def __init__(self, model):
            super(BottleneckExtractor, self).__init__()
            self.model = model
        def construct(self, x):
            # 调用 InceptionV3 的内部方法，获得倒数第二层输出
            x = self.model.forward_preaux(x)
            x = self.model.forward_postaux(x)
            x = self.model.pool(x)      # 全局平均池化，输出维度应为 2048
            x = self.model.dropout(x)   # dropout 层
            return x

    bottleneck_extractor = BottleneckExtractor(pretrained_model)
    # 定义 resized_image_processor 为图像预处理函数
    def resized_image_processor(image):
        from mindspore.dataset.vision import vision
        resize = vision.Resize((model_config["input_height"], model_config["input_width"]))
        return resize(image)
    return pretrained_model, bottleneck_extractor, resized_image_processor

def get_batch_of_stored_bottlenecks(image_lists, batch_size, category, bottleneck_dir, image_dir, jpeg_decoder, decoded_image_func, resized_image_processor, bottleneck_model):
    MAX_NUM_IMAGES_PER_CLASS = 2**27 - 1
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if batch_size >= 0:
        for i in range(batch_size):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            from net import utils
            image_name = utils.get_image_path(image_lists, label_name, image_index, image_dir, category)
            bottleneck = get_bottleneck(bottleneck_model, image_lists, label_name, image_index, image_dir, category, bottleneck_dir)
            # ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            # ground_truth[label_index] = 1.0
            ground_truth[label_index] = 1
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            filenames.append(image_name)
    else:
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(image_lists[label_name][category]):
                from net import utils
                image_name = utils.get_image_path(image_lists, label_name, image_index, image_dir, category)
                bottleneck = get_bottleneck(bottleneck_model, image_lists, label_name, image_index, image_dir, category, bottleneck_dir)
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)
    return np.array(bottlenecks), np.array(ground_truths), filenames

def get_dataset_from_bottlenecks(bottlenecks, ground_truth):
    """
    将存储的瓶颈特征与标签转换为 MindSpore 数据集对象
    """
    import mindspore.dataset as ds
    # 将标签转换为 int32 类型
    # ground_truth = ground_truth.astype(np.int32)
    dataset = ds.NumpySlicesDataset({"bottleneck": bottlenecks, "label": ground_truth}, shuffle=True)
    dataset = dataset.batch(bottlenecks.shape[0])
    return dataset

# def evaluate_model(bottlenecks, ground_truth, final_layer, evaluation_metric, return_details=False):
#     bottlenecks_tensor = Tensor(bottlenecks.astype(np.float32))
#     ground_truth_tensor = Tensor(ground_truth.astype(np.int32))
#     logits = final_layer(bottlenecks_tensor)
#     predictions = logits.asnumpy()
#     predicted_labels = np.argmax(predictions, axis=1)
#     true_labels = np.argmax(ground_truth, axis=1)
#     accuracy = evaluation_metric(Tensor(predicted_labels), Tensor(true_labels))
#     if return_details:
#         return accuracy, predictions, predictions
#     else:
#         return accuracy, np.mean(logits.asnumpy())

def evaluate_model(bottlenecks, ground_truth, final_layer, evaluation_metric, return_details=False):
    bottlenecks_tensor = Tensor(bottlenecks.astype(np.float32))
    logits = final_layer(bottlenecks_tensor)  # logits shape: [batch_size, num_classes]
    predictions = logits.asnumpy()
    # 不对 predictions 进行 argmax，直接传入二维数组
    true_labels = np.argmax(ground_truth, axis=1)  # true_labels shape: [batch_size]
    accuracy = evaluation_metric(Tensor(predictions), Tensor(true_labels))
    if return_details:
        return accuracy, predictions, predictions
    else:
        return accuracy, np.mean(logits.asnumpy())

