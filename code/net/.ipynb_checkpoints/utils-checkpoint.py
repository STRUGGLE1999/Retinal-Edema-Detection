#!/usr/bin/env python
"""
utils.py - MindSpore version
包含文件操作、下载预训练权重、图像列表构建、图像路径获取、瓶颈路径构建、TensorBoard 目录创建以及 JPEG 解码等功能。
"""

import sys
import os
from glob import glob
from six.moves import urllib
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from mindspore import Tensor
import mindspore.common.dtype as mstype

def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_image_files(image_dir):
    fs = glob("{}/*.jpeg".format(image_dir))
    fs = [os.path.basename(filename) for filename in fs]
    return sorted(fs)

def generate_roc(y_test, y_score, pos_label=0):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic curve")
    plt.show()
    return roc_auc

def download_pretrained_weights(inception_url, dest_dir):
    # 如果 inception_url 为空，则不下载
    if not inception_url:
        return
    create_directory(dest_dir)
    filename = inception_url.split("/")[-1]
    filepath = os.path.join(dest_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write("\r>> Downloading {} {:0.1f}%".format(
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(inception_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print("Successfully downloaded {} {} bytes.".format(filename, statinfo.st_size))
        tarfile.open(filepath, "r:gz").extractall(dest_dir)

def save_model(model, output_file, final_tensor_name):
    from mindspore.train.serialization import export
    import numpy as np
    from mindspore.train.serialization import save_checkpoint
    # dummy_input = np.random.randn(1, 3, 299, 299).astype(np.float32)
    # 生成 ckpt 文件名，可以直接在 output_file 后添加 .ckpt 扩展名
    ckpt_file = output_file + ".ckpt"
    save_checkpoint(model, ckpt_file)
    print("Checkpoint saved to", ckpt_file)
    # dummy_input = Tensor(np.random.randn(1, 2048), dtype=mstype.float32) 
    # export(model, dummy_input, file_name=output_file, file_format="MINDIR")
    # print("Model exported to", output_file)

def create_image_lists(image_dir):
    result = {}
    training_images = []
    testing_images = []
    validation_images = []
    for category in ["train", "test", "val"]:
        category_path = os.path.join(image_dir, category)
        try:
            bins = next(os.walk(category_path))[1]
        except StopIteration:
            sys.exit("ERROR: Missing either train/test/val folders in image_dir")
        for diagnosis in bins:
            bin_path = os.path.join(category_path, diagnosis)
            if category == "train":
                training_images.append(get_image_files(bin_path))
            if category == "test":
                testing_images.append(get_image_files(bin_path))
            if category == "val":
                validation_images.append(get_image_files(bin_path))
    for diagnosis in bins:
        result[diagnosis] = {
            "training": training_images[bins.index(diagnosis)],
            "testing": testing_images[bins.index(diagnosis)],
            "validation": validation_images[bins.index(diagnosis)],
        }
    return result

def get_image_path(image_lists, label_name, index, image_dir, category):
    if label_name not in image_lists:
        raise ValueError("Label does not exist: {}".format(label_name))
    label_lists = image_lists[label_name]
    if category not in label_lists:
        raise ValueError("Category does not exist: {}".format(category))
    category_list = label_lists[category]
    if not category_list:
        raise ValueError("Label {} has no images in the category {}.".format(label_name, category))
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    if "train" in category:
        full_path = os.path.join(image_dir, "train", label_name.upper(), base_name)
    elif "test" in category:
        full_path = os.path.join(image_dir, "test", label_name.upper(), base_name)
    elif "val" in category:
        full_path = os.path.join(image_dir, "val", label_name.upper(), base_name)
    return full_path

def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category):
    return get_image_path(image_lists, label_name, index, bottleneck_dir, category) + "_inception_v3.txt"

def create_tensorboard_directories(summaries_dir):
    if os.path.exists(summaries_dir):
        import shutil
        shutil.rmtree(summaries_dir)
    os.makedirs(summaries_dir)

def decode_jpeg(input_width, input_height, input_depth, input_mean, input_std):
    from PIL import Image
    def decode_image_file(image_path):
        img = Image.open(image_path)
        img = img.convert("RGB")
        img = img.resize((input_width, input_height))
        img_array = np.array(img).astype(np.float32)
        img_array = (img_array - input_mean) / input_std
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array
    # 返回同一个函数作为解码函数（MindSpore 中直接使用该函数）
    return decode_image_file, decode_image_file

def decode_image(image_path, target_size):
    from PIL import Image
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32)
    # Normalize using 0 and 1 or as required,下面保持不变
    img_array = np.transpose(img_array, (2, 0, 1))
    return img_array
