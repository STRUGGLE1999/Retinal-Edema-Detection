#!/usr/bin/env python
"""
retrain.py - MindSpore version
Adapted from the original TensorFlow retrain.py by Daniel Kermany and the Zhang Lab team (2017)
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import time
import numpy as np
from net import utils
from net import train
import mindspore as ms
from mindspore import context

# Global flags
FLAGS = None

def main():
    # 设置 MindSpore 运行环境
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    print("Logging: INFO")

    # Create directories to store summary logs
    utils.create_tensorboard_directories(FLAGS.summaries_dir)
    # 创建日志文件，例如 training.log
    log_file_path = os.path.join(FLAGS.summaries_dir, "training.log")
    log_file = open(log_file_path, "w")

    # Set model properties
    model_config = train.get_model_config()

    # 如果 inception_url 非空，则下载预训练权重，否则跳过
    if model_config.get("inception_url"):
        utils.download_pretrained_weights(model_config["inception_url"], FLAGS.model_dir)

    # Create model graph: load pre-trained model and create bottleneck extractor and resized image processor
    pretrained_model, bottleneck_extractor, resized_image_processor = train.create_model_graph(model_config, FLAGS.model_dir)

    # Create lists of all the images
    image_lists = utils.create_image_lists(FLAGS.images)
    class_count = len(image_lists.keys())
    if class_count == 0:
        print("ERROR: No valid folders of images found at " + FLAGS.images)
        return -1
    if class_count == 1:
        print("ERROR: Only one valid folder of images found at " + FLAGS.images + " - multiple classes are needed for classification.")
        return -1

    # Create output_labels.txt displaying classes being trained
    with open(FLAGS.output_labels, "w") as f:
        f.write("\n".join(image_lists.keys()) + "\n")

    # Set up image decoding functions
    # decode_jpeg 返回两个函数用于解码图像
    jpeg_decoder, decoded_image_func = utils.decode_jpeg(
        model_config["input_width"], model_config["input_height"],
        model_config["input_depth"], model_config["input_mean"],
        model_config["input_std"])

    # Store image bottlenecks (pre-compute and save bottleneck features)
    train.store_bottlenecks(
        bottleneck_extractor, image_lists, FLAGS.images, FLAGS.bottleneck_dir,
        jpeg_decoder, decoded_image_func, resized_image_processor)

    # Train newly initialized final layer on bottleneck features
    # print('bottleneck_tensor_size', model_config["bottleneck_tensor_size"])
    final_layer, loss_fn, optimizer = train.train_final_layer(
        len(image_lists.keys()), FLAGS.final_tensor_name, bottleneck_extractor,
        model_config["bottleneck_tensor_size"], FLAGS.learning_rate)

    # Create evaluation metric (e.g., Accuracy)
    evaluation_metric = train.create_evaluation_graph(final_layer, None)

    # In MindSpore, we construct a Model for training final layer.
    # 这里我们假设已经构造好一个数据集生成函数从存储的瓶颈文件创建数据集。
    final_model = ms.Model(final_layer, loss_fn=loss_fn, optimizer=optimizer, metrics={"Accuracy": ms.nn.Accuracy()})

    best_acc = 0.0
    since = time.time()

    # Training loop
    for i in range(FLAGS.training_steps):
        # 获取一个批次的瓶颈特征（训练阶段）
        train_bottlenecks, train_ground_truth, _ = train.get_batch_of_stored_bottlenecks(
            image_lists, FLAGS.train_batch_size, "training",
            FLAGS.bottleneck_dir, FLAGS.images, jpeg_decoder,
            decoded_image_func, resized_image_processor, bottleneck_extractor)
        # 构造一个数据集对象（假设 train.get_dataset_from_bottlenecks 实现了将 numpy 数组转换为 MindSpore 数据集）
        train_dataset = train.get_dataset_from_bottlenecks(train_bottlenecks, train_ground_truth)
        # 训练一步（这里使用 MindSpore Model.train 进行单步训练）
        final_model.train(1, train_dataset=train_dataset, dataset_sink_mode=False)

        # Evaluation at specified frequency
        final_step = (i + 1 == FLAGS.training_steps)
        if (i % FLAGS.eval_frequency == 0) or final_step:
            train_accuracy, ce_value = train.evaluate_model(train_bottlenecks, train_ground_truth, final_layer, evaluation_metric)
            validation_bottlenecks, validation_ground_truth, _ = train.get_batch_of_stored_bottlenecks(
                image_lists, FLAGS.validation_batch_size, "validation",
                FLAGS.bottleneck_dir, FLAGS.images, jpeg_decoder,
                decoded_image_func, resized_image_processor, bottleneck_extractor)
            val_accuracy, _ = train.evaluate_model(validation_bottlenecks, validation_ground_truth, final_layer, evaluation_metric)
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                # utils.save_model(final_model, FLAGS.output_graph, FLAGS.final_tensor_name)
                utils.save_model(final_layer, FLAGS.output_graph, FLAGS.final_tensor_name)
            log_line = "Step {}: loss = {} train acc = {} val acc = {}\n".format(i, ce_value, train_accuracy, val_accuracy)
            # print("Step {}: loss = {} train acc = {} val acc = {}".format(i, ce_value, train_accuracy, val_accuracy))
            print(log_line.strip())
            log_file.write(log_line)

    # Final evaluation on test set
    test_bottlenecks, test_ground_truth, test_filenames = train.get_batch_of_stored_bottlenecks(
        image_lists, FLAGS.test_batch_size, "testing",
        FLAGS.bottleneck_dir, FLAGS.images, jpeg_decoder,
        decoded_image_func, resized_image_processor, bottleneck_extractor)
    test_accuracy, predictions, probabilities = train.evaluate_model(test_bottlenecks, test_ground_truth, final_layer, evaluation_metric, return_details=True)
    print("Best validation accuracy = {}".format(best_acc * 100))
    print("Final test accuracy = {}".format(test_accuracy * 100))

    time_elapsed = time.time() - since
    predictions = np.argmax(probabilities, axis=1)
    labels = np.argmax(test_ground_truth, axis=1)
    # labels = ms.Tensor(labels, dtype=ms.int32)
    print("Total Model Runtime: {}min, {:0.2f}sec".format(int(time_elapsed // 60), time_elapsed % 60))
    log_file.write("Best validation accuracy = {}\n".format(best_acc * 100))
    log_file.write("Final test accuracy = {}\n".format(test_accuracy * 100))
    log_file.write("Final Model AUC: {:0.2f}%\n".format(auc * 100))
    log_file.write("Total Model Runtime: {}min, {:0.2f}sec\n".format(int(time_elapsed // 60), time_elapsed % 60))
    log_file.close()
    pos_idx = 0  # DME 的索引
    roc_labels = [0 if label == pos_idx else 1 for label in labels]  # DME 为 0，其他类别为 1
    pos_probs = probabilities[:, pos_idx]  # 取 DME 预测概率
    roc_probs = pos_probs  # 这里不需要 sum，直接取概率值

    auc = utils.generate_roc(roc_labels, roc_probs, pos_label = 0)
    print("Final Model AUC: {:0.2f}%".format(auc * 100))
      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default="OCT2017/train",
                        help="Path to folder containing subdirectories of training categories (filenames all CAPS)")
    parser.add_argument("--output_graph", type=str, default="/home/mindspore/work/MS_Demo/retrained_model",
                        help="Output file to save the trained model.")
    parser.add_argument("--output_labels", type=str, default="/home/mindspore/work/MS_Demo/output_labels.txt",
                        help="File in which to save the labels.")
    parser.add_argument("--summaries_dir", type=str, default="/home/mindspore/work/MS_Demo/retrain_logs",
                        help="Path to save summary logs.")
    parser.add_argument("--training_steps", type=int, default=5000,
                        help="Number of training steps to run before ending.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Set learning rate.")
    parser.add_argument("--eval_frequency", type=int, default=10,
                        help="How often to evaluate the training results.")
    parser.add_argument("--train_batch_size", type=int, default=256,
                        help="Number of images to train on at a time.")
    parser.add_argument("--test_batch_size", type=int, default=-1,
                        help="Number of images from test set to test on (-1 for all).")
    parser.add_argument("--validation_batch_size", type=int, default=-1,
                        help="Number of images from validation set to validate on (-1 for all).")
    parser.add_argument("--model_dir", type=str, default="/home/mindspore/work/MS_Demo/imagenet",
                        help="Path to pretrained weights.")
    parser.add_argument("--bottleneck_dir", type=str, default="/home/mindspore/work/MS_Demo/bottleneck",
                        help="Path to store bottleneck features.")
    parser.add_argument("--final_tensor_name", type=str, default="final_result",
                        help="Name of the output classification layer in the retrained model.")
    FLAGS, unparsed = parser.parse_known_args()
    main()
