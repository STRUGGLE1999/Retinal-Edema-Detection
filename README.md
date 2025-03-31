# Retina-AI

Retina-AI 是一个基于 MindSpore 的图像分类项目，其主要任务是对眼科 OCT 数据进行训练、评估以及可解释性分析（例如 ROC 曲线生成和遮挡分析）。本 README 文件详细说明了数据集的组织方式、训练与评估流程、ROC 曲线生成以及遮挡分析的使用方法。

---


Requirements:
```bash
mindcv
mindspore==2.5.0
matplotlib
numpy
opencv-python
pandas
```
## 实验环境
本实验基于 MindSpore 框架，并使用 MindCV 套件中 InceptionV3 预训练模型，模型地址：https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/inceptionv3.py

软件环境：

- python==3.9.0
- mindspore==2.5.0
- mindcv==0.3.0

硬件环境：Ascend


## 数据集收集

### 数据集下载

下载链接：https://data.mendeley.com/datasets/rscbjbr9sj/2

### 数据集介绍
该数据集包括四个类别：CNV、DME、DRUSEN、NORMAL，分别代表如下意思
CNV：脉络膜血管新生（Choroidal neovascularization）
DME：糖尿病黄斑水肿（Diabetic macular edema）
DRUSEN:脉络膜玻璃膜疣
NORMAL:正常

### 验证集划分

为了训练和评估模型，需要准备一个数据集，并按照类别将图像存放在对应的目录中。数据集应包含三个父目录，分别命名为 `train`（训练集）、`test`（测试集）和 `val`（验证集）。在每个父目录内，应创建多个子目录，子目录名称必须全部大写，代表各个类别（例如：`CNV`、`DME`、`DRUSEN`、`NORMAL`）。

例如，当你下载附带的 OCT 数据集时，会发现数据集中仅包含 `train` 和 `test` 两个文件夹。此时，你需要从训练集或测试集中抽取部分图像，移动到新建的 `val` 文件夹中。这样，`train`、`test` 和 `val` 三个目录中各自都会包含 4 个子文件夹，对应上述 4 个类别。命令行中指定的图像路径应指向包含这三个父文件夹的根目录。

划分代码

```python
import os
import shutil
import random

random.seed(42)  # 固定随机种子，确保结果可复现

dataset_path = "./OCT2017"  # 数据集根目录
train_folder = os.path.join(dataset_path, "train")
test_folder = os.path.join(dataset_path, "test")
val_folder = os.path.join(dataset_path, "val")

categories = ["CNV", "DME", "DRUSEN", "NORMAL"]

# 创建 val 文件夹及各类别子文件夹
if not os.path.exists(val_folder):
    os.mkdir(val_folder)
for cat in categories:
    cat_val_folder = os.path.join(val_folder, cat)
    if not os.path.exists(cat_val_folder):
        os.mkdir(cat_val_folder)

# 设定从 train 和 test 目录分别抽取 10% 的数据到 val
total_move_ratio = 0.2  # 总共 20% 的数据
train_move_ratio = 0.1   # 10% 来自 train
test_move_ratio = 0.1    # 10% 来自 test

def move_images(source_folder, dest_folder, category, ratio):
    source_cat_folder = os.path.join(source_folder, category)
    dest_cat_folder = os.path.join(dest_folder, category)
    
    if not os.path.exists(source_cat_folder):
        print(f"警告: {source_cat_folder} 不存在，跳过。")
        return
    
    files = [f for f in os.listdir(source_cat_folder) if f.lower().endswith(('.jpeg', '.jpg'))]
    num_to_move = int(len(files) * ratio)
    files_to_move = random.sample(files, num_to_move) if num_to_move > 0 else []
    
    for file in files_to_move:
        src_file = os.path.join(source_cat_folder, file)
        dst_file = os.path.join(dest_cat_folder, file)
        shutil.move(src_file, dst_file)
        print(f"Moved {src_file} to {dst_file}")

# 从 train 和 test 目录中分别抽取部分数据
for cat in categories:
    move_images(train_folder, val_folder, cat, train_move_ratio)
    move_images(test_folder, val_folder, cat, test_move_ratio)

print("val 数据集构建完成。")

```


## 模型训练

### 预训练模型

本实验采用MindSpore框架，使用mindcv套件的inceptionv3模型

地址为：https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/inceptionv3.py

### 训练步骤

本项目的代码结构如下：

```bash
├── MS_Demo
│   └── code
│       ├── retrain.py
│       └── net
│           ├── train.py
│           └── utils.py
└── OCT2017
    ├── train
    │   ├── CNV
    │   │   ├── image1.jpeg
    │   │   ├── image2.jpeg
    │   │   └── ...
    │   ├── DME
    │   │   └── ...
    │   ├── DRUSEN
    │   │   └── ...
    │   └── NORMAL
    │       └── ...
    ├── test
    │   ├── CNV
    │   │   └── ...
    │   ├── DME
    │   │   └── ...
    │   ├── DRUSEN
    │   │   └── ...
    │   └── NORMAL
    │       └── ...
    └── val
        ├── CNV
        │   └── ...
        ├── DME
        │   └── ...
        ├── DRUSEN
        │   └── ...
        └── NORMAL
            └── ...
```

假设数据集存放在 `/home/mindspore/work/OCT2017`，请使用如下命令启动重新训练脚本：

```bash
python retrain.py --images /home/mindspore/work/OCT2017
```
### 遮挡测试

```bash
python occlusion.py     --image_file=/home/mindspore/work/OCT2017/test/DME/DME-1081406-1.jpeg     --ckpt=/home/mindspore/work/MS_Demo/retrained_model.ckpt     --labels=/home/mindspore/work/MS_Demo/output_labels.txt     --roi_size=32     --stride=32

```

# Ciation

Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2
