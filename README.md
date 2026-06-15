# README\.md

# 计算机视觉实验 Lab2 项目说明

## 项目概述

本项目为计算机视觉基础课程实验，完整实现**校园地标图像检索**与**场景文本区域检测**两大核心任务。

## 一、环境配置

### 1\. 基础运行环境

- 操作系统：Windows10 / Windows11

- 运行环境：Anaconda 虚拟环境（Python3\.9\+）

- 加速支持：CUDA GPU 加速（自动适配，无GPU自动使用CPU）

### 2\. 依赖安装

安装基础依赖：

```Plain Text
pip install torch torchvision pillow numpy opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 二、完整项目文件结构

```Plain Text
lab2/
├── image_retrieval.py          # 任务一：图像检索、P@K计算、绘图
├── text_det_train_infer.py  # 文本检测模型训练脚本（首次运行）
├── infer_only.py            # 文本检测推理脚本（无需重训，日常使用）
├── concat_result.py         # 自动分组拼接，生成24组可视化结果
├── text_det_model.pth       # 训练完成的模型权重（自动生成）
├── text_detect_result/      # 单张图片文本检测结果（自动生成）
├── 任务一_PK曲线/            # P@K图结果
└── 任务二_24组可视化/       # 最终实验成品拼接图（自动生成）

```

## 三、标准运行流程（严格顺序）

### 1\. 进入项目目录与虚拟环境

```Plain Text
conda activate ml_env
cd D:\lesson_learn\计算机视觉\lab2
```

### 2\. 运行任务一：图像检索

```Plain Text
python image_retrieval.py
```

功能：自动完成特征匹配、相似度排序、计算P@K指标、绘制精度曲线。

### 3\. 运行任务二：模型训练（仅首次执行）

```Plain Text
python text_det_train_infer.py
```

功能：自动划分8:2训练/验证集，迭代训练并保存模型权重，后续无需重复训练。

### 4\. 批量推理生成检测结果（核心常用）

```Plain Text
python infer_only.py
```

功能：加载已有权重，批量对所有图片做文本检测、绘制红色标注框，完美兼容中文路径、自动跳过损坏图片，无报错崩溃问题。

### 5\. 生成最终24组可视化成果

```Plain Text
python concat_result.py
```

功能：按12类地标、每类2组、每组10张自动拼接，输出课程要求的24组实验效果图。


## 四、项目输出成果清单

1. 任务一成果：图像检索P@K精度曲线、检索排序结果；

2. 任务二成果：24组标准化分组拼接可视化图片。
