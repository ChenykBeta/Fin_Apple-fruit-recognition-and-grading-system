# Apple-fruit-recognition-and-grading-system
本项目为西北农林科技大学信息工程学院2024年度校级大学生科创项目。
基于深度学习的苹果识别与分级系统，支持多种目标检测算法。
## 功能特点
- 支持 YOLOv5, YOLOv8, Faster R-CNN 等多种检测算法
- 图形化用户界面 (GUI)
- 实时检测和分级
- 训练和推理一体化

## 项目结构

```
Apple_Grading_Detection/
├── UI_module_simple/          # 简易 UI 模块
├── YoloV5/                    # YOLOv5 实现
├── YoloV8/                    # YOLOv8 实现
├── YoloV8_Improve/            # 改进版 YOLOv8
├── Faster_Rcnn/               # Faster R-CNN 实现
├── DataSet/                   # 数据集 (需单独下载)
└── README.md
```

## 安装

### 1. 环境要求

- Python 3.7+
- PyTorch 1.7+
- CUDA 10.1+ (推荐 GPU 加速)

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载数据集

数据集不包含在此仓库中，请从以下地址下载：

- [百度网盘链接] (待提供)
- 或使用自己的苹果图像数据集

将数据集解压到 `DataSet/` 目录。

### 4. 下载预训练模型

预训练模型文件：

- [百度网盘链接] (待提供)

将模型文件放到对应目录：
- `UI_module_simple/best.pt`
- `YoloV8/yolov8n.pt`

## 使用方法

### 启动 UI 界面

```bash
cd UI_module_simple
python main.py
```

### 训练模型

```bash
# YOLOv8 训练示例
cd YoloV8
python train.py --data your_dataset.yaml --epochs 100
```

### 推理检测

```bash
python detect.py --source image.jpg --weights best.pt
```

## 数据集格式

数据集应包含以下结构：

```
DataSet/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── utils/
```

标注格式：YOLO 格式 (class_id, x_center, y_center, width, height)

## 模型性能

| 模型 | mAP | 推理速度 |
|------|-----|----------|
| YOLOv8n | TBD | TBD |
| YOLOv5s | TBD | TBD |
| Faster R-CNN | TBD | TBD |

## 作者

- ChenykBeta

## License

MIT License

## 致谢

感谢所有为此项目做出贡献的开发者。
