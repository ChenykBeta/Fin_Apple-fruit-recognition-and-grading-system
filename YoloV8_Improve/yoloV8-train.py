import json
import os
from pathlib import Path


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    # Use a project-local Ultralytics settings dir to avoid affecting other projects
    config_dir = (base_dir / ".ultralytics").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    settings_file = config_dir / "settings.json"
    if not settings_file.exists():
        settings_file.write_text(
            json.dumps(
                {
                    "settings_version": "0.0.6",
                    "datasets_dir": str((base_dir / ".." / "DataSet").resolve()),
                    "weights_dir": str((base_dir / "weights").resolve()),
                    "runs_dir": str((base_dir / "runs").resolve()),
                    "uuid": "local",
                    "sync": False,
                    "api_key": "",
                    "openai_api_key": "",
                    "clearml": False,
                    "comet": False,
                    "dvc": False,
                    "hub": False,
                    "mlflow": False,
                    "neptune": False,
                    "raytune": False,
                    "tensorboard": True,
                    "wandb": False,
                    "vscode_msg": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    from ultralytics import YOLO

    final_path = base_dir / 'train_yaml' / 'improve_all_yolov8.yaml'
    data_path = base_dir / 'my_data.yaml'
    model = YOLO(str(final_path))
    model.train(
        data=str(data_path),         # 指定数据集 yaml
        workers=0,                   # 读取数据的线程数（Windows 下 0 最稳）workers这点看实际需求，原参数是8，如果数据集较大，建议设置为0以避免 Windows 的多线程问题；如果数据集较小，可以适当增加以加快训练速度
        epochs=200,                  # 训练轮数
        batch=8,                    # 每个 batch 图像数（RTX 4060 8G 稳定值）原参数为16
        optimizer='SGD',             # 优化器
        device='0',                  # GPU 编号
        amp=True,                    # 是否开启混合精度
        cache='disk',                # 缓存到磁盘，避免 RAM/句柄压力
        rect=True,                  # 关闭矩形训练，避免 shuffle 被关闭
        imgsz=640,                   # 输入图像尺寸,原尺寸为640，训练时可以适当调整以平衡速度和性能
        plots=False,                 # 关闭 labels.jpg 绘制，避免崩溃
        # 如需恢复绘图：把 plots 改为 True
        # 如需恢复矩形训练：把 rect 改为 True
        # 如需恢复内存缓存：把 cache 改为 True（不推荐在 Windows）

    
    )
