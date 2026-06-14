from pathlib import Path

from ultralytics import YOLO


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    final_path = base_dir / 'train_yaml' / 'yolov8.yaml'
    data_path = base_dir / 'my_data.yaml'
    model = YOLO(str(final_path))
    model.train(data=str(data_path), workers=8, epochs=200, batch=16, optimizer='SGD', device='0')
