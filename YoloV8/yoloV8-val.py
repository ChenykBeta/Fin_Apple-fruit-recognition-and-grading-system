from pathlib import Path

from ultralytics import YOLO

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    weights = base_dir / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt'
    data_path = base_dir / 'my_data.yaml'
    model = YOLO(str(weights))
    model.val(split='test', data=str(data_path), workers=4, batch=16)
