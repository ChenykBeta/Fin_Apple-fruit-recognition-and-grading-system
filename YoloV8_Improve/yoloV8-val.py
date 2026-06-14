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

    weights = base_dir / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt'
    data_path = base_dir / 'my_data.yaml'
    model = YOLO(str(weights))
    model.val(split='test', data=str(data_path), workers=4, batch=16)
