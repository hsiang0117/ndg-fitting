"""Run directories and image/progress output shared by training and launchers."""
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from PIL import Image


def create_run_directory(root='output'):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for suffix in range(1000):
        path = root / (stamp if suffix == 0 else f'{stamp}_{suffix:02d}')
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError('Could not allocate a unique run directory')


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def save_image(path, tensor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = tensor.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.ascontiguousarray(pixels)).save(path)


def save_training_image(directory, iteration, image, camera_name):
    safe_name = str(camera_name).replace('/', '_').replace('\\', '_')
    path = Path(directory) / 'render_test' / f'iter_{iteration:06d}_{safe_name}.png'
    save_image(path, image)
    return path
