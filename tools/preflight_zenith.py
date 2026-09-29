"""Validate Zenith cameras/splits/PLY and create a small integration fixture."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from splatting.dataset_readers.blender import readNerfSyntheticInfo, readCamerasFromTransforms
from splatting.splatting_utils import store_ply

dataset = Path(sys.argv[1] if len(sys.argv) > 1 else r'D:\dataset\CloudDatasetZenith')
data = readNerfSyntheticInfo(str(dataset), False, True)
train = json.loads((dataset / 'transforms_train.json').read_text())
test = json.loads((dataset / 'transforms_test.json').read_text())
assert len(data.train_cameras) == 73 and len(data.test_cameras) == 36
assert len(data.point_cloud.points) == 200000
assert np.isfinite(data.point_cloud.points).all()
names = [cam.image_name for cam in data.train_cameras + data.test_cameras]
assert len(names) == len(set(names))
camera_errors = []
for cam, frame in zip(data.train_cameras + data.test_cameras, train['frames'] + test['frames']):
    c2w = np.asarray(frame['transform_matrix'])
    # The optical center must map to the origin, and OpenGL -Z must map to +Z.
    center = cam.R.T @ c2w[:3, 3] + cam.T
    forward = cam.R.T @ (c2w[:3, 3] - c2w[:3, 2]) + cam.T
    np.testing.assert_allclose(center, 0, atol=1e-6)
    np.testing.assert_allclose(forward, [0, 0, 1], atol=1e-6)
    camera_errors.append(float(np.abs(center).max()))
first_path = dataset / train['frames'][0]['file_path']
with Image.open(first_path) as source:
    rgba = np.asarray(source.convert('RGBA'), dtype=np.float64)
expected = (rgba[..., :3] * (rgba[..., 3:4] / 255)).astype(np.uint8)
np.testing.assert_array_equal(np.asarray(data.train_cameras[0].image), expected)

fixture = ROOT / 'temporary-build' / 'zenith-smoke-data'
fixture.mkdir(parents=True, exist_ok=True)
for split, source, indices in [('train', train, [0, 18]), ('test', test, [0])]:
    frames = [source['frames'][i] for i in indices]
    for frame in frames:
        dst = fixture / frame['file_path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(dataset / frame['file_path']) as im:
            im.resize((64, 64)).save(dst)
    (fixture / f'transforms_{split}.json').write_text(json.dumps(dict(camera_angle_x=source['camera_angle_x'], frames=frames)))
rng = np.random.default_rng(42)
indices = rng.choice(len(data.point_cloud.points), 512, replace=False)
store_ply(str(fixture / 'points3d.ply'), data.point_cloud.points[indices], data.point_cloud.colors[indices] * 255)

# Regression for extension-less paths and background compositing.
Image.new('RGBA', (4, 4), (255, 0, 0, 0)).save(fixture / 'transparent.png')
(fixture / 'transparent.json').write_text(json.dumps(dict(camera_angle_x=1.0,
    frames=[dict(file_path='transparent', transform_matrix=np.eye(4).tolist())])))
assert np.asarray(readCamerasFromTransforms(str(fixture), 'transparent.json', False)[0].image).max() == 0
assert np.asarray(readCamerasFromTransforms(str(fixture), 'transparent.json', True)[0].image).min() == 255
report = dict(dataset=str(dataset), train_views=73, test_views=36, initial_points=200000,
              resolution=list(data.train_cameras[0].image.size), unique_names=True,
              maximum_camera_center_error=max(camera_errors), pixels_match_gt=True,
              fixture=str(fixture), xyz_min=data.point_cloud.points.min(axis=0).tolist(),
              xyz_max=data.point_cloud.points.max(axis=0).tolist())
(ROOT / 'temporary-build' / 'zenith-preflight.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
