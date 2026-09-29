"""Exercise Taichi/PyTorch interop and the actual N-DG 6D training path."""
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'splatting')]

import numpy as np
import torch
import taichi as ti
from evaluators.lsh_evaluator import EvaluatorLSH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--taichi-only', action='store_true')
    args = parser.parse_args()
    assert Path(sys.prefix).resolve() == (ROOT / '.venv').resolve(), 'Use this repository venv'
    assert Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert torch.cuda.is_available()
    torch.manual_seed(123)
    # Explicitly disallow Taichi's CPU fallback.
    ti.init(arch=ti.cuda, enable_fallback=False, device_memory_GB=0.1)
    assert ti.lang.impl.current_cfg().arch == ti.cuda
    q = torch.randn(257, 8, device='cuda')
    centers = torch.randn_like(q) * 0.2
    widths = torch.full_like(q, 3.0)
    actual = EvaluatorLSH.cull(q.contiguous(), centers.contiguous(), widths.contiguous())
    expected = ((q >= centers - widths / 2) & (q <= centers + widths / 2)).all(dim=1)
    torch.cuda.synchronize()
    assert expected.any() and (~expected).any()
    assert torch.equal(actual.bool(), expected), 'Taichi culling differs from PyTorch reference'
    result = dict(python=sys.version, torch=torch.__version__, taichi=ti.__version__,
                  gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                  taichi_cuda_culling='passed', python_executable=sys.executable)
    if not args.taichi_only:
        from simple_knn._C import distCUDA2
        from generators.gaussian_generator import GaussianGenerator
        from splatting.camera import Camera
        from splatting.render import render
        from losses import DssimL1Loss
        from utils import sigmoid, inverse_sigmoid

        extension_paths = {}
        for name in ('simple_knn._C', 'diff_gaussian_rasterization._C'):
            location = Path(importlib.import_module(name).__file__).resolve()
            assert location.is_relative_to(Path(sys.prefix).resolve()), location
            extension_paths[name] = str(location)

        points = torch.randn(64, 3, device='cuda') * 0.25
        points[:, 2] += 3.0
        distances = distCUDA2(points)
        reference = torch.cdist(points, points).square()
        reference.fill_diagonal_(float('inf'))
        reference = reference.topk(3, largest=False).values.mean(dim=1)
        torch.testing.assert_close(distances, reference, atol=2e-5, rtol=2e-3)

        directions = points / points.norm(dim=-1, keepdim=True)
        directions = directions + 0.15  # Exercise nonzero angular gradients.
        init = torch.cat([points, directions], dim=1)
        scales = torch.cat([torch.full_like(points, 0.15), torch.ones_like(points)], dim=1)
        model = GaussianGenerator(6, init, sigmoid, inverse_sigmoid,
                                  lambda x: x, lambda x: x, 0.05, 0.005, 0.025,
                                  n_projection_vectors=8, cov_bias=scales,
                                  init_ws=0.25, init_fs=0.6)
        assert ti.lang.impl.current_cfg().arch == ti.cuda
        camera = Camera(0, np.eye(3), np.zeros(3), 1.0, 1.0, 0.0, 0.0,
                        torch.zeros(3, 64, 64), None, 'smoke', 0)
        background = torch.zeros(3, device='cuda')
        with torch.no_grad():
            initial, _ = render(camera, model, background)
            target = initial * 0.7
        assert initial.max() > 0 and torch.isfinite(initial).all()
        loss_fn = DssimL1Loss(permute=False)
        losses = []
        gradient_norms = {}
        for step in range(30):
            model.optimizer.zero_grad()
            prediction, _ = render(camera, model, background)
            loss = loss_fn(prediction[None], target[None])
            assert torch.isfinite(loss)
            loss.backward()
            if step == 0:
                for name in ('m', 'diags', 'l_triangs', 'ws', 'fs'):
                    grad = getattr(model, name).grad
                    assert grad is not None and torch.isfinite(grad).all(), name
                    gradient_norms[name] = grad.norm().item()
                    assert gradient_norms[name] > 0, name
            model.optimizer.step()
            model.ema.update()
            losses.append(loss.item())
        with torch.no_grad():
            final, _ = render(camera, model, background)
            final_loss = loss_fn(final[None], target[None]).item()
        assert final_loss < losses[0], (losses[0], final_loss)
        result.update(knn='passed', render_shape=list(final.shape), steps=30,
                      initial_loss=losses[0], final_loss=final_loss,
                      gradient_norms=gradient_norms, losses=losses,
                      extension_paths=extension_paths)
        torch.cuda.synchronize()
    output = ROOT / '.local_setup' / ('taichi-check.json' if args.taichi_only else 'smoke-result.json')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
