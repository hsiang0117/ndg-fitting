# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: MIT License

import copy
import time
from pathlib import Path

import configargparse
import tqdm
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from simple_knn._C import distCUDA2

from generators.gaussian_generator import GaussianGenerator
from losses import *
from splatting.scene import Scene
from splatting.render import render
from splatting.training_output import create_run_directory, save_training_image, save_image, write_json
from utils import *


def sample_gs(scene, xyz_pcd, device):
    # XYZ
    xyz = xyz_pcd

    # View direction
    dir = torch.randn((xyz_pcd.shape[0], 3), device=device)
    dir = dir / dir.norm(dim=1, keepdim=True)

    params = torch.rand((xyz_pcd.shape[0], scene.total_parameters()), device=device)

    init_samples = torch.cat([xyz, dir, params], dim=-1)

    return init_samples


def run():

    conf = configargparse.ArgumentParser()

    # Directories
    conf.add('--dataset_path', required=True, help='Path to the dataset to be trained on')
    conf.add('--models_path', default='./output', help='Root for timestamped run directories')
    conf.add('--output_dir', default=None, help='Use this exact run directory instead of creating a timestamp')
    conf.add('--iterations', type=int, default=30000)
    conf.add('--save_interval', type=int, default=1000)
    conf.add('--eval_interval', type=int, default=1000)
    conf.add('--image_interval', type=int, default=1000)
    conf.add('--resolution_scale', type=float, default=1.0)
    conf.add('--data_device', default='cpu', help='Storage device for GT images')

    # Misc
    conf.add('--white_background', action='store_true', help='Whether to use white background')
    conf.add('--seed', type=int, default=0, help='Seed for random numbers generator')
    conf.add('--device', type=str, default='cuda', help='Device to use for Pytorch training')
    conf.add('--tensorboard', action='store_true', help='Whether to use tensorboard for visualization')

    conf = conf.parse_args()
    if min(conf.iterations, conf.save_interval, conf.eval_interval, conf.image_interval) < 1:
        raise ValueError('Iterations and intervals must be positive')
    models_path = Path(conf.output_dir) if conf.output_dir else create_run_directory(conf.models_path)
    models_path.mkdir(parents=True, exist_ok=True)
    if (models_path / 'config.json').exists():
        raise FileExistsError(f'Refusing to overwrite an existing run: {models_path}')
    write_json(models_path / 'config.json', vars(conf))
    print(f'Output: {models_path.resolve()}', flush=True)

    # Set random seeds
    np.random.seed(conf.seed)
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed(conf.seed)

    if conf.tensorboard:
        summary_writer = SummaryWriter(str(models_path / 'tensorboard'))

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    criterion_train = DssimL1Loss(permute=False)
    criterion_test = AllMetrics()

    scene = Scene(conf.dataset_path, white_background=conf.white_background,
                  resolution_scales=[conf.resolution_scale], data_device=conf.data_device)
    train_names = [c.image_name for c in scene.train_cameras]
    test_names = [c.image_name for c in scene.test_cameras]
    if set(train_names) & set(test_names):
        raise ValueError('Training and test images overlap')
    write_json(models_path / 'dataset.json', dict(train=train_names, test=test_names,
               initial_points=len(scene.data.point_cloud.points),
               resolution=[scene.train_cameras[0].image_width, scene.train_cameras[0].image_height]))

    xyz_pcd = torch.from_numpy(np.asarray(scene.data.point_cloud.points)).float().to(conf.device)
    rgb_pcd = torch.from_numpy(np.asarray(scene.data.point_cloud.colors)).float().to(conf.device)

    # Randomize colors if they are all zeros
    if rgb_pcd.max() == 0.0:
        rgb_pcd = torch.rand_like(rgb_pcd)

    # Compute scales in xyz
    dist2 = torch.clamp_min(distCUDA2(xyz_pcd), 1e-7)
    scales = (torch.sqrt(dist2))[..., None].repeat(1, 3)

    init_gs = sample_gs(scene, xyz_pcd=xyz_pcd, device=conf.device)

    cov_bias = torch.cat([scales, torch.tensor([[1.0, 1.0, 1.0]], device=conf.device).repeat(scales.shape[0], 1)], dim=-1)

    model = GaussianGenerator(dimensions=init_gs.shape[-1], init_gs=init_gs,
                              ws_act=lambda x: sigmoid(x), ws_act_inv=lambda x: inverse_sigmoid(x),
                              fs_act=lambda x: x, fs_act_inv=lambda x: x,
                              ws_lr=0.05,
                              fs_lr=0.005,
                              m_sb_lr=0.025,
                              init_ws=1e-1,
                              init_fs=rgb_pcd,
                              n_projection_vectors=8,
                              cov_bias=cov_bias)

    train_cameras = scene.train_cameras.copy()
    test_cameras = scene.test_cameras.copy()

    bg_color = [1, 1, 1] if conf.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    losses_log = [0]
    started = time.perf_counter()
    evaluation_history = {}

    for training_iterations in tqdm(range(1, conf.iterations + 1), desc='Training model', mininterval=10):
        iter_start.record()

        model.optimizer.zero_grad()

        # Sample cameras
        train_camera_id = random.randint(0, len(train_cameras)-1)
        train_camera = train_cameras[train_camera_id]

        # Render using splatting
        render_result, cull_percent = render(train_camera, model, background)

        prediction = render_result.unsqueeze(0)
        gt = train_camera.original_image.to(conf.device).unsqueeze(0)

        # Loss
        loss = criterion_train(prediction, gt)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at step {training_iterations}')
        loss.backward()

        model.optimizer.step()
        model.ema.update()

        losses_log.append(loss.item())

        iter_end.record()

        iter_start.synchronize()
        iter_end.synchronize()

        if conf.tensorboard:
            summary_writer.add_scalar('Iteration time', iter_start.elapsed_time(iter_end), training_iterations)
            summary_writer.add_scalar('Cull percent', cull_percent, training_iterations)

        if len(losses_log) > 200:
            losses_log.pop(0)

        if training_iterations % conf.image_interval == 0 or training_iterations == conf.iterations:
            save_training_image(models_path, training_iterations, render_result, train_camera.image_name)

        if training_iterations % 100 == 0 or training_iterations == conf.iterations:
            write_json(models_path / 'progress.json', dict(iteration=training_iterations,
                       total_iterations=conf.iterations, loss=loss.item(),
                       mean_recent_loss=float(np.mean(losses_log)),
                       gaussians=model.n_gs, child_gaussians=model.n_gs_sb,
                       elapsed_seconds=time.perf_counter() - started,
                       cuda_allocated_mb=torch.cuda.memory_allocated() / 2**20,
                       cuda_peak_mb=torch.cuda.max_memory_allocated() / 2**20))

        if training_iterations % conf.save_interval == 0 or training_iterations == conf.iterations:
            with torch.no_grad():
                model_to_save = copy.deepcopy(model)
                # Apply EMA before merging children, which rebuilds the EMA object.
                model_to_save.ema.copy_to()
                model_to_save.finalize_gs()

                save_dict = {
                    'training_iterations': training_iterations,
                    'n_gs': model_to_save.n_gs,
                    'model': model_to_save.state_dict()
                }

                torch.save(save_dict, models_path / f'model{training_iterations}.pth')
                model_to_save.save_ply(str(models_path / f'point_cloud{training_iterations}.ply'))
                del model_to_save, save_dict

        if test_cameras and (training_iterations % conf.eval_interval == 0 or training_iterations == conf.iterations):
            with (torch.no_grad(), model.ema.average_parameters()):
                criterion_test.reset()

                for i, test_camera in enumerate(test_cameras):
                    render_result, _ = render(test_camera, model, background)

                    prediction = render_result.unsqueeze(0)
                    gt = test_camera.original_image.to(conf.device).unsqueeze(0)

                    criterion_test(prediction.permute(0, 2, 3, 1), gt.permute(0, 2, 3, 1))

                    if conf.tensorboard and i == 0:
                        summary_writer.add_images('Test Samples', torch.cat([prediction, gt]).cpu(), global_step=training_iterations)
                    if training_iterations == conf.iterations:
                        save_image(models_path / 'test' / 'renders' / f'{test_camera.image_name}.png', prediction[0])
                        save_image(models_path / 'test' / 'gt' / f'{test_camera.image_name}.png', gt[0])

                # Upstream AllMetrics has an unused LPIPS placeholder; do not report it.
                metrics = {name: float(value / criterion_test.samples)
                           for name, value in criterion_test.metrics.items() if name != 'lpips'}
                metrics['ssim'] = 1.0 - metrics.pop('dssim')
                evaluation_history[str(training_iterations)] = metrics
                write_json(models_path / 'evaluation.json', evaluation_history)
                print(f'\nStep {training_iterations}: test PSNR={metrics["psnr"]:.4f}, SSIM={metrics["ssim"]:.6f}', flush=True)

                if conf.tensorboard:
                    summary_writer.add_scalar('Number of Gaussians', model.n_gs + model.n_gs_sb, training_iterations)

                    summary_writer.add_scalar('Test Loss', criterion_test.metrics['psnr'] / criterion_test.samples, training_iterations)


        if training_iterations == 300:
            model.prune_gs(0.1)
            model.seed_gs(0.1)
        elif training_iterations % 300 == 0:
            model.grow_gs_sb(0.1)
            model.seed_gs(0.1)

    if conf.tensorboard:
        summary_writer.close()
    write_json(models_path / 'training_complete.json', dict(iterations=conf.iterations,
               elapsed_seconds=time.perf_counter() - started,
               final_metrics=evaluation_history.get(str(conf.iterations))))


if __name__ == "__main__":
    run()
