# Python 3.12 offline 6D training

This setup targets the 6D radiance-field application, not Mitsuba/10D+ or the
interactive preview. The upstream list is preserved as `requirements-upstream.txt`.
The default `requirements.txt` includes `requirements-train.txt` and pins the
validated CUDA PyTorch build using the official cu128 wheel index.

## Dependency changes

| Component | Upstream | Training environment | Reason |
|---|---|---|---|
| Python | Not pinned | 3.12 | Existing local runtime |
| PyTorch | 2.13.0 | 2.11.0 + CUDA 12.8 | Same CUDA build already used on the local RTX 5060 Laptop; installed independently |
| NumPy | 1.24.2 | 1.26.4 | Python 3.12 wheels without moving to NumPy 2 |
| Taichi | 1.6.0 | 1.7.4 | Python 3.12 wheels; GPU functionality must be tested |
| PyOpenGL | Both 3.1.6 and 3.1.7 | 3.1.7 | Remove contradictory pins |
| imgui | 1.4.1 | Omitted | Only the interactive preview imports it |
| Matplotlib | 3.7.1 | Omitted | Not imported by offline 6D training |
| TensorBoard/Pillow | Not explicitly listed | Included | Required by training/data loading |

GLFW/OpenGL remain installed because the shared utility module imports them.
Offline training does not create a window. A server without the corresponding
system libraries may require moving preview imports into preview functions.

## Installation

Create a repository-local `.venv` with Python 3.12. On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements-train-lock.txt` records the exact Python package versions used in
the successful Windows verification. It excludes pip and the two locally built
CUDA extensions. After installing the CUDA PyTorch wheel, it can replace
`requirements-train.txt` when reproducing this specific environment. Linux
installation and server GPU compatibility have not been tested here.

The local installation reused the previously downloaded official PyTorch wheel
as a package file only. It does not import PyTorch, headers, or compiled extensions
from another project's virtual environment.

For this partial/sparse checkout, include `ext/pytorch_ssim` as well as
`splatting`, `generators`, `evaluators`, `ext/simple-knn`, and
`ext/diff-gaussian-rasterization`. Initialize both CUDA submodules recursively.

## CUDA extensions on Windows

Use Visual Studio's MSVC 14.44 toolset and CUDA Toolkit 12.8. The local GPU has
compute capability 12.0; change `-CudaArch` for a different GPU.

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_cuda_extensions.ps1
```

The unmodified PyTorch 2.11.0 headers reproduced MSVC C2872 in
`torch/csrc/dynamo/compiled_autograd.h`. For this exact Windows build, rerun:

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_cuda_extensions.ps1 -PatchTorchHeaders
```

`tools/patch_torch_header.py` applies the same two audited compatibility edits
used by the local 6DGS environment, but to this project's own PyTorch headers:

- Disable the two-line `std::string` type-mapping branch in
  `compiled_autograd.h` that triggers NVCC/MSVC's `std` ambiguity. This changes
  that compiled-autograd string mapping; this training uses ordinary eager
  autograd, not compiled autograd.
- Rename the `small` constructor parameter in `CUDACachingAllocator.h` to
  `is_small_flag`, avoiding the Windows SDK `small` macro collision. This is
  a parameter-name change only.

The script checks the exact PyTorch version and original/patched SHA256 hashes
before any write, saves originals in `.local_setup/`, and supports recovery:

```powershell
.\.venv\Scripts\python.exe tools/patch_torch_header.py --restore
```

Do not apply this Windows-specific patch to a different PyTorch build or to a
Linux server. No N-DG or rasterizer CUDA kernel is changed by these patches.

The build uses only this repository's interpreter and installed PyTorch.
`--no-build-isolation --no-deps` is used for local extensions after resolving
the environment, so pip does not substitute a different PyTorch build.

## Verification and training entry point

```powershell
.\.venv\Scripts\python.exe tools/verify_environment.py --taichi-only
.\.venv\Scripts\python.exe tools/verify_environment.py
.\.venv\Scripts\python.exe -m splatting.splatting_train --help
.\.venv\Scripts\python.exe -m splatting.splatting_train --dataset_path D:\path\to\dataset --tensorboard
```

Module-qualified imports in the offline training/rendering entry points allow
`python -m` from the repository root without a global PYTHONPATH setting.

The environment check verifies actual Taichi CUDA culling against a PyTorch
reference, KNN against brute force, N-DG rendering and parameter gradients,
and 30 optimization steps with the upstream L1+DSSIM loss. It writes results
under `.local_setup/`. This is an environment smoke test, not a full dataset
reproduction or a guarantee about convergence/metrics on Zenith.

## Validation status

Verified on 2026-09-29 with Python 3.12.10, RTX 5060 Laptop (sm_120),
PyTorch 2.11.0+cu128, Taichi 1.7.4, MSVC 14.44, and CUDA Toolkit 12.8:

- Dependency installation and `pip check`: passed.
- Taichi CUDA culling on PyTorch tensors: matches the reference, with CPU fallback disabled.
- Both CUDA extensions built in this repository's venv; upstream kernel sources unchanged.
- KNN against brute force, 64x64 N-DG rendering, and finite nonzero gradients for
  means, diagonal/off-diagonal covariance parameters, opacity, and color: passed.
- 30-step synthetic optimization: L1+DSSIM decreased from 0.02152324 to 0.00129787.
- Offline training module `--help`: passed without a global PYTHONPATH override.

Taichi emits a non-leaf `.grad` inspection warning while invoking the culling
kernel. The tested parameter gradients and optimization remain valid; the
warning was not suppressed. This check does not validate gradients through
the discrete culling decision itself.

Logs and numerical results are in `.local_setup/`. The checks above initially
covered only the environment. The later Zenith validation is documented below.
Interactive preview and Mitsuba/10D+ remain untested.

## Zenith training (2026-09-29)

The Zenith adapter was subsequently validated with 73 training views, 36 held-out
views, 1024x1024 images, and the existing 200,000-point `points3d.ply`.
Blender loading now handles filenames with extensions, preserves camera IDs in
nested image names, loads existing PLYs, respects the background flag, and uses
the standard OpenGL-to-COLMAP camera conversion. The preflight checks optical
centers, viewing directions, pixel values, and disjoint splits.

```powershell
.\.venv\Scripts\python.exe tools/preflight_zenith.py
.\.venv\Scripts\python.exe -u tools/run_zenith.py
```

The launcher runs 30,000 steps at original resolution. Each run is written to
`output/YYYYMMDD_HHMMSS` (a suffix prevents timestamp collisions), with logs,
configuration, source snapshot, models/PLYs every 1,000 steps, and:

- `render_test/iter_001000_camXX_images_0000.png`: the sampled training view,
  using the existing forward render; no extra training render pass is required.
- `progress.json`: step, recent loss, primitive counts, timing, and GPU memory.
- `evaluation.json`: held-out metrics every 1,000 steps.
- `test/renders` and `test/gt`: all held-out images at the final step.
- `run_status.json`: running/completed/failed, updated by the launcher.

Ground-truth images reside on CPU and only the current image is transferred to
CUDA. This does not resize images or change the training loss. The step counter
now counts completed optimizer steps exactly (the original counter started at
one and incremented again before saving). Model exports apply EMA before
merging child Gaussians, since that merge rebuilds the EMA object; evaluation
also uses EMA. Copied export models are freed immediately after saving.

The upstream evaluator's unused zero-valued LPIPS placeholder is omitted from
reports. A separate, consistent LPIPS evaluation is needed for paper tables.
No LPIPS loss is added to training.

A 320-step integration run on a small Zenith fixture passed, including the
300-step pruning/seeding event, child-Gaussian export, previews, and evaluation.
The current full run is recorded in `.local_setup/zenith-current-run.json`;
consult that run's status rather than interpreting this document as proof of
full training completion.

### Local run status and server handoff

The 20260929_180251 Zenith run was stopped at the user's request at roughly
12,300/30,000 steps. The last saved model is step 12,000. This was not a completed
30k reproduction. Models, logs, datasets and virtual environments are not tracked.

For a Linux server, use Python 3.12 and a CUDA toolkit compatible with the chosen
PyTorch CUDA build. The following is a starting procedure, not a verified Linux
installation. The manifests currently pin the locally tested cu128 build; review
the driver and GPU requirements before installing. Do not apply Windows header
patches on Linux.

```bash
git submodule update --init --recursive ext/simple-knn ext/diff-gaussian-rasterization
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Set CUDA_HOME and TORCH_CUDA_ARCH_LIST for the server, not the laptop.
python -m pip install --no-build-isolation --no-deps ./ext/simple-knn ./ext/diff-gaussian-rasterization
python -m pip check
python -m splatting.splatting_train --dataset_path /path/to/CloudDatasetZenith --iterations 30000 --data_device cpu --tensorboard
```

The module command creates the same timestamped output directory and saves a
training image every 1,000 steps by default. `tools/run_zenith.py` is the local
Windows launcher with its local dataset path; use the module command on a server.
