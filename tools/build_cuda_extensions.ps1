param(
    [string]$CudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8',
    [string]$Toolset = '14.44',
    [string]$CudaArch = '12.0',
    [int]$Jobs = 4,
    [switch]$PatchTorchHeaders
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Create this repository venv first.' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vsRoot) { throw 'Visual Studio C++ tools not found.' }
$vcvars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvarsall.bat'
if (!(Test-Path -LiteralPath "$CudaPath\bin\nvcc.exe")) { throw 'CUDA toolkit not found.' }
$env:VSLANG = '1033'
& cmd.exe /d /c "call `"$vcvars`" x64 -vcvars_ver=$Toolset >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($matches[1])" $matches[2] }
}
if ($LASTEXITCODE -ne 0) { throw 'MSVC initialization failed.' }
& cmd.exe /d /c 'chcp 936 >nul'
if ($LASTEXITCODE -ne 0) { throw 'Run in a normal Windows console.' }
$env:DISTUTILS_USE_SDK = '1'
$env:CUDA_HOME = $CudaPath
$env:TORCH_CUDA_ARCH_LIST = $CudaArch
$env:MAX_JOBS = "$Jobs"
$env:PATH = "$(Split-Path $python -Parent);$CudaPath\bin;$env:PATH"
New-Item -ItemType Directory -Force (Join-Path $repo '.local_setup') | Out-Null
Push-Location -LiteralPath $repo
try {
    & $python -I -c 'import sys,torch; from pathlib import Path; assert Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); print(sys.executable,torch.__version__,torch.version.cuda)'
    if ($LASTEXITCODE -ne 0) { throw 'Wrong Python/PyTorch environment.' }
    if ($PatchTorchHeaders) {
        & $python -I tools/patch_torch_header.py
        if ($LASTEXITCODE -ne 0) { throw 'Header patch validation failed.' }
    }
    foreach ($extension in @('simple-knn', 'diff-gaussian-rasterization')) {
        & $python -I -m pip install --no-build-isolation --no-deps --force-reinstall "ext/$extension" --log ".local_setup/$extension-build.log"
        if ($LASTEXITCODE -ne 0) { throw "Build failed: $extension" }
    }
    & $python -I -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
} finally {
    Pop-Location
}
