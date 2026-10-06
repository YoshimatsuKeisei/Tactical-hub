$ErrorActionPreference = "Stop"

$ToolDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ToolDir

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' が見つかりません。Python 3.12 をインストールしてください。"
}

if (-not (Test-Path ".venv")) {
    py -3.12 -m venv .venv
}

$Python = Join-Path $ToolDir ".venv\Scripts\python.exe"

& $Python -m pip install --upgrade pip

Write-Host "[1/4] CPU版PyTorchをインストールします..."
& $Python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu

Write-Host "[2/4] GUI依存関係をインストールします..."
& $Python -m pip install -r requirements.txt

Write-Host "[3/4] Meta公式SAM 2を固定commitからインストールします..."
$env:SAM2_BUILD_CUDA = "0"
& $Python -m pip install "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"

Write-Host "[4/4] 公式SAM 2.1 Hiera Tiny checkpointを確認します..."
$ModelDir = Join-Path $ToolDir "models"
$Checkpoint = Join-Path $ModelDir "sam2.1_hiera_tiny.pt"
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

if (-not (Test-Path $Checkpoint)) {
    Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" -OutFile $Checkpoint
}

$Hash = (Get-FileHash $Checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
$Expected = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
if ($Hash -ne $Expected) {
    throw "checkpoint SHA-256 が一致しません: $Hash"
}

Write-Host ""
Write-Host "セットアップ完了。run_windows.bat をダブルクリックして起動できます。"
