param(
    [Parameter(Mandatory = $true)]
    [string]$RunLabel,
    [string[]]$ExtraArgs = @(),
    [string]$Distro = "Ubuntu-MIMO"
)

$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runName = "$RunLabel-$timestamp"
$outputDir = Join-Path "C:\Users\taniguchi\MahjongLM\outputs" $runName
$stopFile = Join-Path $outputDir "STOP"
$stdoutLog = Join-Path $outputDir "stdout.log"
$stderrLog = Join-Path $outputDir "stderr.log"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$workspaceLinux = "/mnt/c/Users/taniguchi/MahjongLM"
$outputDirLinux = $outputDir -replace '^C:', '/mnt/c'
$outputDirLinux = $outputDirLinux -replace '\\', '/'
$stopFileLinux = $stopFile -replace '^C:', '/mnt/c'
$stopFileLinux = $stopFileLinux -replace '\\', '/'
$stdoutLogLinux = $stdoutLog -replace '^C:', '/mnt/c'
$stdoutLogLinux = $stdoutLogLinux -replace '\\', '/'
$stderrLogLinux = $stderrLog -replace '^C:', '/mnt/c'
$stderrLogLinux = $stderrLogLinux -replace '\\', '/'
$wandbApiKey = $env:WANDB_API_KEY

$baseArgs = @(
    "scripts/train_qwen3.py",
    "--model-family", "qwen3",
    "--qwen-arch", "custom",
    "--qwen-hidden-size", "768",
    "--qwen-intermediate-size", "2304",
    "--qwen-num-hidden-layers", "14",
    "--qwen-num-attention-heads", "12",
    "--qwen-num-key-value-heads", "6",
    "--qwen-head-dim", "64",
    "--qwen-max-position-embeddings", "8192",
    "--packing-mode", "unpadded",
    "--attn-implementation", "flash_attention_2",
    "--per-device-train-batch-size", "40",
    "--per-device-eval-batch-size", "8",
    "--gradient-accumulation-steps", "1",
    "--optimizer-name", "muon",
    "--use-muon-plus",
    "--learning-rate", "3e-2",
    "--muon-aux-learning-rate", "3e-2",
    "--warmup-steps", "100",
    "--train-steps", "400",
    "--train-split-eval-ratio", "0.001",
    "--eval-interval", "20",
    "--save-interval", "200",
    "--label-smoothing", "0.01",
    "--gradient-checkpointing",
    "--separate-eval-process",
    "--eval-device", "cpu",
    "--mamba3-chunk-size", "32",
    "--wandb-project", "mahjongLM_qwen_arch",
    "--wandb-run-name", $runName,
    "--output-dir", $outputDirLinux,
    "--stop-file", $stopFileLinux
)

$quotedArgs = @($baseArgs + $ExtraArgs) | ForEach-Object {
    "'" + ($_ -replace "'", "'\\''") + "'"
}
$argString = [string]::Join(" ", $quotedArgs)

$bashCommand = @"
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin:`$PATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export COMPILER_PATH=/usr/bin
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu
$(if ($wandbApiKey) { "export WANDB_API_KEY='$($wandbApiKey -replace "'", "'\\''")'" })
cd '$workspaceLinux'
source /root/mimo-venv/bin/activate
python $argString > '$stdoutLogLinux' 2> '$stderrLogLinux'
"@

$process = Start-Process `
    -FilePath "wsl.exe" `
    -ArgumentList @("-d", $Distro, "-u", "root", "--", "bash", "-lc", $bashCommand) `
    -WorkingDirectory "C:\Users\taniguchi\MahjongLM" `
    -PassThru

Write-Output "run_name=$runName"
Write-Output "pid=$($process.Id)"
Write-Output "output_dir=$outputDir"
Write-Output "distro=$Distro"
