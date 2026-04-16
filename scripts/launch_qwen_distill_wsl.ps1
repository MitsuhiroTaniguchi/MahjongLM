param(
    [string]$StudentArch = "custom",
    [int]$HiddenSize = 384,
    [int]$IntermediateSize = 1152,
    [int]$NumHiddenLayers = 14,
    [int]$NumAttentionHeads = 6,
    [int]$NumKeyValueHeads = 3,
    [int]$HeadDim = 64,
    [string]$TeacherModelDir = "C:\Users\taniguchi\MahjongLM\outputs\q100-xsa-allyears-1ep-20260405-175045\final_model",
    [double]$DistillationAlpha = 0.5,
    [double]$DistillationTemperature = 2.0,
    [string]$ProjectName = "mahjongLM_distillation",
    [string]$RunPrefix = "qwen-distill",
    [string]$Distro = "Ubuntu-MIMO",
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runLabel = "$RunPrefix-$HiddenSize-h$NumHiddenLayers-l"
$runName = "$runLabel-$timestamp"
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
$teacherModelDirLinux = $TeacherModelDir -replace '^C:', '/mnt/c'
$teacherModelDirLinux = $teacherModelDirLinux -replace '\\', '/'

$datasetDirs = 2011..2024 | ForEach-Object { "/mnt/c/Users/taniguchi/MahjongLM/data/processed/$_" }
$datasetArgs = @()
foreach ($datasetDir in $datasetDirs) {
    $datasetArgs += @("--dataset-dir", $datasetDir)
}

$baseArgs = @(
    "scripts/train_qwen3.py"
    "--model-family", "qwen3"
    "--qwen-arch", $StudentArch
    "--qwen-hidden-size", "$HiddenSize"
    "--qwen-intermediate-size", "$IntermediateSize"
    "--qwen-num-hidden-layers", "$NumHiddenLayers"
    "--qwen-num-attention-heads", "$NumAttentionHeads"
    "--qwen-num-key-value-heads", "$NumKeyValueHeads"
    "--qwen-head-dim", "$HeadDim"
    "--use-exclusive-self-attention"
    "--packing-mode", "unpadded"
    "--attn-implementation", "flash_attention_2"
    "--pad-to-multiple-of", "16"
    "--per-device-train-batch-size", "8"
    "--per-device-eval-batch-size", "8"
    "--gradient-accumulation-steps", "64"
    "--optimizer-name", "muon"
    "--use-muon-plus"
    "--learning-rate", "3e-2"
    "--muon-aux-learning-rate", "3e-2"
    "--warmup-steps", "100"
    "--train-steps", "0"
    "--train-epochs", "1"
    "--train-split-eval-ratio", "0.001"
    "--eval-interval", "100"
    "--save-interval", "200"
    "--lr-scheduler-type", "linear"
    "--label-smoothing", "0.01"
    "--gradient-checkpointing"
    "--eval-device", "cuda"
    "--teacher-model-dir", $teacherModelDirLinux
    "--distillation-alpha", "$DistillationAlpha"
    "--distillation-temperature", "$DistillationTemperature"
    "--wandb-project", $ProjectName
    "--wandb-run-name", $runName
    "--output-dir", $outputDirLinux
    "--stop-file", $stopFileLinux
)

$allArgs = @($baseArgs + $datasetArgs + $ExtraArgs)
$quotedArgs = $allArgs | ForEach-Object {
    "'" + ($_ -replace "'", "'\''") + "'"
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
cd '$workspaceLinux'
source /root/mimo-venv/bin/activate
/root/mimo-venv/bin/python $argString > '$stdoutLogLinux' 2> '$stderrLogLinux'
"@

$process = Start-Process `
    -FilePath "wsl.exe" `
    -ArgumentList @("-d", $Distro, "-u", "root", "--", "bash", "-lc", $bashCommand) `
    -WorkingDirectory "C:\Users\taniguchi\MahjongLM" `
    -PassThru

Write-Output "run_name=$runName"
Write-Output "pid=$($process.Id)"
Write-Output "output_dir=$outputDir"
Write-Output "project=$ProjectName"
Write-Output "teacher_model_dir=$TeacherModelDir"
