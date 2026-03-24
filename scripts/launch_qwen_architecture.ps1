param(
    [Parameter(Mandatory = $true)]
    [string]$RunLabel,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runName = "$RunLabel-$timestamp"
$outputDir = Join-Path "C:\Users\taniguchi\MahjongLM\outputs" $runName
$stopFile = Join-Path $outputDir "STOP"
$stdoutLog = Join-Path $outputDir "stdout.log"
$stderrLog = Join-Path $outputDir "stderr.log"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$python = "C:\Users\taniguchi\MahjongLM\.venv\Scripts\python.exe"

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
    "--per-device-eval-batch-size", "40",
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
    "--wandb-project", "mahjongLM_qwen_arch",
    "--wandb-run-name", $runName,
    "--output-dir", $outputDir,
    "--stop-file", $stopFile
)

$allArgs = @($baseArgs + $ExtraArgs)

$process = Start-Process `
    -FilePath $python `
    -ArgumentList $allArgs `
    -WorkingDirectory "C:\Users\taniguchi\MahjongLM" `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Write-Output "run_name=$runName"
Write-Output "pid=$($process.Id)"
Write-Output "output_dir=$outputDir"
