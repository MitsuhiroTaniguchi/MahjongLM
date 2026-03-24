param(
    [string]$RunName,
    [string]$OutputDir
)

$ErrorActionPreference = "Stop"
$apiKey = [Environment]::GetEnvironmentVariable("WANDB_API_KEY", "User")
if (-not $apiKey) {
    throw "WANDB_API_KEY is missing in the user environment."
}

$env:WANDB_API_KEY = $apiKey

Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not $RunName) {
    $RunName = "qwen3-q1-y2021-lr1e-3-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}
if (-not $OutputDir) {
    $OutputDir = Join-Path "outputs" $RunName
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$stdout = Join-Path $OutputDir "stdout.log"
$stderr = Join-Path $OutputDir "stderr.log"
$stopFile = Join-Path $OutputDir "STOP"

$pythonExe = '.venv\Scripts\python.exe'
$args = @(
    '-u',
    'scripts\train_qwen3.py',
    '--model-family', 'qwen3',
    '--qwen-arch', 'Q1',
    '--packing-mode', 'unpadded',
    '--attn-implementation', 'flash_attention_2',
    '--max-seq-length', '8192',
    '--per-device-train-batch-size', '8',
    '--per-device-eval-batch-size', '2',
    '--train-steps', '0',
    '--train-epochs', '1',
    '--train-split-eval-ratio', '0.001',
    '--eval-interval', '20',
    '--save-interval', '200',
    '--log-interval', '10',
    '--learning-rate', '1e-3',
    '--warmup-steps', '100',
    '--gradient-accumulation-steps', '64',
    '--label-smoothing', '0.01',
    '--gradient-checkpointing',
    '--wandb-mode', 'online',
    '--wandb-project', 'mahjongLM_qwen3',
    '--eval-device', 'cuda',
    '--stop-file', $stopFile,
    '--wandb-run-name', $RunName,
    '--output-dir', $OutputDir
)

$process = Start-Process -FilePath $pythonExe `
    -ArgumentList $args `
    -WorkingDirectory (Get-Location) `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -NoNewWindow `
    -PassThru

Write-Output ("PID=" + $process.Id)
Write-Output ("RUN_NAME=" + $RunName)
Write-Output ("OUTPUT_DIR=" + $OutputDir)
Write-Output ("STOP_FILE=" + $stopFile)
