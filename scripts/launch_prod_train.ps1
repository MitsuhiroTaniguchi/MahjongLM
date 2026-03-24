param(
    [ValidateSet("A", "B", "C", "D")]
    [string]$Arch = "A",
    [string]$RunName,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir
)

$ErrorActionPreference = "Stop"
$apiKey = [Environment]::GetEnvironmentVariable("WANDB_API_KEY", "User")
if (-not $apiKey) {
    throw "WANDB_API_KEY is missing in the user environment."
}

$env:WANDB_API_KEY = $apiKey

.venv\Scripts\wandb.exe login --relogin $apiKey | Out-Null

Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not $RunName) {
    $RunName = "gpt2-$Arch-y2021-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$stdout = Join-Path $OutputDir "stdout.log"
$stderr = Join-Path $OutputDir "stderr.log"

$pythonExe = '.venv\Scripts\python.exe'
$args = @(
    '-u',
    'scripts\train_gpt2.py',
    '--arch', $Arch,
    '--attn-implementation', 'sdpa',
    '--n-positions', '8192',
    '--max-seq-length', '8192',
    '--max-tokens-per-batch', '98304',
    '--eval-max-tokens-per-batch', '65536',
    '--train-steps', '1400',
    '--train-split-eval-ratio', '0.001',
    '--eval-interval', '100',
    '--save-interval', '100',
    '--log-interval', '10',
    '--learning-rate', '1e-3',
    '--gradient-checkpointing',
    '--wandb-mode', 'online',
    '--eval-device', 'cuda',
    '--wandb-run-name', $RunName,
    '--output-dir', $OutputDir
)

$process = Start-Process -FilePath $pythonExe `
    -ArgumentList $args `
    -WorkingDirectory (Get-Location) `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -NoNewWindow `
    -PassThru `
    -Wait

if ($process.ExitCode -ne 0) {
    exit $process.ExitCode
}
