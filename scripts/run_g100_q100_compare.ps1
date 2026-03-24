param(
    [int]$TrainSteps = 400
)

$ErrorActionPreference = "Stop"
$apiKey = [Environment]::GetEnvironmentVariable("WANDB_API_KEY", "User")
if (-not $apiKey) {
    throw "WANDB_API_KEY is missing in the user environment."
}

$env:WANDB_API_KEY = $apiKey

Set-Location (Split-Path -Parent $PSScriptRoot)

$pythonExe = '.venv\Scripts\python.exe'
$projectName = 'mahjongLM_100m_compare'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'

function Start-CompareRun {
    param(
        [Parameter(Mandatory = $true)][string]$Family,
        [Parameter(Mandatory = $true)][string]$Preset,
        [Parameter(Mandatory = $true)][string]$RunName
    )

    $outputDir = Join-Path 'outputs' $RunName
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

    $stdout = Join-Path $outputDir 'stdout.log'
    $stderr = Join-Path $outputDir 'stderr.log'
    $stopFile = Join-Path $outputDir 'STOP'

    $args = @(
        '-u'
    )

    if ($Family -eq 'gpt2') {
        $args += @(
            'scripts\train_gpt2.py',
            '--model-family', 'gpt2',
            '--arch', $Preset
        )
    }
    elseif ($Family -eq 'qwen3') {
        $args += @(
            'scripts\train_qwen3.py',
            '--model-family', 'qwen3',
            '--qwen-arch', $Preset
        )
    }
    else {
        throw "Unsupported family: $Family"
    }

    $args += @(
        '--packing-mode', 'unpadded',
        '--attn-implementation', 'flash_attention_2',
        '--max-seq-length', '8192',
        '--per-device-train-batch-size', '8',
        '--per-device-eval-batch-size', '2',
        '--train-steps', $TrainSteps.ToString(),
        '--train-epochs', '0',
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
        '--wandb-project', $projectName,
        '--eval-device', 'cuda',
        '--stop-file', $stopFile,
        '--wandb-run-name', $RunName,
        '--output-dir', $outputDir
    )

    $process = Start-Process -FilePath $pythonExe `
        -ArgumentList $args `
        -WorkingDirectory (Get-Location) `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -NoNewWindow `
        -PassThru

    Write-Output ("RUN_NAME=" + $RunName)
    Write-Output ("OUTPUT_DIR=" + $outputDir)
    Write-Output ("STOP_FILE=" + $stopFile)
    Write-Output ("PID=" + $process.Id)

    $process.WaitForExit()
    if ($process.ExitCode -ne 0) {
        throw "Run failed for $RunName with exit code $($process.ExitCode)"
    }
}

$gptRun = "gpt2-g100-y2021-lr1e-3-$timestamp"
$qwenRun = "qwen3-q100-y2021-lr1e-3-$timestamp"

Start-CompareRun -Family 'gpt2' -Preset 'G100' -RunName $gptRun
Start-CompareRun -Family 'qwen3' -Preset 'Q100' -RunName $qwenRun
