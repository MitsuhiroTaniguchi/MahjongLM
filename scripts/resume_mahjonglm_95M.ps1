param()

$ErrorActionPreference = "Stop"

$extra = @(
  "--qwen-arch", "Q20E2S64",
  "--use-gated-attention",
  "--use-mamba3-hybrid",
  "--mamba3-chunk-size", "16",
  "--train-steps", "0",
  "--train-epochs", "1",
  "--eval-interval", "500",
  "--early-stopping-metric", "eval/perplexity",
  "--early-stopping-patience", "4",
  "--wandb-project", "mahjongLM_pretraining",
  "--wandb-run-id", "miptgs8f",
  "--resume-from-checkpoint", "/mnt/c/Users/taniguchi/MahjongLM/outputs/mahjonglm-95M/checkpoints/step_0001400"
)

foreach ($year in 2011..2024) {
  $extra += @("--dataset-dir", "data/processed/$year")
}

$stopFile = "C:\Users\taniguchi\MahjongLM\outputs\mahjonglm-95M\STOP"
if (Test-Path $stopFile) {
  Remove-Item $stopFile -Force
}

& "$PSScriptRoot\launch_qwen_mamba_wsl.ps1" `
  -RunLabel "mahjonglm-95M" `
  -ExactRunName `
  -ExtraArgs $extra
