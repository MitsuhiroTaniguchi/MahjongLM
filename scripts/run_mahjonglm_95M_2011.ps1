param()

$ErrorActionPreference = "Stop"

$extra = @(
  "--qwen-arch", "Q20E2S64",
  "--use-gated-attention",
  "--use-mamba3-hybrid",
  "--mamba3-chunk-size", "16",
  "--dataset-dir", "data/processed/2011",
  "--per-device-train-batch-size", "32",
  "--per-device-eval-batch-size", "32",
  "--gradient-accumulation-steps", "4",
  "--train-steps", "7500",
  "--eval-interval", "100",
  "--early-stopping-patience", "0",
  "--lr-scheduler-type", "linear",
  "--wandb-project", "mahjongLM_pretraining",
  "--wandb-run-name", "mahjonglm-95M"
)

& "$PSScriptRoot\launch_qwen_mamba_wsl.ps1" `
  -RunLabel "mahjonglm-95M-2011-linear" `
  -ExtraArgs $extra
