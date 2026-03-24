param(
    [string[]]$ArchList = @("A", "B", "C", "D")
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

foreach ($arch in $ArchList) {
    $runName = "gpt2-$arch-y2021-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    $outputDir = Join-Path "outputs" $runName
    Write-Host "Starting $runName"
    powershell.exe -ExecutionPolicy Bypass -NoProfile -File "scripts\\launch_prod_train.ps1" -Arch $arch -RunName $runName -OutputDir $outputDir
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed for $runName with exit code $LASTEXITCODE"
    }
    Write-Host "Finished $runName"
}
