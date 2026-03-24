param(
    [string[]]$ArchList = @("A", "B", "C", "D")
)

$ErrorActionPreference = "Stop"

Write-Warning "scripts\\run_abcd_1000.ps1 is deprecated. Forwarding to scripts\\run_abcd_1400.ps1."
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "scripts\\run_abcd_1400.ps1" -ArchList $ArchList
if ($LASTEXITCODE -ne 0) {
    throw "Run failed with exit code $LASTEXITCODE"
}
