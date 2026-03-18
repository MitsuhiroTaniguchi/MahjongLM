param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$rawDir = Join-Path $Root 'data\raw\2021'
$processedDir = Join-Path $Root 'data\processed\2021'

New-Item -ItemType Directory -Force -Path $rawDir, $processedDir | Out-Null

Add-Type -AssemblyName System.IO.Compression.FileSystem

$readyMarker = Join-Path $processedDir 'dataset_info.json'
if (Test-Path $readyMarker) {
    Write-Host "Dataset already prepared at $processedDir"
    return
}

$zipFiles = Get-ChildItem -Path $rawDir -Filter '*.zip' | Sort-Object Name
foreach ($zip in $zipFiles) {
    [System.IO.Compression.ZipFile]::ExtractToDirectory($zip.FullName, $processedDir)
}

$nestedDir = Join-Path $processedDir '2021'
if (Test-Path (Join-Path $nestedDir 'dataset_info.json')) {
    Get-ChildItem -Path $nestedDir -Force | ForEach-Object {
        Move-Item -Force $_.FullName $processedDir
    }
}

Write-Host "Dataset prepared at $processedDir"
