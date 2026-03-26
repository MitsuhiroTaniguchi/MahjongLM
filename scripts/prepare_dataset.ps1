param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = "Stop"

$rawDir = Join-Path $Root 'data\raw'
$processedDir = Join-Path $Root 'data\processed'
$stateDir = Join-Path $processedDir '.ingested'

New-Item -ItemType Directory -Force -Path $rawDir, $processedDir, $stateDir | Out-Null

function Get-SafeMarkerName {
    param([string]$PathText)
    $hash = [System.BitConverter]::ToString(
        [System.Security.Cryptography.SHA256]::Create().ComputeHash(
            [System.Text.Encoding]::UTF8.GetBytes($PathText)
        )
    ).Replace('-', '').ToLowerInvariant()
    return "$hash.done"
}

function Expand-ZipIntoProcessed {
    param([System.IO.FileInfo]$ZipFile)

    $rawUri = [System.Uri]((Resolve-Path $rawDir).Path + [System.IO.Path]::DirectorySeparatorChar)
    $zipUri = [System.Uri]((Resolve-Path $ZipFile.FullName).Path)
    $relativeZipPath = [System.Uri]::UnescapeDataString($rawUri.MakeRelativeUri($zipUri).ToString()).Replace('/', '\')
    $markerPath = Join-Path $stateDir (Get-SafeMarkerName $relativeZipPath)
    if (Test-Path $markerPath) {
        Write-Host "Skip already ingested zip: $relativeZipPath"
        return
    }

    Write-Host "Extracting $relativeZipPath -> $processedDir"
    Expand-Archive -LiteralPath $ZipFile.FullName -DestinationPath $processedDir -Force
    Set-Content -Path $markerPath -Value $relativeZipPath -Encoding UTF8
}

function Move-RawYearDirectory {
    param([System.IO.DirectoryInfo]$YearDir)

    $datasetInfo = Join-Path $YearDir.FullName 'dataset_info.json'
    if (-not (Test-Path $datasetInfo)) {
        return
    }

    $targetDir = Join-Path $processedDir $YearDir.Name
    if ($YearDir.FullName -ieq $targetDir) {
        return
    }
    if (Test-Path (Join-Path $targetDir 'dataset_info.json')) {
        Write-Host "Processed year already exists, skipping raw directory: $($YearDir.Name)"
        return
    }

    Write-Host "Moving prepared year $($YearDir.Name) -> $targetDir"
    Move-Item -Force $YearDir.FullName $targetDir
}

function Repair-NestedYearDirectory {
    param([System.IO.DirectoryInfo]$YearDir)

    $nestedDir = Join-Path $YearDir.FullName $YearDir.Name
    if (-not (Test-Path (Join-Path $nestedDir 'dataset_info.json'))) {
        return
    }

    Write-Host "Flattening nested dataset directory: $nestedDir"
    Get-ChildItem -Path $nestedDir -Force | ForEach-Object {
        Move-Item -Force $_.FullName $YearDir.FullName
    }
    Remove-Item -Force -Recurse $nestedDir
}

$zipFiles = Get-ChildItem -Path $rawDir -Recurse -Filter '*.zip' -File | Sort-Object FullName
foreach ($zip in $zipFiles) {
    Expand-ZipIntoProcessed -ZipFile $zip
}

$rawYearDirs = Get-ChildItem -Path $rawDir -Directory | Sort-Object Name
foreach ($yearDir in $rawYearDirs) {
    Move-RawYearDirectory -YearDir $yearDir
}

$processedYearDirs = Get-ChildItem -Path $processedDir -Directory | Where-Object { $_.Name -ne '.ingested' } | Sort-Object Name
foreach ($yearDir in $processedYearDirs) {
    Repair-NestedYearDirectory -YearDir $yearDir
}

$readyYears = Get-ChildItem -Path $processedDir -Directory |
    Where-Object { $_.Name -ne '.ingested' -and (Test-Path (Join-Path $_.FullName 'dataset_info.json')) } |
    Sort-Object Name |
    Select-Object -ExpandProperty Name

if ($readyYears.Count -eq 0) {
    Write-Host "No processed datasets found under $processedDir"
    exit 1
}

Write-Host "Prepared dataset years: $($readyYears -join ', ')"
