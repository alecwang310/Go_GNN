# Optimization: Hide the PowerShell progress bar (helps some background tasks)
$ProgressPreference = 'SilentlyContinue'

$targetDir = "raw_expanded"
if (!(Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir
}

# Define the date range
$startDate = Get-Date "2026-02-15"
$endDate   = Get-Date "2026-03-04"

$currentDate = $startDate

while ($currentDate -le $endDate) {
    $dateStr = $currentDate.ToString("yyyy-MM-dd")
    $fileName = "$($dateStr)npzs.tgz"
    $url = "https://katagoarchive.org/kata1/trainingdata/${fileName}"

    Write-Host "`n--- Processing: ${fileName} ---" -ForegroundColor Cyan

    try {
        # 1. Download using curl.exe (Fastest method)
        Write-Host "Downloading..." -ForegroundColor Gray
        curl.exe -L -o $fileName $url --fail --silent
        
        # Check if curl actually succeeded
        if ($LASTEXITCODE -ne 0) { 
            throw "Download failed or file not found on server." 
        }

        # 2. Extract
        Write-Host "Extracting..." -ForegroundColor Gray
        tar -xf $fileName

        # 3. Move .npz files to target directory
        # We find files NOT already in the target directory and move them
        $extractedFiles = Get-ChildItem -Filter "*.npz" -Recurse | Where-Object { $_.FullName -notlike "*$targetDir*" }
        
        if ($extractedFiles) {
            $extractedFiles | Move-Item -Destination $targetDir -Force
        }

        # 4. Cleanup
        Write-Host "Cleaning up temporary files..." -ForegroundColor Gray
        if (Test-Path $fileName) { Remove-Item $fileName -Force }
        
        # Remove any leftover folders created by the tar extraction (excluding targetDir)
        Get-ChildItem -Directory | Where-Object { $_.Name -ne $targetDir } | Remove-Item -Recurse -Force
        
        Write-Host "Successfully processed ${dateStr}" -ForegroundColor Green
    }
    catch {
        Write-Host "Skipping ${dateStr}: $($_.Exception.Message)" -ForegroundColor Yellow
        # Cleanup partial downloads if any
        if (Test-Path $fileName) { Remove-Item $fileName -Force }
    }

    $currentDate = $currentDate.AddDays(1)
}

# Final count
$totalCount = (Get-ChildItem "${targetDir}" -Filter "*.npz").Count
Write-Host "`n===============================================" -ForegroundColor Magenta
Write-Host "Operation complete. Total files in ${targetDir}: ${totalCount}" -ForegroundColor Magenta