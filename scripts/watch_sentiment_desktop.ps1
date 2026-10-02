$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$Host.UI.RawUI.WindowTitle = 'DATA266 Part 2 - Live Desktop Training'
$taskRepo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $taskRepo
$taskRun = Join-Path $taskRepo 'reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full'
Write-Host 'LIVE PART 2 TRAINING OUTPUT - existing job on RTX 5090' -ForegroundColor Cyan
Write-Host '.venv\Scripts\python.exe scripts/run_sentiment_desktop.py --output reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full --stage all --device cuda'
Write-Host 'This window follows the existing job. Closing it does not stop training.' -ForegroundColor Yellow
$taskLogs = @(
    'training/maxpool_mlp/RUN_LOG.txt',
    'training/bilstm/RUN_LOG.txt',
    'training/dilated_cnn/RUN_LOG.txt',
    'selected/RUN_LOG.txt'
)
$taskSeen = @{}
while ($true) {
    foreach ($taskRelative in $taskLogs) {
        $taskPath = Join-Path $taskRun $taskRelative
        if (-not (Test-Path -LiteralPath $taskPath -PathType Leaf)) { continue }
        $taskStream = [System.IO.File]::Open($taskPath, [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        $taskReader = [System.IO.StreamReader]::new($taskStream, [System.Text.Encoding]::UTF8)
        try { $taskContent = $taskReader.ReadToEnd() } finally { $taskReader.Dispose() }
        $taskAllLines = $taskContent -split '\r?\n'
        # Emit complete lines only; a print may still be appending its newline.
        $taskLines = @($taskAllLines | Select-Object -SkipLast 1)
        if (-not $taskSeen.ContainsKey($taskRelative)) {
            Write-Host "`n--- $taskRelative ---" -ForegroundColor Green
            $taskSeen[$taskRelative] = [Math]::Max(0, $taskLines.Length - 12)
        }
        for ($taskIndex = $taskSeen[$taskRelative]; $taskIndex -lt $taskLines.Length; $taskIndex++) {
            Write-Host $taskLines[$taskIndex]
        }
        $taskSeen[$taskRelative] = $taskLines.Length
    }
    Start-Sleep -Seconds 2
}
