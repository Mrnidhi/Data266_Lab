param([switch]$Once, [int]$RefreshSeconds = 2)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$studyRoot = Join-Path $projectRoot 'reproducibility/raw_logs/srinidhi/desktop-quality-20261001'
$seenLines = @{}
$Host.UI.RawUI.WindowTitle = 'DATA266 - Live Part 1 and Part 2 training'
Write-Host 'DATA266 | LIVE PART 1 + PART 2 TRAINING' -ForegroundColor Cyan
Write-Host 'This viewer reads the actual logs. Closing it does not stop training.'
Write-Host 'Part 1: larger character GPT. Part 2: validation-only sentiment search.'
Write-Host ''
do {
    $logs = @()
    $gptLog = Join-Path $studyRoot 'part1/depth_context_full/RUN_LOG.txt'
    if (Test-Path -LiteralPath $gptLog) {
        $logs += [pscustomobject]@{ Path = $gptLog; Label = 'PART 1 GPT'; Color = 'Cyan' }
    }
    $trainingRoot = Join-Path $studyRoot 'part2-search/training'
    if (Test-Path -LiteralPath $trainingRoot) {
        foreach ($folder in (Get-ChildItem -LiteralPath $trainingRoot -Directory | Sort-Object CreationTime)) {
            $log = Join-Path $folder.FullName 'RUN_LOG.txt'
            if (Test-Path -LiteralPath $log) {
                $logs += [pscustomobject]@{ Path = $log; Label = "PART 2 $($folder.Name)"; Color = 'Green' }
            }
        }
    }
    $evaluationLog = Join-Path $studyRoot 'part2-search/selected/RUN_LOG.txt'
    if (Test-Path -LiteralPath $evaluationLog) {
        $logs += [pscustomobject]@{ Path = $evaluationLog; Label = 'PART 2 FINAL EVALUATION'; Color = 'Yellow' }
    }
    # Later verification stages write their real console streams to these
    # local-only logs, so the same window stays useful after training ends.
    foreach ($phaseLog in (Get-ChildItem -LiteralPath (Join-Path $projectRoot '.bootstrap') -Filter 'quality_phase_*.log' -File -ErrorAction SilentlyContinue)) {
        $logs += [pscustomobject]@{
            Path = $phaseLog.FullName
            Label = $phaseLog.BaseName.Replace('quality_phase_', '').Replace('_', ' ').ToUpperInvariant()
            Color = 'Yellow'
        }
    }
    foreach ($item in $logs) {
        try {
            $stream = [System.IO.File]::Open($item.Path, [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
            $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8)
            try { $contents = $reader.ReadToEnd() } finally { $reader.Dispose() }
            $lines = @($contents -split "`n")
            # The final element is either empty or a partially written line.
            $lines = if ($lines.Length -gt 1) { @($lines[0..($lines.Length - 2)]) } else { @() }
            $start = if ($seenLines.ContainsKey($item.Path)) { $seenLines[$item.Path] } else { [Math]::Max(0, $lines.Length - 3) }
            if ($start -gt $lines.Length) { $start = 0 }
            for ($index = $start; $index -lt $lines.Length; $index++) {
                if ($lines[$index].Trim().Length -gt 0) {
                    Write-Host "[$($item.Label)] " -NoNewline -ForegroundColor $item.Color
                    Write-Host $lines[$index]
                }
            }
            $seenLines[$item.Path] = $lines.Length
        } catch [System.IO.IOException] {
            # Retry a transient sharing conflict on the next refresh.
        }
    }
    if (-not $Once) { Start-Sleep -Seconds ([Math]::Max(1, $RefreshSeconds)) }
} while (-not $Once)
