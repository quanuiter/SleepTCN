# SleepTCN SHHS1 holdout - Amendment A1 (AdaBN + input-amplitude sweep, E4 seed 42), CPU.
#  0) checks: no SleepTCN python running; amendment files committed and unchanged
#  1) prepare (AdaBN sample, amplitude statistics; no labels)   2) gate (checkpoint hashes + reproduction)
#  3) N parallel workers (5 conditions x 10 folds)               4) ensembles   5) analysis (once)
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File D:\SleepTCN\scripts\run_holdout_amendment_a1.ps1
#   powershell -ExecutionPolicy Bypass -File D:\SleepTCN\scripts\run_holdout_amendment_a1.ps1 -Workers 4 -Threads 3
# Safe to stop (close the window, then end leftover python.exe in Task Manager); run the same command
# again to continue: finished predictions are kept. Keep the computer awake:
#   powercfg /change standby-timeout-ac 0
param([int]$Workers = 6, [int]$Threads = 2)

$ErrorActionPreference = "Stop"
$Repo = "D:\SleepTCN"
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
$Script = Join-Path $Repo "scripts\run_holdout_amendment_a1.py"
$PreManifest = "E:\research\Dataset\SHHS_v1\manifests\preprocess_holdout_v1.json"
$Out = Join-Path $Repo "runs\holdout_amendment_a1"
$Logs = Join-Path $Out "logs"

foreach ($p in @($Py, $Script, $PreManifest)) { if (-not (Test-Path -LiteralPath $p)) { throw "Not found: $p" } }
Set-Location -LiteralPath $Repo
$env:PYTHONPATH = Join-Path $Repo "src"

# 0) Guards
$running = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "D:\SleepTCN\.venv*" }
if ($running) { throw "SleepTCN python processes are already running; stop them first (Task Manager)." }
foreach ($f in @("scripts/run_holdout_amendment_a1.py", "configs/shhs_holdout_amendment_a1.json",
                 "docs/SHHS_HOLDOUT_AMENDMENT_A1.md")) {
    git ls-files --error-unmatch $f *> $null
    if ($LASTEXITCODE -ne 0) { throw "$f is not committed yet (commit the amendment before running)" }
    git diff --quiet HEAD -- $f
    if ($LASTEXITCODE -ne 0) { throw "$f has uncommitted changes" }
}
$commit = git rev-parse HEAD
Write-Host "Amendment code commit: $commit"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$Common = @($Script, "--preprocess-manifest", $PreManifest, "--output-root", $Out)

Write-Host "== 1/5 Prepare (AdaBN sample, amplitude statistics; no labels read)"
& $Py @Common --mode prepare --threads $Threads
if ($LASTEXITCODE -ne 0) { throw "Prepare failed" }

Write-Host "== 2/5 Gate (checkpoint hashes, reproduction of stored E4_s42 probabilities)"
& $Py @Common --mode gate --threads $Threads
if ($LASTEXITCODE -ne 0) { throw "Gate failed" }

Write-Host "== 3/5 Inference: $Workers workers x $Threads threads"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$procs = @()
for ($i = 0; $i -lt $Workers; $i++) {
    $argList = ($Common + @("--mode", "worker", "--worker-index", "$i", "--num-workers", "$Workers", "--threads", "$Threads")) |
        ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }
    $procs += Start-Process -FilePath $Py -ArgumentList $argList -WorkingDirectory $Repo -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $Logs "worker_${i}_$stamp.out.txt") `
        -RedirectStandardError (Join-Path $Logs "worker_${i}_$stamp.err.txt")
}
$start = Get-Date
$first = $null
while (($procs | Where-Object { -not $_.HasExited }).Count -gt 0) {
    Start-Sleep -Seconds 120
    $done = (& $Py @Common --mode progress --threads 1) | Select-Object -Last 1
    $parts = $done -split "/"
    $n = [int]$parts[0]; $total = [int]$parts[1]
    if ($null -eq $first) { $first = @{ n = $n; t = Get-Date } }
    $eta = ""
    $elapsed = ((Get-Date) - $first.t).TotalSeconds
    if ($elapsed -gt 0 -and $n -gt $first.n) {
        $rate = ($n - $first.n) / $elapsed
        $eta = " | ~{0:N1} h left" -f ((($total - $n) / $rate) / 3600)
    }
    $alive = ($procs | Where-Object { -not $_.HasExited }).Count
    Write-Host ("[{0}] {1}/{2} fold predictions ({3:P1}) | workers running: {4}{5}" -f (Get-Date -Format "HH:mm"), $n, $total, ($n / $total), $alive, $eta)
}
$failed = $procs | Where-Object { $_.ExitCode -ne 0 }
if ($failed) { throw "Some workers stopped with an error; see $Logs (*.err.txt). Re-run to continue." }

Write-Host "== 4/5 Ensembles and manifests (no metrics)"
if (-not (Test-Path (Join-Path $Out "run_manifest.json"))) {
    & $Py @Common --mode finalize --threads $Threads
    if ($LASTEXITCODE -ne 0) { throw "Finalize did not complete; re-run this script to continue" }
}

Write-Host "== 5/5 Pre-specified descriptive analysis (runs once)"
if (Test-Path (Join-Path $Out "analysis\analysis.json")) {
    Write-Host "Analysis already exists: $Out\analysis\analysis.md"
} else {
    & $Py @Common --mode analyze --threads 4
    if ($LASTEXITCODE -ne 0) { throw "Analysis failed" }
    Set-Content -Path (Join-Path $Out "analysis\analysis_code_commit.txt") -Value $commit
}
Write-Host ("Done in {0:N1} h. Report: $Out\analysis\analysis.md" -f (((Get-Date) - $start).TotalHours))
