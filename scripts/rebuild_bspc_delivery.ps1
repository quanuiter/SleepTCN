param([switch]$PackageOnly, [switch]$VerifyOnly, [switch]$StageOnly, [string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$buildArgs = @((Join-Path $repo 'scripts/build_public_delivery.py'))
if ($PackageOnly) { $buildArgs += '--package-only' }
if ($VerifyOnly) { $buildArgs += '--verify-only' }
if ($StageOnly) { $buildArgs += '--stage-only' }
# Python checks native TeX exit status; stderr warnings alone are not failures.
& $Python @buildArgs
if ($LASTEXITCODE -ne 0) { throw "Public delivery build/verification failed ($LASTEXITCODE)" }
