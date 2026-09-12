param([switch]$PackageOnly)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
function Invoke-Tex($directory, $document, $bibliography) {
    Push-Location (Join-Path $repo $directory)
    try {
        & pdflatex -interaction=nonstopmode -halt-on-error "$document.tex" | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "First TeX pass failed: $directory/$document" }
        if ($bibliography) {
            & bibtex $document | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "BibTeX failed: $directory/$document" }
        }
        1..2 | ForEach-Object {
            & pdflatex -interaction=nonstopmode -halt-on-error "$document.tex" | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "TeX pass failed: $directory/$document" }
        }
        $log = Get-Content "$document.log" -Raw
        if ($log -match 'There were undefined references|Citation .+ undefined|Reference .+ undefined') {
            throw "Unresolved references: $directory/$document"
        }
    } finally { Pop-Location }
}
if (!$PackageOnly) {
    Push-Location (Join-Path $repo 'Reports/paper_en/figure_sources')
    try {
        foreach ($figure in @('gate8_pipeline_overview_en','gate8_primary_speedup_f1_en')) {
            & pdflatex -interaction=nonstopmode -halt-on-error '-output-directory=../figures' "$figure.tex" | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "Vector figure build failed: $figure" }
        }
    } finally { Pop-Location }
    Invoke-Tex 'Reports/paper_en' 'supplement' $false
    Invoke-Tex 'Reports/paper_en' 'main' $true
    Invoke-Tex 'Reports/paper' 'main' $true
    Invoke-Tex 'Reports' 'main' $true
}
$deliveries = @{
    'Reports/paper_en/main.pdf' = 'SleepTCN_Scientific_Article_EN.pdf'
    'Reports/paper_en/supplement.pdf' = 'SleepTCN_Supplement_EN.pdf'
    'Reports/paper/main.pdf' = 'SleepTCN_Scientific_Article_VI.pdf'
    'Reports/main.pdf' = 'SleepTCN_Gate1_8_SHHS_Report.pdf'
}
foreach ($source in $deliveries.Keys) {
    Copy-Item -LiteralPath (Join-Path $repo $source) -Destination (Join-Path $repo "Reports/output/pdf/$($deliveries[$source])") -Force
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.IO.Compression
$zipPath = Join-Path $repo 'Reports/output/source/SleepTCN_BSPC_Manuscript_Source.zip'
# Replacing only the generated, explicitly named submission archive.
$stream = [IO.File]::Open($zipPath, [IO.FileMode]::Create)
$archive = New-Object IO.Compression.ZipArchive($stream, [IO.Compression.ZipArchiveMode]::Create)
try {
    $entries = @('main.tex','supplement.tex','references.bib','highlights.txt',
        'figures/gate8_pipeline_overview_en.pdf','figures/gate8_primary_speedup_f1_en.pdf',
        'figures/gate8_feature_silhouette_en.pdf','figures/gate8_context_ablation_effects_en.pdf',
        'figure_sources/gate8_primary_speedup_f1_en.tex','figure_sources/gate8_pipeline_overview_en.tex')
    $entries += @(Get-ChildItem -LiteralPath (Join-Path $repo 'Reports/paper_en') -Filter '*.bst' | ForEach-Object { $_.Name })
    foreach ($entry in $entries) {
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, (Join-Path $repo "Reports/paper_en/$entry"), $entry) | Out-Null
    }
    [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, (Join-Path $repo 'Reports/paper_en/SUBMISSION_BUILD.md'), 'README.md') | Out-Null
} finally { $archive.Dispose(); $stream.Dispose() }
# Refresh packaging manifests only. Never rewrite experimental protocol/run hashes.
foreach ($relative in @('Reports/paper_en/SUBMISSION_MANIFEST.sha256','Reports/REPORT_MANIFEST.sha256')) {
    $manifest = Join-Path $repo $relative
    $base = Split-Path -Parent $manifest
    $paths = @(Get-Content -LiteralPath $manifest | ForEach-Object {
        if ($_ -match '^[a-fA-F0-9]{64}\s+(.+)$') { $Matches[1] }
    })
    if ($relative -like '*SUBMISSION_MANIFEST*') {
        $paths += @('figures/gate8_pipeline_overview_en.pdf','figures/gate8_primary_speedup_f1_en.pdf',
            'figure_sources/gate8_pipeline_overview_en.tex',
            'figure_sources/gate8_primary_speedup_f1_en.tex','../BSPC_SUBMISSION_CHECKLIST_20260911.md',
            '../../scripts/rebuild_bspc_delivery.ps1')
        $paths += @(Get-ChildItem -LiteralPath $base -Filter '*.bst' | ForEach-Object { $_.Name })
    }
    $lines = foreach ($path in ($paths | Select-Object -Unique)) {
        $hash = (Get-FileHash -LiteralPath (Join-Path $base $path) -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash  $path"
    }
    [IO.File]::WriteAllLines($manifest, $lines, (New-Object Text.UTF8Encoding($false)))
}
Write-Output 'Four delivery PDFs copied; source ZIP and packaging manifests refreshed.'
