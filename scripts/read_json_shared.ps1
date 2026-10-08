param([Parameter(Mandatory=$true)][string]$Path)
# A status reader must not block a writer's atomic rename on Windows.
$ErrorActionPreference = 'Stop'
$stream = [IO.FileStream]::new([IO.Path]::GetFullPath($Path),[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
try {
    $reader = [IO.StreamReader]::new($stream,[Text.Encoding]::UTF8)
    try { Write-Output $reader.ReadToEnd() } finally { $reader.Dispose() }
} finally { $stream.Dispose() }
