param(
    [Parameter(Mandatory=$true)][string]$InputMxf,
    [Parameter(Mandatory=$true)][string]$FFmpeg,
    [string]$OutputDir = 'target\benchmarks\rtx5060-nvenc-20261020',
    [int]$Frames = 240,
    [int]$Rounds = 6,
    [string]$Preset = 'p4',
    [int]$Qp = 23,
    [ValidateSet("ffmpeg", "scan")][string]$Demux = "ffmpeg",
    [string]$ExpectedHash = "",
    [switch]$SkipHash
)
$ErrorActionPreference = 'Stop'
if ($Frames -le 0 -or $Rounds -le 0) { throw 'Frames and rounds must be positive' }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$saved = @{}
Get-ChildItem Env:VK_J2K* | ForEach-Object { $saved[$_.Name] = $_.Value; Remove-Item -LiteralPath "Env:$($_.Name)" }
try {
    $env:VK_J2K_DEMUX = $Demux
    $script:verifiedHash = $null
    $env:VK_J2K_NVENC_PRESET = $Preset
    $env:VK_J2K_NVENC_QP = "$Qp"
    function Run-Encode([string]$Label, [string]$Mode, [bool]$Verify = $false) {
        Remove-Item Env:VK_J2K_BATCH,Env:VK_J2K_CPU_CBLKS,Env:VK_J2K_HASH_FRAMES -ErrorAction SilentlyContinue
        if ($Mode -eq 'gpu') { $env:VK_J2K_BATCH = '8'; $env:VK_J2K_CPU_CBLKS = '0' }
        if ($Verify) { $env:VK_J2K_HASH_FRAMES = '1' }
        $env:VK_J2K_NVENC_OUTPUT = "$OutputDir\$Label.mp4"
        $watch = [Diagnostics.Stopwatch]::StartNew()
        $lines = & target\release\vk-j2k-dec.exe profile-mxf-ffmpeg $InputMxf $Frames 0 gpu-tier1 "$OutputDir\unused.rgb48le" $FFmpeg "$OutputDir\demux" 2>&1
        $exitCode = $LASTEXITCODE
        $watch.Stop()
        $lines | Set-Content "$OutputDir\$Label.log"
        $lines | ForEach-Object { Write-Output $_ }
        $text = $lines -join "`n"
        if ($exitCode -ne 0) { throw "$Label exited with $exitCode" }
        if ($Verify) {
            $hash = [regex]::Match($text, 'frames_hash ([0-9a-f]{16})').Groups[1].Value
            if (-not $hash) { throw "$Label missing verification hash" }
            if ($ExpectedHash -and $hash -ne $ExpectedHash) { throw "$Label reference hash mismatch" }
            if ($script:verifiedHash -and $hash -ne $script:verifiedHash) { throw "$Label GPU/auto hash mismatch" }
            $script:verifiedHash = $hash
        }
        $loop = [regex]::Match($text, 'transcode_loop_ms ([\d.]+) avg_ms ([\d.]+) fps ([\d.]+)')
        $total = [regex]::Match($text, 'total_with_demux_setup_ms ([\d.]+) fps ([\d.]+)')
        if (-not $loop.Success -or -not $total.Success) { throw "$Label missing timing output" }
        $row = [pscustomobject]@{label=$Label;mode=$Mode;frames=$Frames;loop_ms=[double]$loop.Groups[1].Value;loop_fps=[double]$loop.Groups[3].Value;total_ms=[double]$total.Groups[1].Value;total_fps=[double]$total.Groups[2].Value;process_ms=$watch.Elapsed.TotalMilliseconds;process_fps=$Frames/$watch.Elapsed.TotalSeconds}
        $row | Export-Csv "$OutputDir\results.csv" -Append -NoTypeInformation
    }
    if (-not $SkipHash) { Run-Encode 'hash-gpu' 'gpu' $true; Run-Encode 'hash-auto' 'auto' $true }
    Run-Encode 'warmup-gpu' 'gpu'
    Run-Encode 'warmup-auto' 'auto'
    foreach ($round in 1..$Rounds) {
        $modes = if ($round % 2 -eq 1) { @('gpu','auto') } else { @('auto','gpu') }
        foreach ($mode in $modes) { Run-Encode "round-$round-$mode" $mode }
    }
} finally {
    Get-ChildItem Env:VK_J2K* | ForEach-Object { Remove-Item -LiteralPath "Env:$($_.Name)" }
    foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
}
