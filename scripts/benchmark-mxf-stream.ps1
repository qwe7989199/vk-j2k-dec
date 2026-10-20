param(
    [Parameter(Mandatory=$true)][string]$InputMxf,
    [string]$OutputDir = 'target\benchmarks\artglass-stream-runs',
    [int]$Frames = 0,
    [int]$Batch = 2,
    [int]$Rounds = 3,
    [int]$WarmupFrames = 120
)
$ErrorActionPreference = 'Stop'
if ($Frames -lt 0 -or $Batch -lt 1 -or $Batch -gt 8 -or $Rounds -lt 1 -or $WarmupFrames -lt 0) { throw 'Invalid benchmark parameters' }
if (Test-Path "$OutputDir\results.csv") { throw 'Choose a fresh output directory to keep independent runs separate' }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$saved = @{}
Get-ChildItem Env:VK_J2K* | ForEach-Object { $saved[$_.Name] = $_.Value; Remove-Item -LiteralPath "Env:$($_.Name)" }
try {
    $script:expectedCount = $null
    $script:expectedBytes = $null
    function Run-Decode([string]$Label, [string]$Backend, [int]$Count) {
        $watch = [Diagnostics.Stopwatch]::StartNew()
        $lines = & target\release\vk-j2k-dec.exe bench-mxf-stream $InputMxf $Backend $Count $Batch NUL 2>&1
        $exitCode = $LASTEXITCODE
        $watch.Stop()
        $lines | Set-Content "$OutputDir\$Label.log"
        $lines | ForEach-Object { Write-Output $_ }
        if ($exitCode -ne 0) { throw "$Label failed with exit code $exitCode" }
        $text = $lines -join "`n"
        $framesMatch = [regex]::Match($text, 'frames (\d+) output_bytes (\d+) essence_bytes (\d+)')
        $loop = [regex]::Match($text, 'stream_loop_ms ([\d.]+) fps ([\d.]+)')
        $total = [regex]::Match($text, 'end_to_end_ms ([\d.]+) fps ([\d.]+)')
        $batchMatch = [regex]::Match($text, 'sequential, batch (\d+), slots 2')
        if (-not $framesMatch.Success -or -not $loop.Success -or -not $total.Success -or -not $batchMatch.Success) { throw "$Label missing benchmark output" }
        if ([int]$batchMatch.Groups[1].Value -ne $Batch) { throw "$Label reduced batch size; use that smaller batch for both backends" }
        $decoded = [int]$framesMatch.Groups[1].Value
        $bytes = [long]$framesMatch.Groups[2].Value
        if ($Label.StartsWith('round-')) {
            if ($null -ne $script:expectedCount -and ($decoded -ne $script:expectedCount -or $bytes -ne $script:expectedBytes)) { throw "$Label frame/output count differs from previous run" }
            $script:expectedCount = $decoded
            $script:expectedBytes = $bytes
        }
        [pscustomobject]@{
            label=$Label;backend=$Backend;batch=$Batch;frames=$decoded;output_bytes=$bytes;
            loop_ms=[double]$loop.Groups[1].Value;loop_fps=[double]$loop.Groups[2].Value;
            end_to_end_ms=[double]$total.Groups[1].Value;end_to_end_fps=[double]$total.Groups[2].Value;
            process_ms=$watch.Elapsed.TotalMilliseconds;process_fps=$decoded/$watch.Elapsed.TotalSeconds
        } | Export-Csv "$OutputDir\results.csv" -Append -NoTypeInformation
    }
    if ($WarmupFrames -gt 0) {
        Run-Decode 'warmup-vulkan' 'vulkan' $WarmupFrames
        Run-Decode 'warmup-nvj2k' 'nvj2k' $WarmupFrames
    }
    foreach ($round in 1..$Rounds) {
        $order = if ($round % 2 -eq 1) { @('vulkan','nvj2k') } else { @('nvj2k','vulkan') }
        foreach ($backend in $order) { Run-Decode "round-$round-$backend" $backend $Frames }
    }
} finally {
    Get-ChildItem Env:VK_J2K* | ForEach-Object { Remove-Item -LiteralPath "Env:$($_.Name)" }
    foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
}
