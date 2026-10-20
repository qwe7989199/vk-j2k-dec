param(
    [string]$CudaSdk = $env:CUDA_PATH,
    [string]$NvJpeg2k = (Join-Path $PSScriptRoot '..\refer')
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$output = Join-Path $root 'target\nvj2k'
New-Item -ItemType Directory -Force -Path $output | Out-Null
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio C++ tools not found' }
$devcmd = Join-Path $vs 'Common7\Tools\VsDevCmd.bat'
cmd /c "`"$devcmd`" -arch=x64 -host_arch=x64 >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
& "$CudaSdk\bin\nvcc.exe" -shared -O3 -std=c++17 -arch=sm_89 -Xcompiler /MT -Xcompiler /utf-8 `
    "-I$NvJpeg2k" "$root\native\nvjpeg2k_bridge.cu" "$NvJpeg2k\nvjpeg2k.lib" `
    -o "$output\nvjpeg2k_bridge.dll"
if ($LASTEXITCODE -ne 0) { throw 'nvJPEG2000 benchmark bridge compilation failed' }
Copy-Item -LiteralPath "$NvJpeg2k\nvjpeg2k_0.dll" -Destination $output
Write-Host "Built $output\nvjpeg2k_bridge.dll"
