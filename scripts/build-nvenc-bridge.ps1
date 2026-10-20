param(
    [Parameter(Mandatory=$true)][string]$FFmpeg,
    [string]$VulkanSdk = $env:VULKAN_SDK,
    [string]$CudaSdk = $env:CUDA_PATH
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$output = Join-Path $root 'target\nvenc'
New-Item -ItemType Directory -Force -Path $output | Out-Null
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio C++ tools not found' }
$devcmd = Join-Path $vs 'Common7\Tools\VsDevCmd.bat'
cmd /c "`"$devcmd`" -arch=x64 -host_arch=x64 >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
& "$CudaSdk\bin\nvcc.exe" -ptx -O3 -arch=compute_75 "$root\native\rgb48_to_nv12.cu" -o "$output\rgb48_to_nv12.ptx"
if ($LASTEXITCODE -ne 0) { throw 'CUDA kernel compilation failed' }
& cl.exe /nologo /utf-8 /LD /O2 /EHsc /std:c++17 /MD /D_CRT_SECURE_NO_WARNINGS "/I$FFmpeg\include" "/I$VulkanSdk\Include" "/I$CudaSdk\include" "$root\native\nvenc_bridge.cpp" "/Fo$output\nvenc_bridge.obj" "/Fe$output\nvenc_bridge.dll" /link "/LIBPATH:$FFmpeg\lib" "/LIBPATH:$VulkanSdk\Lib" "/LIBPATH:$CudaSdk\lib\x64" avcodec.lib avformat.lib avutil.lib cuda.lib vulkan-1.lib
if ($LASTEXITCODE -ne 0) { throw 'NVENC bridge compilation failed' }
Get-ChildItem "$FFmpeg\bin\*.dll" | Copy-Item -Destination $output
Write-Host "Built $output\nvenc_bridge.dll"
