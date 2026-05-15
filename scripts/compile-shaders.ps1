param(
    [string]$VulkanSdk = $(if ($env:VULKAN_SDK) { $env:VULKAN_SDK } else { "D:\VulkanSDK" }),
    [string]$OutputDir = "target\shaders"
)

$ErrorActionPreference = "Stop"

$glslc = Join-Path $VulkanSdk "Bin\glslc.exe"
$spirvVal = Join-Path $VulkanSdk "Bin\spirv-val.exe"

if (!(Test-Path -LiteralPath $glslc)) {
    throw "glslc.exe not found at $glslc"
}
if (!(Test-Path -LiteralPath $spirvVal)) {
    throw "spirv-val.exe not found at $spirvVal"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$shaders = @(
    @{ Source = "shaders\dequant.comp.glsl"; Output = "dequant.comp.spv" },
    @{ Source = "shaders\idwt53_1d.comp.glsl"; Output = "idwt53_1d.comp.spv" },
    @{ Source = "shaders\idwt97_1d.comp.glsl"; Output = "idwt97_1d.comp.spv" },
    @{ Source = "shaders\color_convert.comp.glsl"; Output = "color_convert.comp.spv" }
)

foreach ($shader in $shaders) {
    $output = Join-Path $OutputDir $shader.Output
    & $glslc -fshader-stage=compute $shader.Source -o $output
    & $spirvVal $output
    Write-Host "compiled $($shader.Source) -> $output"
}
