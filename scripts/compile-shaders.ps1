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
    @{ Source = "shaders\idwt53_offset_1d.comp.glsl"; Output = "idwt53_offset_1d.comp.spv" },
    @{ Source = "shaders\idwt53_fused_2d.comp.glsl"; Output = "idwt53_fused_2d.comp.spv" },
    @{ Source = "shaders\idwt97_1d.comp.glsl"; Output = "idwt97_1d.comp.spv" },
    @{ Source = "shaders\color_convert.comp.glsl"; Output = "color_convert.comp.spv" },
    @{ Source = "shaders\color_convert_i32.comp.glsl"; Output = "color_convert_i32.comp.spv" },
    @{ Source = "shaders\tier1_plan_smoke.comp.glsl"; Output = "tier1_plan_smoke.comp.spv" },
    @{ Source = "shaders\tier1_mq_smoke.comp.glsl"; Output = "tier1_mq_smoke.comp.spv" },
    @{ Source = "shaders\tier1_cleanup_smoke.comp.glsl"; Output = "tier1_cleanup_smoke.comp.spv" },
    @{ Source = "shaders\tier1_cleanup_smoke.comp.glsl"; Output = "tier1_cleanup_nostats.comp.spv"; Defines = @("-DTIER1_NO_STATS") },
    @{ Source = "shaders\tier1_thread_per_codeblock.comp.glsl"; Output = "tier1_cleanup_smoke_16.comp.spv" },
    @{ Source = "shaders\tier1_thread_per_codeblock.comp.glsl"; Output = "tier1_thread_per_codeblock_nostats.comp.spv"; Defines = @("-DTIER1_NO_STATS") },
    @{ Source = "shaders\tier1_packed.comp.glsl"; Output = "tier1_packed.comp.spv" },
    @{ Source = "shaders\tier1_scatter_smoke.comp.glsl"; Output = "tier1_scatter_smoke.comp.spv" }
)

foreach ($shader in $shaders) {
    $output = Join-Path $OutputDir $shader.Output
    $extraArgs = @()
    if ($shader.Source -match "tier1_cleanup_smoke|tier1_thread_per_codeblock|tier1_packed") {
        $extraArgs = @("--target-env=vulkan1.1")
    }
    if ($shader.Defines) {
        $extraArgs += $shader.Defines
    }
    & $glslc -fshader-stage=compute @extraArgs $shader.Source -o $output
    & $spirvVal $output
    Write-Host "compiled $($shader.Source) -> $output"
}
