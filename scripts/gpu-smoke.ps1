param(
    [string]$VulkanSdk = $(if ($env:VULKAN_SDK) { $env:VULKAN_SDK } else { "D:\VulkanSDK" })
)

$ErrorActionPreference = "Stop"

& powershell -ExecutionPolicy Bypass -File scripts\compile-shaders.ps1 -VulkanSdk $VulkanSdk
& cargo run -- vk-info
& cargo run -- vk-smoke
& cargo run -- vk-color-smoke
& cargo run -- vk-idwt53-smoke
& cargo run -- vk-idwt97-smoke
& cargo run -- vk-irreversible-smoke
