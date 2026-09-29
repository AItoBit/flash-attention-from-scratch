param(
  [string]$BuildDir = "build"
)
$Root = Split-Path -Parent $PSScriptRoot
if (-not $PSScriptRoot) { $Root = Get-Location }
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
cmake -S $Root -B (Join-Path $Root $BuildDir) -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build (Join-Path $Root $BuildDir) --config Release --parallel
