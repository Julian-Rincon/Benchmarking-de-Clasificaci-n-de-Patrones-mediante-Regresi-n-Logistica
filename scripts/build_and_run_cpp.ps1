param(
    [string]$Dataset = "data\mnist_binary_3_8.csv"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath $Dataset)) {
    Write-Host "[INFO] Dataset no encontrado. Generando CSV..."
    python scripts\export_csv.py
}

$gpp = (Get-Command g++ -ErrorAction SilentlyContinue).Source
$make = (Get-Command mingw32-make -ErrorAction SilentlyContinue).Source

if (-not $gpp -or -not $make) {
    $wingetPackages = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    $winlibs = Get-ChildItem -Directory -Force $wingetPackages -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "BrechtSanders.WinLibs*" } |
        Select-Object -First 1

    if ($winlibs) {
        $binDir = Join-Path $winlibs.FullName "mingw64\bin"
        $gpp = Join-Path $binDir "g++.exe"
        $make = Join-Path $binDir "mingw32-make.exe"
    }
}

if (-not (Test-Path -LiteralPath $gpp) -or -not (Test-Path -LiteralPath $make)) {
    throw "No se encontro MinGW. Instala WinLibs con: winget install --id BrechtSanders.WinLibs.POSIX.UCRT --scope user"
}

cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_CXX_COMPILER="$gpp" -DCMAKE_MAKE_PROGRAM="$make"
cmake --build build --config Release -j 4

& ".\build\logistic_regression_eigen.exe" $Dataset
