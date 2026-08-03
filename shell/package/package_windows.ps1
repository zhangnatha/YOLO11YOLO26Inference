[CmdletBinding()]
param(
    [string]$MingwRoot = $env:MINGW_ROOT,
    [switch]$BuildOnly,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$ThirdPartyRoot = Join-Path $ProjectRoot "3rdparty/windows"
$VcpkgRoot = Join-Path $ThirdPartyRoot "vcpkg"
$OrtRoot = Join-Path $ThirdPartyRoot "onnxruntime-win-x64-1.17.3"
$BuildDir = Join-Path $ProjectRoot "build/windows-mingw64"
$DistRoot = Join-Path $ProjectRoot "dist"
$PackageName = "YOLO11YOLO26Inference-windows-x64"
$PackageDir = Join-Path $DistRoot $PackageName
$VcpkgCommit = "9d7f79f56ae1a9b4704d6a7fb8237e347a974133"
$OrtUrl = "https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-win-x64-1.17.3.zip"
$OrtSha256 = "356A33D024F2709786BEBD5D4CA06CD5392875DA95DAA0455AAE72EDC8993256"

function Assert-NativeSuccess([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

function Resolve-MingwTool([string]$Name) {
    if ($MingwRoot) {
        $Candidate = Join-Path (Join-Path $MingwRoot "bin") $Name
        if (Test-Path $Candidate) { return (Resolve-Path $Candidate).Path }
    }
    $Command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($Command) { return $Command.Source }
    throw "MinGW tool not found: $Name. Set MINGW_ROOT to mingw64_x86_64-15.2.0."
}

$Gcc = Resolve-MingwTool "gcc.exe"
$Gxx = Resolve-MingwTool "g++.exe"
$Make = Resolve-MingwTool "mingw32-make.exe"
$Objdump = Resolve-MingwTool "objdump.exe"
$Dlltool = Resolve-MingwTool "dlltool.exe"
$MingwBin = Split-Path $Gxx -Parent
$env:PATH = "$MingwBin;$env:PATH"

$CompilerVersion = (& $Gxx -dumpfullversion).Trim()
Assert-NativeSuccess "MinGW version check"
if ($CompilerVersion -ne "15.2.0") {
    throw "Unsupported MinGW GCC version $CompilerVersion; exactly 15.2.0 is required."
}
$CompilerTarget = (& $Gxx -dumpmachine).Trim()
Assert-NativeSuccess "MinGW target check"
if ($CompilerTarget -notmatch "^x86_64-w64-mingw32") {
    throw "Unsupported compiler target $CompilerTarget; x86_64-w64-mingw32 is required."
}
Write-Host "Using MinGW-w64 $CompilerTarget GCC $CompilerVersion"

function Get-PeDependencies([string]$Binary) {
    $Output = & $Objdump -p $Binary 2>$null
    Assert-NativeSuccess "Inspecting $Binary"
    @($Output | ForEach-Object {
        if ($_ -match "DLL Name:\s*(\S+)") { $Matches[1] }
    })
}

function Copy-RuntimeDependencies([string]$Destination) {
    $System32 = Join-Path $env:WINDIR "System32"
    $SearchDirectories = @($Destination, $MingwBin, (Join-Path $OrtRoot "lib"), (Join-Path $OrtRoot "bin"))
    $Queue = New-Object System.Collections.Generic.Queue[string]
    $Visited = @{}
    Get-ChildItem $Destination -File | Where-Object { $_.Extension -in ".exe", ".dll" } |
        ForEach-Object { $Queue.Enqueue($_.FullName) }

    while ($Queue.Count -gt 0) {
        $Current = $Queue.Dequeue()
        foreach ($Dependency in (Get-PeDependencies $Current)) {
            $Key = $Dependency.ToLowerInvariant()
            if ($Visited.ContainsKey($Key)) { continue }
            $Visited[$Key] = $true

            $Existing = Join-Path $Destination $Dependency
            if (Test-Path $Existing) {
                $Queue.Enqueue((Resolve-Path $Existing).Path)
                continue
            }

            $Resolved = $null
            foreach ($Directory in $SearchDirectories) {
                $Candidate = Join-Path $Directory $Dependency
                if (Test-Path $Candidate) {
                    $Resolved = (Resolve-Path $Candidate).Path
                    break
                }
            }
            if ($Resolved) {
                Copy-Item $Resolved $Destination
                $Queue.Enqueue((Join-Path $Destination $Dependency))
                continue
            }

            $SystemCandidate = Join-Path $System32 $Dependency
            if ($Dependency -match "^(msvcp|vcruntime|concrt)[0-9_]*\.dll$" -and
                (Test-Path $SystemCandidate)) {
                Copy-Item $SystemCandidate $Destination
                $Queue.Enqueue((Join-Path $Destination $Dependency))
                continue
            }
            if ((Test-Path $SystemCandidate) -or
                $Dependency -match "^(api-ms-win-|ext-ms-win-)") {
                continue
            }
            throw "Runtime DLL not found: $Dependency (required by $Current)"
        }
    }
}

New-Item -ItemType Directory -Force -Path $ThirdPartyRoot, $DistRoot | Out-Null

if (-not (Test-Path (Join-Path $VcpkgRoot ".git"))) {
    if ($Offline) { throw "Offline mode: cached vcpkg checkout is missing: $VcpkgRoot" }
    git clone https://github.com/microsoft/vcpkg.git $VcpkgRoot
    Assert-NativeSuccess "vcpkg clone"
}
git -C $VcpkgRoot checkout --detach $VcpkgCommit
Assert-NativeSuccess "vcpkg checkout"

$VcpkgExe = Join-Path $VcpkgRoot "vcpkg.exe"
if (-not (Test-Path $VcpkgExe)) {
    if ($Offline) { throw "Offline mode: vcpkg.exe is missing." }
    & (Join-Path $VcpkgRoot "bootstrap-vcpkg.bat") -disableMetrics
    Assert-NativeSuccess "vcpkg bootstrap"
}

if (-not (Test-Path (Join-Path $OrtRoot "include/onnxruntime_cxx_api.h"))) {
    if ($Offline) { throw "Offline mode: cached ONNX Runtime 1.17.3 SDK is missing." }
    $OrtZip = Join-Path $ThirdPartyRoot "onnxruntime-win-x64-1.17.3.zip"
    Invoke-WebRequest -UseBasicParsing -Uri $OrtUrl -OutFile $OrtZip
    $ActualHash = (Get-FileHash -Algorithm SHA256 $OrtZip).Hash
    if ($ActualHash -ne $OrtSha256) {
        throw "ONNX Runtime archive checksum mismatch: $ActualHash"
    }
    Expand-Archive -Force -Path $OrtZip -DestinationPath $ThirdPartyRoot
}

# The official SDK ships an MSVC .lib. Generate the small GNU import library
# used by MinGW; ONNX Runtime's public C API enters through OrtGetApiBase.
$OrtLibDirectory = Join-Path $OrtRoot "lib"
$OrtDefinition = Join-Path $OrtLibDirectory "onnxruntime.def"
$OrtImportLibrary = Join-Path $OrtLibDirectory "libonnxruntime.dll.a"
@("LIBRARY onnxruntime.dll", "EXPORTS", "    OrtGetApiBase") |
    Set-Content -Encoding Ascii $OrtDefinition
& $Dlltool -d $OrtDefinition -D onnxruntime.dll -l $OrtImportLibrary
Assert-NativeSuccess "ONNX Runtime MinGW import library generation"

$env:VCPKG_ROOT = $VcpkgRoot
$env:VCPKG_FEATURE_FLAGS = "manifests"
$OfflineArg = @()
if ($Offline) { $OfflineArg = @("--no-downloads") }
Push-Location $ProjectRoot
try {
    & $VcpkgExe install --triplet x64-mingw-static @OfflineArg
    Assert-NativeSuccess "vcpkg dependency install"
} finally {
    Pop-Location
}

if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
$CmakeArguments = @(
    "-S", $ProjectRoot,
    "-B", $BuildDir,
    "-G", "MinGW Makefiles",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_C_COMPILER=$Gcc",
    "-DCMAKE_CXX_COMPILER=$Gxx",
    "-DCMAKE_MAKE_PROGRAM=$Make",
    "-DCMAKE_TOOLCHAIN_FILE=$VcpkgRoot/scripts/buildsystems/vcpkg.cmake",
    "-DVCPKG_TARGET_TRIPLET=x64-mingw-static",
    "-DONNXRUNTIME_DIR=$OrtRoot"
)
& cmake @CmakeArguments
Assert-NativeSuccess "CMake configure"
& cmake --build $BuildDir --parallel
Assert-NativeSuccess "CMake build"

$BuiltExecutable = Join-Path $BuildDir "YOLO_seg.exe"
if (-not (Test-Path $BuiltExecutable)) {
    throw "Built executable is missing: $BuiltExecutable"
}
Copy-RuntimeDependencies $BuildDir

if ($BuildOnly) {
    Write-Host "Windows source build complete: $BuiltExecutable"
    Write-Host "Run YOLO_seg.exe directly; no BAT file or Visual Studio is required."
    exit 0
}

if (Test-Path $PackageDir) { Remove-Item -Recurse -Force $PackageDir }
New-Item -ItemType Directory -Force -Path $PackageDir,
    (Join-Path $PackageDir "results"), (Join-Path $PackageDir "original") | Out-Null
Copy-Item $BuiltExecutable $PackageDir
Get-ChildItem $BuildDir -File -Filter "*.dll" | Copy-Item -Destination $PackageDir
Copy-Item -Recurse (Join-Path $ProjectRoot "model") (Join-Path $PackageDir "model")
Copy-Item -Recurse (Join-Path $ProjectRoot "images") (Join-Path $PackageDir "images")
Copy-Item (Join-Path $ProjectRoot "README.md") $PackageDir
Copy-RuntimeDependencies $PackageDir

$Archive = Join-Path $DistRoot "$PackageName.zip"
if (Test-Path $Archive) { Remove-Item -Force $Archive }
Compress-Archive -CompressionLevel Optimal -Path (Join-Path $PackageDir "*") -DestinationPath $Archive
Write-Host "Offline Windows package: $Archive"
Write-Host "Extract it and run YOLO_seg.exe directly. No installer or BAT file is required."
