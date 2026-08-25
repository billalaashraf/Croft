<#
    bootstrap_install.ps1 — Windows / WSL2 bootstrap for Local LLM Chat.

    Windows support for this project means WSL2, deliberately. The inference
    stacks it installs (vLLM, TGI, the CUDA and Metal builds of llama.cpp) have
    no native-Windows support, and CUDA reaches WSL through the host NVIDIA
    driver anyway. So this script installs nothing on Windows itself:

      * verifies WSL is present and that the default distro is version 2,
      * offers to install WSL2 + Ubuntu when it is not,
      * hands off to bootstrap_install.sh inside WSL, against the project
        directory as WSL sees it,
      * smoke-tests the result, so a silent failure inside WSL is not reported
        back here as success.

    Usage (elevated PowerShell):
      ./bootstrap_install.ps1 -Mode native -Yes
#>
param(
    [ValidateSet("docker", "native")] [string]$Mode = "native",
    [switch]$Yes,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Log  ($m) { Write-Host "[bootstrap] $m" -ForegroundColor Cyan }
function Warn ($m) { Write-Host "[warn] $m"      -ForegroundColor Yellow }
function Die  ($m) { Write-Host "[error] $m"     -ForegroundColor Red; exit 1 }

function Confirm-Step ($prompt) {
    if ($Yes) { return $true }
    return ((Read-Host "$prompt [y/N]") -match '^(y|Y)')
}

# wsl.exe writes UTF-16LE down the pipe. PowerShell decodes it as ANSI, so every
# character arrives followed by a NUL: "Ubuntu" becomes "U`0b`0u`0n`0t`0u`0",
# which is not empty, does not equal "Ubuntu", and matches no pattern you meant
# to write — the distro check silently found nothing. WSL_UTF8 fixes it at the
# source on current builds; stripping NULs covers the older ones.
function Invoke-Wsl {
    param([Parameter(ValueFromRemainingArguments = $true)] [string[]]$WslArgs)
    $prev = $env:WSL_UTF8
    $env:WSL_UTF8 = "1"
    try     { return (& wsl.exe @WslArgs 2>&1 | Out-String) -replace "`0", "" }
    finally { $env:WSL_UTF8 = $prev }
}

# ---- WSL present? ----------------------------------------------------------
Log "Checking for WSL2..."
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    Warn "WSL is not installed."
    if (Confirm-Step "Install WSL2 + Ubuntu now? (requires a reboot)") {
        if (-not $DryRun) { wsl.exe --install -d Ubuntu }
        Die "WSL install started. Reboot, finish Ubuntu's first-run setup, then re-run this script."
    }
    Die "WSL2 is required on Windows. Aborting."
}

# ---- a distro, and specifically a version 2 one ----------------------------
# `wsl -l -q` lists names only. The VERSION column from `-v` is the part that
# matters: a WSL1 distro runs the bash script happily and then fails at the
# first CUDA call, a long way from the cause.
$verbose = Invoke-Wsl -l -v
$distroLines = $verbose -split "`r?`n" |
    Where-Object { $_ -match '\S' } |
    Select-Object -Skip 1                      # drop the header row

if (-not $distroLines) {
    Warn "No WSL distro found."
    if (Confirm-Step "Install Ubuntu under WSL now?") {
        if (-not $DryRun) { wsl.exe --install -d Ubuntu }
        Die "Ubuntu installed. Complete its first-run setup, then re-run this script."
    }
    Die "A WSL distro is required. Aborting."
}

# The default distro carries a '*', and is the one a bare `wsl <cmd>` uses.
$defaultLine = $distroLines |
    Where-Object { $_.TrimStart().StartsWith('*') } |
    Select-Object -First 1
if (-not $defaultLine) { $defaultLine = $distroLines[0] }

$fields = ($defaultLine -replace '^\s*\*?\s*', '') -split '\s+' |
    Where-Object { $_ -ne '' }
$distroName    = $fields[0]
$distroVersion = $fields[-1]
Log "Default WSL distro: $distroName (WSL version $distroVersion)"

if ($distroVersion -ne "2") {
    Warn "'$distroName' is running on WSL $distroVersion, not WSL 2."
    Warn "GPU passthrough and the CUDA stacks this installs both need WSL 2."
    if (Confirm-Step "Convert '$distroName' to WSL 2 now? (can take several minutes)") {
        if (-not $DryRun) { wsl.exe --set-version $distroName 2 }
    } else {
        Die "WSL 2 is required. Convert it with: wsl --set-version $distroName 2"
    }
}

# ---- GPU note --------------------------------------------------------------
if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
    Log "NVIDIA driver detected on host — CUDA passthrough to WSL2 available."
} else {
    Warn "No NVIDIA driver on host; WSL will run CPU-only."
    Warn "For GPU, install the NVIDIA driver for WSL on Windows — not inside WSL."
}

# ---- hand off to the Linux bootstrap ---------------------------------------
$projectWinPath = $PSScriptRoot
$wslPath = (Invoke-Wsl wslpath -a "$projectWinPath").Trim()
if (-not $wslPath) { Die "Could not translate '$projectWinPath' to a WSL path." }
Log "Project (WSL path): $wslPath"

# This path is interpolated into a bash command line, and "C:\Users\Sam's PC\"
# is a perfectly legal Windows path. Escape for a single-quoted bash string the
# way bash requires: close it, add an escaped quote, reopen. Unescaped, a space
# splits the argument and an apostrophe turns the rest of the line into garbage.
function ConvertTo-BashSingleQuoted ($s) { "'" + ($s -replace "'", "'\''") + "'" }

$flags = ""
if ($Yes)    { $flags += " --yes" }
if ($DryRun) { $flags += " --dry-run" }

$quotedPath = ConvertTo-BashSingleQuoted $wslPath
$cmd = "cd $quotedPath && chmod +x bootstrap_install.sh && ./bootstrap_install.sh --mode $Mode$flags"
Log "Handing off to WSL: $cmd"

if ($DryRun) {
    Log "[dry-run] wsl bash -lic `"$cmd`""
    Log "Windows bootstrap complete (dry run)."
    exit 0
}

wsl.exe bash -lic "$cmd"
$bootstrapExit = $LASTEXITCODE
if ($bootstrapExit -ne 0) {
    Die "The WSL bootstrap exited with code $bootstrapExit. See its output above."
}

# ---- smoke test ------------------------------------------------------------
# Announcing "complete" without checking has been wrong before: the handoff can
# return 0 while the app inside WSL never came up.
Log "Verifying the install inside WSL..."
$status = Invoke-Wsl bash -lic "cd $quotedPath && ./webui.sh status"
Write-Host $status
if ($status -match 'chat app is live') {
    Log "✓ Chat app is live inside WSL."
    Log "  Open the http://127.0.0.1:8090/#t=... link printed above. Windows"
    Log "  forwards localhost into WSL2, so it works from your browser as-is."
} else {
    Warn "The app is not answering yet. Start it with:"
    Warn "  wsl bash -lic ""cd $quotedPath && ./webui.sh start"""
}

Log "Windows bootstrap complete."
