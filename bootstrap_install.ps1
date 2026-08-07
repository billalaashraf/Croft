<#
    bootstrap_install.ps1 — Windows / WSL2 bootstrap stub for Local LLM Chat.

    Windows GPU inference for this stack is only supported *inside WSL2*
    (CUDA passthrough via the WSL NVIDIA driver). This stub:
      * verifies WSL2 + a distro are present (offers to install them),
      * copies the project into the WSL filesystem,
      * hands off to bootstrap_install.sh inside WSL.

    Usage (elevated PowerShell):
      ./bootstrap_install.ps1 -Mode native -Yes
#>
param(
    [ValidateSet("docker", "native")] [string]$Mode = "native",
    [switch]$Yes,
    [switch]$DryRun
)

function Log  ($m) { Write-Host "[bootstrap] $m" -ForegroundColor Cyan }
function Warn ($m) { Write-Host "[warn] $m"      -ForegroundColor Yellow }
function Die  ($m) { Write-Host "[error] $m"     -ForegroundColor Red; exit 1 }

Log "Checking for WSL2..."
$wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
if (-not $wsl) {
    Warn "WSL is not installed."
    if ($Yes -or (Read-Host "Install WSL2 + Ubuntu now? (requires reboot) [y/N]") -match '^(y|Y)') {
        if (-not $DryRun) { wsl --install -d Ubuntu }
        Die "WSL installed. Reboot, then re-run this script."
    } else { Die "WSL2 is required on Windows. Aborting." }
}

# Ensure a default distro exists
$distros = (wsl -l -q) -join "`n"
if ([string]::IsNullOrWhiteSpace($distros)) {
    Warn "No WSL distro found."
    if (-not $DryRun) { wsl --install -d Ubuntu }
    Die "Ubuntu installed under WSL. Complete its first-run setup, then re-run."
}
Log "WSL distro(s): $distros"

# GPU note
$nvsmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($nvsmi) {
    Log "NVIDIA driver detected on host — CUDA passthrough to WSL2 available."
} else {
    Warn "No NVIDIA driver on host; WSL will run CPU-only. Install the NVIDIA WSL driver for GPU."
}

# Copy project into WSL home and hand off
$projectWinPath = $PSScriptRoot
$wslPath = (wsl wslpath -a "$projectWinPath").Trim()
Log "Project (WSL path): $wslPath"

$flags = ""
if ($Yes)    { $flags += " --yes" }
if ($DryRun) { $flags += " --dry-run" }

$cmd = "cd '$wslPath' && chmod +x bootstrap_install.sh && ./bootstrap_install.sh --mode $Mode$flags"
Log "Handing off to WSL: $cmd"
if (-not $DryRun) {
    wsl bash -lic "$cmd"
} else {
    Log "[dry-run] wsl bash -lic `"$cmd`""
}
Log "Windows bootstrap complete."
