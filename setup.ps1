# Sets up Odycentric: a private Python environment with its libraries, and a desktop shortcut.
# Run from this folder:  powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Odycentric needs Python 3.10 or newer. Install it from python.org or the Microsoft Store, then run this again."
}
if (-not (Test-Path "$root\.venv\Scripts\python.exe")) {
    Write-Host "Creating the Python environment..."
    python -m venv "$root\.venv"
}
Write-Host "Installing libraries (CPU only)..."
& "$root\.venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
& "$root\.venv\Scripts\python.exe" -m pip install -r "$root\requirements.txt" --quiet

$desktop = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut("$desktop\Odycentric.lnk")
$link.TargetPath = "$root\.venv\Scripts\pythonw.exe"
$link.Arguments = "-m odycentric"
$link.WorkingDirectory = $root
$link.IconLocation = "$root\assets\odycentric.ico"
$link.Description = "Remove backgrounds from photos"
$link.Save()

Write-Host "Done. Odycentric is on your desktop. Each AI model downloads the first time you use it."
