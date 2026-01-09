# Ativação automática do venv para PowerShell
if (Test-Path -Path ".venv/Scripts/Activate.ps1") {
    & ".venv/Scripts/Activate.ps1"
}
