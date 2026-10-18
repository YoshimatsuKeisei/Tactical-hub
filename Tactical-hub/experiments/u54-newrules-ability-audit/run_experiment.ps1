param([switch]$Execute)

$ErrorActionPreference = "Stop"
$ExperimentDir = $PSScriptRoot
$Metadata = Get-Content -LiteralPath (Join-Path $ExperimentDir "kernel-metadata.json") -Raw | ConvertFrom-Json
$ExpectedKernelSource = "keisabo/tactical-hub-u52-one-update-balance-trial"
$ExpectedMarker = "TTH_U54_NEWRULES_ABILITY_AUDIT_20261019"
$Runner = "C:\Users\jingc\Documents\Tactical hub\kaggle-50k-current-audit\run_kaggle_experiment.ps1"

if ([string]$Metadata.id -ne "keisabo/tactical-hub-u54-newrules-ability-audit") {
  throw "Unexpected target Kernel id."
}
if (@($Metadata.kernel_sources).Count -ne 1 -or @($Metadata.kernel_sources)[0] -ne $ExpectedKernelSource) {
  throw "Unexpected u53 source Kernel."
}
if (-not (Test-Path -LiteralPath (Join-Path $ExperimentDir "tth-u54-newrules-ability-audit.ipynb"))) {
  throw "Prepared Notebook is missing."
}
if (-not $Execute) {
  & $Runner -NotebookDir $ExperimentDir -KernelSlug $Metadata.id -ExpectedNotebookMarker $ExpectedMarker -TimeoutMinutes 720 -PreflightOnly
  exit $LASTEXITCODE
}

& $Runner -NotebookDir $ExperimentDir -KernelSlug $Metadata.id -ExpectedNotebookMarker $ExpectedMarker -TimeoutMinutes 720
exit $LASTEXITCODE
