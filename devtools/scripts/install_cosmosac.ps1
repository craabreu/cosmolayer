$ErrorActionPreference = "Stop"

$cosmosacRepo = $env:COSMOSAC_GIT_URL
if ([string]::IsNullOrWhiteSpace($cosmosacRepo)) {
  $cosmosacRepo = "https://github.com/usnistgov/COSMOSAC"
}

$cosmosacRef = $env:COSMOSAC_GIT_REF
$cosmosacWorkdir = $env:COSMOSAC_WORKDIR
if ([string]::IsNullOrWhiteSpace($cosmosacWorkdir)) {
  $cosmosacWorkdir = Join-Path (Get-Location) ".cosmosac"
}
$env:COSMOSAC_WORKDIR = $cosmosacWorkdir

if (Test-Path $cosmosacWorkdir) {
  Remove-Item -Recurse -Force $cosmosacWorkdir
}

git clone --depth 1 --recurse-submodules $cosmosacRepo $cosmosacWorkdir
if ($LASTEXITCODE -ne 0) {
  git clone --recurse-submodules $cosmosacRepo $cosmosacWorkdir
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to clone $cosmosacRepo"
  }
}

if (-not [string]::IsNullOrWhiteSpace($cosmosacRef)) {
  git -C $cosmosacWorkdir fetch --depth 1 origin $cosmosacRef
  git -C $cosmosacWorkdir checkout FETCH_HEAD
}

git -C $cosmosacWorkdir submodule update --init --recursive --depth 1
if ($LASTEXITCODE -ne 0) {
  git -C $cosmosacWorkdir submodule update --init --recursive
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to initialize COSMOSAC submodules."
  }
}

$pythonCommand = @'
import os
from pathlib import Path

root = Path(os.environ["COSMOSAC_WORKDIR"])
candidates = []
for path in [root] + list(root.glob("**/pyproject.toml")) + list(
    root.glob("**/setup.py")
):
    if path.is_file():
        candidates.append(path.parent)

def score(path: Path) -> tuple[int, int]:
    depth = len(path.relative_to(root).parts)
    is_root = 0 if path == root else 1
    return (is_root, depth)

if candidates:
    best = sorted(set(candidates), key=score)[0]
    print(best)
'@

if (Get-Command micromamba -ErrorAction SilentlyContinue) {
  $installDir = (micromamba run -n test python -c $pythonCommand | Select-Object -First 1).Trim()
} else {
  $installDir = (python -c $pythonCommand | Select-Object -First 1).Trim()
}

if ([string]::IsNullOrWhiteSpace($installDir)) {
  Write-Error "No Python packaging metadata found in $cosmosacWorkdir."
}

if (Get-Command micromamba -ErrorAction SilentlyContinue) {
  micromamba run -n test python -m pip install $installDir --no-deps
} else {
  python -m pip install $installDir --no-deps
}
