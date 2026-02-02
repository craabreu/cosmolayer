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

# Determine what ref to use (default to specific commit)
$gitRef = if ([string]::IsNullOrWhiteSpace($cosmosacRef)) { "21dd92b" } else { $cosmosacRef }

# Clone the repo
git clone --recurse-submodules $cosmosacRepo $cosmosacWorkdir
if ($LASTEXITCODE -ne 0) {
  throw "Failed to clone $cosmosacRepo"
}

# Reset to the desired ref (works for commits, branches, and tags)
git -C $cosmosacWorkdir reset --hard $gitRef
if ($LASTEXITCODE -ne 0) {
  throw "Failed to reset to $gitRef"
}

# Ensure submodules are synced to the checked-out ref
git -C $cosmosacWorkdir submodule update --init --recursive --depth 1
if ($LASTEXITCODE -ne 0) {
  git -C $cosmosacWorkdir submodule update --init --recursive
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to initialize COSMOSAC submodules."
  }
}

# Apply patch to increase iteration count (post-clone, pre-build)
$patchFile = Join-Path $PSScriptRoot "cosmosac.patch"
if (Test-Path $patchFile) {
  Write-Host "Applying patch from $patchFile..."
  git -C $cosmosacWorkdir apply $patchFile
  if ($LASTEXITCODE -eq 0) {
    Write-Host "Patch applied successfully"
  } else {
    Write-Warning "Failed to apply patch (may already be applied or incompatible)"
  }
} else {
  Write-Warning "Patch file not found at $patchFile; skipping patch"
}

# Check if COSMOSAC directory contains Python package metadata
$installDir = $null

# Check root directory first
if (Test-Path (Join-Path $cosmosacWorkdir "pyproject.toml")) {
  $installDir = $cosmosacWorkdir
  Write-Host "Found pyproject.toml at: $installDir"
} elseif (Test-Path (Join-Path $cosmosacWorkdir "setup.py")) {
  $installDir = $cosmosacWorkdir
  Write-Host "Found setup.py at: $installDir"
} else {
  # Search subdirectories for Python package metadata
  Write-Host "Searching for Python package metadata in subdirectories..."
  $pyprojects = Get-ChildItem -Path $cosmosacWorkdir -Recurse -Filter "pyproject.toml" -ErrorAction SilentlyContinue | Select-Object -First 1
  $setups = Get-ChildItem -Path $cosmosacWorkdir -Recurse -Filter "setup.py" -ErrorAction SilentlyContinue | Select-Object -First 1
  
  if ($pyprojects) {
    $installDir = $pyprojects.DirectoryName
    Write-Host "Found pyproject.toml at: $installDir"
  } elseif ($setups) {
    $installDir = $setups.DirectoryName
    Write-Host "Found setup.py at: $installDir"
  }
}

if ([string]::IsNullOrWhiteSpace($installDir)) {
  Write-Error "No Python packaging metadata found in $cosmosacWorkdir."
  exit 1
}

Write-Host "Installing COSMOSAC package from: $installDir"

if (Get-Command micromamba -ErrorAction SilentlyContinue) {
  micromamba run -n test python -m pip install $installDir --no-deps
} else {
  python -m pip install $installDir --no-deps
}
