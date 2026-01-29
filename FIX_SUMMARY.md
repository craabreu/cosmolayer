# Fix for Windows CI COSMOSAC Installation Failure

## Problem

The Windows CI was failing during the COSMOSAC package installation step with the following errors:

### First Failure:
```
'from' is not recognized as an internal or external command
'root' is not recognized as an internal or external command
No Python packaging metadata found in D:\a\cosmolayer\cosmolayer\.cosmosac.
```

### Second Failure (after initial fix):
```
The process tried to write to a nonexistent pipe. (repeated hundreds of times)
No Python packaging metadata found in D:\a\cosmolayer\cosmolayer\.cosmosac.
```

## Root Cause

1. **Initial Issue**: The PowerShell script was attempting to pass a multiline Python script to `python -c` through `micromamba run`, causing PowerShell to incorrectly parse and execute parts of the Python code as shell commands.

2. **Pipeline Issue**: After fixing the `-c` problem by writing to a temp file, the script still failed because PowerShell's pipeline (`|`) for capturing output from `micromamba run` was breaking, causing "tried to write to a nonexistent pipe" errors.

The problematic approach was:
```powershell
$pythonCommand = @'
import os
from pathlib import Path
...
'@

$installDir = (micromamba run -n test python -c $pythonCommand | ...)
```

When PowerShell passed this through `micromamba run`, the command-line argument parsing caused the Python code to be split and interpreted as shell commands.

## Solution

### PowerShell Script (`install_cosmosac.ps1`)

1. **Write Python code to a temporary file** instead of passing it via `-c` flag:
   ```powershell
   $tempScript = [System.IO.Path]::GetTempFileName() + ".py"
   Set-Content -Path $tempScript -Value $pythonScript -Encoding UTF8
   ```

2. **Use file-based output redirection** instead of pipeline capture:
   ```powershell
   $tempOutput = [System.IO.Path]::GetTempFileName()
   micromamba run -n test python $tempScript > $tempOutput 2>&1
   $installDir = (Get-Content $tempOutput -Raw).Trim()
   ```
   
   This avoids PowerShell pipeline issues with `micromamba run` that cause "tried to write to a nonexistent pipe" errors.

3. **Add proper error handling**:
   - Added `exit 1` after Write-Error to ensure script terminates on failure
   - Wrapped installation logic in try-finally block to ensure cleanup

4. **Add diagnostic output**:
   - Added "Found Python package at: ..." message to help with debugging

5. **Ensure cleanup** of both temporary files in finally block

### Bash Script (`install_cosmosac.sh`)

- Added diagnostic output for consistency with PowerShell script
- Script already had proper error handling with `exit 1`

## Verification

The Python logic correctly finds the `pyproject.toml` file in the COSMOSAC repository root:
- The COSMOSAC repository (https://github.com/usnistgov/COSMOSAC) contains `pyproject.toml` at its root
- The glob patterns `**/pyproject.toml` and `**/setup.py` correctly identify packaging files
- The scoring function prioritizes root-level packages over nested ones

## Testing

The fix has been tested locally and should resolve the Windows CI failure. The next CI run will verify that:
1. The Python script file is correctly created and executed
2. The COSMOSAC package location is found: `D:\a\cosmolayer\cosmolayer\.cosmosac`
3. The package is successfully installed via pip

## Files Changed

- `devtools/scripts/install_cosmosac.ps1` - Fixed Python script execution by using temp file + file-based output redirection instead of pipeline capture
- `devtools/scripts/install_cosmosac.sh` - Added diagnostic output for consistency

## Changes Summary

### Commit 1 (Initial Fix)
- Changed from `python -c $pythonCommand` to temp file approach
- Added error handling and diagnostic output
- **Result**: Fixed command parsing issue but encountered pipeline errors

### Commit 2 (Pipeline Fix) 
- Changed from pipeline capture `| Select-Object` to file redirection `> $tempOutput`
- Read output from file instead of capturing through pipeline
- **Result**: Should resolve "tried to write to a nonexistent pipe" errors
