<#
.SYNOPSIS
    Helper functions

.DESCRIPTION
    Helper functions for OS and native command handling
#>

# Stop if errors:
$ErrorActionPreference="Stop"

# Get the OS:
function Get-OS {
    $OSCheck = [Environment]::OSVersion.VersionString
    $OS = "unknown"
    if ($IsLinux) {
        $OS = "linux"
    }
    elseif ($IsMacOS) {
        $OS = "macos"
    }
    elseif ($OSCheck.Contains("Windows")) {
        $OS = "windows"
    }

    return $OS
}

# Run a native command:
function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory=$true)]
        [string] $Cmd,
        [Parameter(Mandatory=$true)]
        [string[]] $Arguments
    )

    $StartTime = $(get-date)
    Write-Host "Executing: '$Cmd $Arguments'"
    & $Cmd @Arguments
    $ExitCode = $LastExitCode
    $ElapsedTime = $(get-date) - $StartTime
    Write-Host "Wall clock runtime: $ElapsedTime"

    if ($ExitCode -ne 0) {
        Write-Error "Exit code $LastExitCode while running $Cmd $Arguments"
    }
}
