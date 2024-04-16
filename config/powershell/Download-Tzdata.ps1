#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Download tzdata

.DESCRIPTION
    This script downloads the latest tzdata and places it in $Path

.EXAMPLE
    Download-Tzdata.ps1
#>

# Source helper functions:
. $PSScriptRoot/Helper-Functions.ps1

# Stop if errors:
$ErrorActionPreference="Stop"

# Build all:
function Download-Tzdata {

    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$Path = "$HOME/.howso"
    )

    $OS = Get-OS
    Write-Host "OS: $OS"
    
    $TzDataPath = "$Path/tzdata"
    if(-not (Test-Path "$TzDataPath")) {
        New-Item -ItemType Directory -Force -Path "$TzDataPath" | Out-Null
        $LocalTarGz = "$Path/tzdata.tar.gz"
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri "https://data.iana.org/time-zones/releases/tzdata2024a.tar.gz" -Outfile "$LocalTarGz"
        $ProgressPreference = 'Continue'
        Invoke-NativeCommand -Cmd "tar" -Arguments @("-xzf", "$LocalTarGz", "-C", "$TzDataPath")
        Remove-Item -Path "$LocalTarGz"
        Write-Host "tzdata written to: $TzDataPath"

        if ($OS.equals("windows")) {
            $TimeZoneFile = "$TzDataPath/windowsZones.xml"
            Invoke-WebRequest -Uri "https://raw.githubusercontent.com/unicode-org/cldr/main/common/supplemental/windowsZones.xml" -Outfile "$TimeZoneFile"
            Write-Host "Windows time zones written to: $TimeZoneFile"
        }
    } else {
        Write-Host "tzdata already exists, nothing to do"
    }
}

Download-Tzdata @args

exit 0