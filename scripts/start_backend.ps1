Param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $RemainingArgs
)

$ErrorActionPreference = "Stop"

function Get-EnvOrDefault {
    Param(
        [Parameter(Mandatory = $true)]
        [string] $Name,
        [Parameter(Mandatory = $true)]
        [string] $Default
    )

    if (Test-Path -LiteralPath "Env:$Name") {
        return (Get-Item -LiteralPath "Env:$Name").Value
    }
    return $Default
}

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
Set-Location -LiteralPath $repoRoot

$asrModel = Get-EnvOrDefault "FUNYI_ASR_MODEL" "Qwen/Qwen3-ASR-1.7B"
$hostName = Get-EnvOrDefault "FUNYI_HOST" "127.0.0.1"
$port = Get-EnvOrDefault "FUNYI_PORT" "8000"
$translationModel = Get-EnvOrDefault "FUNYI_TRANSLATION_MODEL" "tencent/Hy-MT2-1.8B"
$timestampModel = Get-EnvOrDefault "FUNYI_TIMESTAMP_MODEL" "Qwen/Qwen3-ForcedAligner-0.6B"
$allowDownloads = Get-EnvOrDefault "FUNYI_ALLOW_DOWNLOADS" "0"

$uvArgs = @(
    "run",
    "--frozen",
    "python",
    "realtime_server.py",
    "--model",
    $asrModel,
    "--host",
    $hostName,
    "--port",
    $port
)

if ($translationModel -ne "") {
    $uvArgs += @("--translation-model", $translationModel)
}

if ($timestampModel -eq "") {
    [Console]::Error.WriteLine("FUNYI_TIMESTAMP_MODEL is required for realtime ASR.")
    exit 64
}
$uvArgs += @("--timestamp-model", $timestampModel)

switch ($allowDownloads.ToLowerInvariant()) {
    { $_ -in @("1", "true", "yes", "on") } {
        if ($translationModel -ne "") {
            $uvArgs += "--no-translation-local-files-only"
        }
        $uvArgs += "--no-timestamp-local-files-only"
    }
}

if ($RemainingArgs) {
    $uvArgs += $RemainingArgs
}

& uv @uvArgs
exit $LASTEXITCODE
