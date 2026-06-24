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
$allowCpu = Get-EnvOrDefault "FUNYI_ALLOW_CPU" "0"
$fireRedVadModelDir = Get-EnvOrDefault "FUNYI_FIRERED_VAD_MODEL_DIR" "third_party/firered-stream-vad-onnx"

function Resolve-RepoPath {
    Param([Parameter(Mandatory = $true)][string] $Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $repoRoot $Path)
}

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
    $port,
    "--firered-vad-model-dir",
    (Resolve-RepoPath $fireRedVadModelDir)
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

switch ($allowCpu.ToLowerInvariant()) {
    { $_ -in @("1", "true", "yes", "on") } {
        $uvArgs += "--allow-cpu"
        if ($translationModel -ne "") {
            [Console]::Error.WriteLine("FUNYI_ALLOW_CPU set: CPU mode is slow and not realtime; HY-MT (1.8B) is heavy on CPU. Consider FUNYI_TRANSLATION_MODEL= and a smaller ASR model such as Qwen/Qwen3-ASR-0.6B.")
        }
    }
}

if ($RemainingArgs) {
    $uvArgs += $RemainingArgs
}

& uv @uvArgs
exit $LASTEXITCODE
