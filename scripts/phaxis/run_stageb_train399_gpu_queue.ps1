<#
.SYNOPSIS
Fail-closed single-GPU queue for the five formal PHAxis Stage-B train399 seeds.

.DESCRIPTION
The script owns one physical GPU for the lifetime of a queue, exposes only that
GPU through CUDA_VISIBLE_DEVICES, and always invokes the trainer on internal
device cuda:0.  It never changes any training hyperparameter.  A checkpoint is
resumed when present; a new member is started only when its seed directory is
absent.  Completed members are hash-verified and skipped.

The odd-resume lane is intended for the interrupted seed 2026082801 followed by
2026082803 and 2026082805.  The even-fresh lane covers 2026082802 and
2026082804.  The pending-single-gpu lane contains only 2026082803, 2026082804
and 2026082805; it is the safe continuation after seeds 1 and 2 have already
completed on different physical GPUs.  All lanes are restartable after a
host/session interruption.

Examples
--------
# Inspect the immutable execution plan without touching a GPU.
pwsh -File scripts/phaxis/run_stageb_train399_gpu_queue.ps1 `
  -Lane odd-resume -PhysicalGpu 1 -PlanOnly

# Run the odd lane.  Launch this supervisor itself with Start-Process when it
# must outlive the calling terminal; every Python child is already hidden and
# writes directly to durable stdout/stderr files.
pwsh -File scripts/phaxis/run_stageb_train399_gpu_queue.ps1 `
  -Lane odd-resume -PhysicalGpu 1 -AttachExistingPid 44608

# Omit -AttachExistingPid when no matching first-seed process is already live;
# seed1 will then be resumed from last.pt by a new hidden child.

# Continue only the three unfinished members on one physical GPU.
pwsh -File scripts/phaxis/run_stageb_train399_gpu_queue.ps1 `
  -Lane pending-single-gpu -PhysicalGpu 1

# CPU-only verification of one completed member.
pwsh -File scripts/phaxis/run_stageb_train399_gpu_queue.ps1 `
  -VerifySeedDirectory models/phaxis_stageb_train399_v1_0_20260828/seed_2026082802 `
  -VerifySeed 2026082802 -ExpectedPhysicalGpu 0
#>

[CmdletBinding(DefaultParameterSetName = 'Run')]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'Run')]
    [Parameter(Mandatory = $true, ParameterSetName = 'Plan')]
    [ValidateSet('odd-resume', 'even-fresh', 'pending-single-gpu')]
    [string]$Lane,

    [Parameter(Mandatory = $true, ParameterSetName = 'Run')]
    [Parameter(Mandatory = $true, ParameterSetName = 'Plan')]
    [ValidateRange(0, 1)]
    [int]$PhysicalGpu,

    [Parameter(Mandatory = $true, ParameterSetName = 'Plan')]
    [switch]$PlanOnly,

    [Parameter(Mandatory = $true, ParameterSetName = 'Verify')]
    [ValidateNotNullOrEmpty()]
    [string]$VerifySeedDirectory,

    [Parameter(Mandatory = $true, ParameterSetName = 'Verify')]
    [ValidateSet(2026082801, 2026082802, 2026082803, 2026082804, 2026082805)]
    [int]$VerifySeed,

    [Parameter(Mandatory = $true, ParameterSetName = 'Verify')]
    [ValidateRange(0, 1)]
    [int]$ExpectedPhysicalGpu,

    [Parameter(ParameterSetName = 'Verify')]
    [Nullable[DateTime]]$VerifyReceiptNotBeforeUtc,

    [Parameter(ParameterSetName = 'Run')]
    [ValidateRange(1, 16)]
    [int]$GpuSamples = 3,

    [Parameter(ParameterSetName = 'Run')]
    [ValidateRange(0, 30)]
    [int]$GpuSampleIntervalSeconds = 2,

    [Parameter(ParameterSetName = 'Run')]
    [ValidateRange(1, 24576)]
    [int]$ExpectedTrainingPeakMiB = 11000,

    [Parameter(ParameterSetName = 'Run')]
    [ValidateRange(2048, 24576)]
    [int]$RequiredFreeMarginMiB = 2048,

    [Parameter(ParameterSetName = 'Run')]
    [ValidateRange(1, 100)]
    [int]$MaximumGpuUtilizationPercent = 79,

    [Parameter(ParameterSetName = 'Run')]
    [int]$AttachExistingPid = 0,

    [Parameter(Mandatory = $true, ParameterSetName = 'AuditAttach')]
    [ValidateNotNullOrEmpty()]
    [string]$AuditAttachFixture,

    [Parameter(Mandatory = $true, ParameterSetName = 'AuditAttach')]
    [ValidateSet(2026082801, 2026082802, 2026082803, 2026082804, 2026082805)]
    [int]$AuditAttachSeed,

    [Parameter(Mandatory = $true, ParameterSetName = 'AuditAttach')]
    [ValidateRange(0, 1)]
    [int]$AuditAttachPhysicalGpu,

    [Parameter(Mandatory = $true, ParameterSetName = 'AuditLiveAttach')]
    [ValidateRange(1, 2147483647)]
    [int]$AuditLiveAttachPid,

    [Parameter(Mandatory = $true, ParameterSetName = 'AuditLiveAttach')]
    [ValidateSet(2026082801, 2026082802, 2026082803, 2026082804, 2026082805)]
    [int]$AuditLiveAttachSeed,

    [Parameter(Mandatory = $true, ParameterSetName = 'AuditLiveAttach')]
    [ValidateRange(0, 1)]
    [int]$AuditLiveAttachPhysicalGpu
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw 'PHAxis train399 queue requires PowerShell 7 or newer.'
}

$script:ExpectedEpochs = 60
$script:ExpectedStepsPerEpoch = 399
$script:ExpectedGlobalSteps = 23940
$script:FormalSeeds = @(2026082801, 2026082802, 2026082803, 2026082804, 2026082805)
$script:ProjectRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot '..\..')
)
$script:TrainingScript = Join-Path $script:ProjectRoot 'scripts\phaxis\train_stageb_train399.py'
$script:PythonExecutable = Join-Path $script:ProjectRoot 'envs\rhpheno\python.exe'
$script:ModelRoot = Join-Path $script:ProjectRoot 'models\phaxis_stageb_train399_v1_0_20260828'
$script:QueueRoot = Join-Path $script:ProjectRoot 'outputs\phaxis_stageb_train399_queue_runs'
$script:LockRoot = Join-Path $script:QueueRoot 'locks'
$script:Utf8NoBom = [System.Text.UTF8Encoding]::new($false)

function Get-UtcTimestamp {
    return [DateTime]::UtcNow.ToString('o')
}

function ConvertTo-CanonicalPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [System.IO.Path]::GetFullPath($Path).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
}

function Write-AtomicJson {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Payload
    )
    $parent = Split-Path -Parent $Path
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    $temporary = "$Path.tmp.$PID.$([Guid]::NewGuid().ToString('N'))"
    try {
        $json = $Payload | ConvertTo-Json -Depth 64
        [System.IO.File]::WriteAllText($temporary, $json + [Environment]::NewLine, $script:Utf8NoBom)
        [System.IO.File]::Move($temporary, $Path, $true)
    }
    finally {
        if ([System.IO.File]::Exists($temporary)) {
            [System.IO.File]::Delete($temporary)
        }
    }
}

function Enter-ExclusiveFileLock {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Metadata
    )
    $parent = Split-Path -Parent $Path
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    try {
        $stream = [System.IO.File]::Open(
            $Path,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    }
    catch [System.IO.IOException] {
        throw "exclusive train399 lock is already held: $Path"
    }
    try {
        Write-ExclusiveLockMetadata -Stream $stream -Metadata $Metadata
        return $stream
    }
    catch {
        $stream.Dispose()
        throw
    }
}

function Write-ExclusiveLockMetadata {
    param(
        [Parameter(Mandatory = $true)][System.IO.FileStream]$Stream,
        [Parameter(Mandatory = $true)]$Metadata
    )
    $json = ($Metadata | ConvertTo-Json -Depth 32) + [Environment]::NewLine
    $bytes = $script:Utf8NoBom.GetBytes($json)
    $Stream.Position = 0
    $Stream.SetLength(0)
    $Stream.Write($bytes, 0, $bytes.Length)
    $Stream.Flush($true)
}

function Exit-ExclusiveFileLock {
    param(
        [Parameter(Mandatory = $true)][System.IO.FileStream]$Stream,
        [Parameter(Mandatory = $true)]$Metadata
    )
    try {
        Write-ExclusiveLockMetadata -Stream $Stream -Metadata $Metadata
    }
    finally {
        $Stream.Dispose()
    }
}

function Get-LanePlan {
    param([Parameter(Mandatory = $true)][string]$Name)
    switch ($Name) {
        'odd-resume' {
            return @(
                [ordered]@{ seed = 2026082801; initial_policy = 'resume-required-if-incomplete' },
                [ordered]@{ seed = 2026082803; initial_policy = 'fresh-or-resume-if-incomplete' },
                [ordered]@{ seed = 2026082805; initial_policy = 'fresh-or-resume-if-incomplete' }
            )
        }
        'even-fresh' {
            return @(
                [ordered]@{ seed = 2026082802; initial_policy = 'fresh-or-resume-if-incomplete' },
                [ordered]@{ seed = 2026082804; initial_policy = 'fresh-or-resume-if-incomplete' }
            )
        }
        'pending-single-gpu' {
            return @(
                [ordered]@{ seed = 2026082803; initial_policy = 'fresh-or-resume-if-incomplete' },
                [ordered]@{ seed = 2026082804; initial_policy = 'fresh-or-resume-if-incomplete' },
                [ordered]@{ seed = 2026082805; initial_policy = 'fresh-or-resume-if-incomplete' }
            )
        }
        default { throw "unsupported lane: $Name" }
    }
}

function Get-QueuePlanPayload {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][int]$Gpu
    )
    $members = foreach ($entry in (Get-LanePlan -Name $Name)) {
        [ordered]@{
            seed = [int]$entry.seed
            initial_policy = [string]$entry.initial_policy
            run_directory = (Join-Path $script:ModelRoot "seed_$($entry.seed)")
            internal_device = 'cuda:0'
        }
    }
    return [ordered]@{
        schema_version = 'PHAxis-StageB-train399-GPU-queue-plan-1.0'
        lane = $Name
        physical_gpu = $Gpu
        cuda_visible_devices = [string]$Gpu
        cuda_device_order = 'PCI_BUS_ID'
        python = $script:PythonExecutable
        training_script = $script:TrainingScript
        expected_epochs = $script:ExpectedEpochs
        expected_steps_per_epoch = $script:ExpectedStepsPerEpoch
        expected_global_steps = $script:ExpectedGlobalSteps
        child_process_policy = 'hidden-direct-durable-stdout-stderr'
        failure_policy = 'stop-entire-lane-on-first-failure'
        members = @($members)
    }
}

function Assert-CoreFiles {
    foreach ($path in @($script:PythonExecutable, $script:TrainingScript)) {
        if (-not [System.IO.File]::Exists($path)) {
            throw "required train399 file is missing: $path"
        }
    }
    $requiredInputs = @(
        (Join-Path $script:ProjectRoot 'outputs\phaxis_stageb_train399_dataset_audit_run1\dataset_audit.json'),
        (Join-Path $script:ProjectRoot 'outputs\phaxis_stageb_train399_cache_canonical_v1_0\cache_audit.json'),
        (Join-Path $script:ProjectRoot 'configs\rhaxis_nextgen\splits\qc_development_v1_0\split_manifest.csv')
    )
    foreach ($path in $requiredInputs) {
        if (-not [System.IO.File]::Exists($path)) {
            throw "required locked train399 input is missing: $path"
        }
    }
}

function Assert-NoDuplicateSeedProcess {
    param([Parameter(Mandatory = $true)][int]$Seed)
    $needle = [string]$Seed
    $matches = @(
        Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            $_.ProcessId -ne $PID -and
            $_.CommandLine -and
            $_.CommandLine -match 'train_stageb_train399\.py"?\s+train' -and
            $_.CommandLine -match "--seed\s+$needle(?:\s|$)"
        }
    )
    if ($matches.Count -gt 0) {
        $descriptions = $matches | ForEach-Object {
            "PID=$($_.ProcessId) parent=$($_.ParentProcessId) command=$($_.CommandLine)"
        }
        throw "seed $Seed already has a live training process: $($descriptions -join ' | ')"
    }
}

function Normalize-ExactCommandLine {
    param([Parameter(Mandatory = $true)][string]$CommandLine)
    $normalized = $CommandLine.Replace('/', '\')
    $normalized = [regex]::Replace($normalized.Trim(), '\s+', ' ')
    return $normalized.ToLowerInvariant()
}

function Confirm-AttachIdentityRecord {
    param(
        [Parameter(Mandatory = $true)]$ProcessRecord,
        [Parameter(Mandatory = $true)][string]$SeedDirectory,
        [Parameter(Mandatory = $true)][int]$Seed,
        [Parameter(Mandatory = $true)][int]$Gpu
    )
    $processId = [int]$ProcessRecord.process_id
    if ($processId -le 0) {
        throw 'attach process identity has an invalid PID'
    }
    $actualExecutable = ConvertTo-CanonicalPath -Path ([string]$ProcessRecord.executable_path)
    if ($actualExecutable -ne (ConvertTo-CanonicalPath -Path $script:PythonExecutable)) {
        throw "attach PID $processId executable is not the locked rhpheno Python"
    }
    $expectedRelative = (
        '"' + $script:PythonExecutable +
        '" scripts\phaxis\train_stageb_train399.py train --seed ' +
        $Seed + ' --device cuda:0 --resume'
    )
    $expectedAbsolute = (
        '"' + $script:PythonExecutable + '" "' + $script:TrainingScript +
        '" train --seed ' + $Seed + ' --device cuda:0 --resume'
    )
    $actualCommand = Normalize-ExactCommandLine -CommandLine ([string]$ProcessRecord.command_line)
    $allowedCommands = @(
        (Normalize-ExactCommandLine -CommandLine $expectedRelative),
        (Normalize-ExactCommandLine -CommandLine $expectedAbsolute)
    )
    if ($allowedCommands -notcontains $actualCommand) {
        throw (
            "attach PID $processId command is not the exact formal resume command; " +
            "actual=$($ProcessRecord.command_line)"
        )
    }

    try {
        $created = [DateTimeOffset]::Parse(
            [string]$ProcessRecord.creation_utc,
            [System.Globalization.CultureInfo]::InvariantCulture
        ).UtcDateTime
    }
    catch {
        throw "attach PID $processId has an invalid creation_utc"
    }
    $preflightFiles = @(
        Get-ChildItem -LiteralPath $SeedDirectory -File -Filter 'nvidia_smi_preflight*.json' |
            Sort-Object LastWriteTimeUtc
    )
    $bindingMatches = @()
    $rejections = @()
    foreach ($file in $preflightFiles) {
        if ($file.Name -eq 'nvidia_smi_preflight.json') {
            $suffix = ''
        }
        elseif ($file.Name -match '^nvidia_smi_preflight(?<suffix>_resume_[0-9]{3})\.json$') {
            $suffix = [string]$Matches.suffix
        }
        else {
            continue
        }
        try {
            $receipt = Read-JsonObject -Path $file.FullName
            $captured = [DateTimeOffset]::Parse(
                [string]$receipt.captured_utc,
                [System.Globalization.CultureInfo]::InvariantCulture
            ).UtcDateTime
            if ($captured -lt $created.AddSeconds(-2)) {
                continue
            }
            if ([string]$receipt.schema_version -ne 'PHAxis-nvidia-smi-preflight-1.0') {
                throw 'schema mismatch'
            }
            if ([string]$receipt.status -ne 'passed') {
                throw 'status is not passed'
            }
            if ([string]$receipt.cuda_visible_devices -ne [string]$Gpu) {
                throw "CUDA_VISIBLE_DEVICES is not physical GPU $Gpu"
            }
            if ([string]$receipt.internal_device -ne 'cuda:0') {
                throw 'internal device is not cuda:0'
            }
            if ([bool]$receipt.torch_cuda_initialized_before_preflight) {
                throw 'CUDA was initialized before trainer preflight'
            }
            if ([bool]$receipt.existing_processes_killed_or_suspended) {
                throw 'trainer preflight reports a killed or suspended process'
            }
            $bindingMatches += [ordered]@{
                path = $file.FullName
                sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                captured_utc = $captured.ToString('o')
                artifact_suffix = $suffix
                expected_completion_receipt = (Join-Path $SeedDirectory "training_receipt${suffix}.json")
                cuda_visible_devices = [string]$receipt.cuda_visible_devices
                internal_device = [string]$receipt.internal_device
            }
        }
        catch {
            $rejections += "$($file.Name): $($_.Exception.Message)"
        }
    }
    if ($bindingMatches.Count -ne 1) {
        throw (
            "attach PID $processId requires exactly one post-creation trainer preflight " +
            "receipt, found $($bindingMatches.Count); rejected=$($rejections -join ' | ')"
        )
    }
    return [ordered]@{
        status = 'verified_attach_identity'
        process_id = $processId
        executable_path = $actualExecutable
        command_line = [string]$ProcessRecord.command_line
        creation_utc = $created.ToString('o')
        seed = $Seed
        physical_gpu = $Gpu
        cuda_visible_devices = [string]$Gpu
        internal_device = 'cuda:0'
        trainer_preflight = $bindingMatches[0]
    }
}

function Confirm-LiveAttachProcess {
    param(
        [Parameter(Mandatory = $true)][int]$ProcessId,
        [Parameter(Mandatory = $true)][string]$SeedDirectory,
        [Parameter(Mandatory = $true)][int]$Seed,
        [Parameter(Mandatory = $true)][int]$Gpu
    )
    if ($ProcessId -le 0) {
        throw 'AttachExistingPid must be a positive PID'
    }
    $wmi = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction Stop
    if ($null -eq $wmi) {
        throw "attach PID $ProcessId is not live"
    }
    $sameSeedProcesses = @(
        Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            $_.CommandLine -and
            $_.CommandLine -match 'train_stageb_train399\.py"?\s+train' -and
            $_.CommandLine -match "--seed\s+$Seed(?:\s|$)"
        }
    )
    if ($sameSeedProcesses.Count -ne 1 -or [int]$sameSeedProcesses[0].ProcessId -ne $ProcessId) {
        throw "attach seed $Seed does not have exactly one live trainer at PID $ProcessId"
    }
    $record = [ordered]@{
        process_id = [int]$wmi.ProcessId
        executable_path = [string]$wmi.ExecutablePath
        command_line = [string]$wmi.CommandLine
        creation_utc = ([DateTime]$wmi.CreationDate).ToUniversalTime().ToString('o')
    }
    $identity = Confirm-AttachIdentityRecord `
        -ProcessRecord $record `
        -SeedDirectory $SeedDirectory `
        -Seed $Seed `
        -Gpu $Gpu
    $process = [System.Diagnostics.Process]::GetProcessById($ProcessId)
    return [ordered]@{
        identity = $identity
        process = $process
    }
}

function Invoke-NvidiaSmiQuery {
    param([Parameter(Mandatory = $true)][int]$Gpu)
    $arguments = @(
        '-i', [string]$Gpu,
        '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'
    )
    $raw = @(& nvidia-smi @arguments 2>&1)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "nvidia-smi query failed with exit code $exitCode`: $($raw -join ' ')"
    }
    $lines = @($raw | ForEach-Object { [string]$_ } | Where-Object { $_.Trim() })
    if ($lines.Count -ne 1) {
        throw "nvidia-smi returned $($lines.Count) rows for physical GPU $Gpu"
    }
    $fields = @($lines[0].Split(',') | ForEach-Object { $_.Trim() })
    if ($fields.Count -ne 7) {
        throw "nvidia-smi returned an unexpected field count for physical GPU $Gpu"
    }
    $numbers = @{}
    foreach ($index in @(0, 3, 4, 5, 6)) {
        $parsed = 0
        if (-not [int]::TryParse($fields[$index], [ref]$parsed)) {
            throw "nvidia-smi returned a non-integer numeric field: $($fields[$index])"
        }
        $numbers[$index] = $parsed
    }
    if ($numbers[0] -ne $Gpu) {
        throw "nvidia-smi physical GPU index mismatch: expected $Gpu got $($numbers[0])"
    }
    return [ordered]@{
        sampled_utc = Get-UtcTimestamp
        index = $numbers[0]
        uuid = $fields[1]
        name = $fields[2]
        memory_total_mib = $numbers[3]
        memory_used_mib = $numbers[4]
        memory_free_mib = $numbers[5]
        utilization_gpu_percent = $numbers[6]
    }
}

function Invoke-GpuPreflight {
    param(
        [Parameter(Mandatory = $true)][int]$Gpu,
        [Parameter(Mandatory = $true)][int]$Seed,
        [Parameter(Mandatory = $true)][string]$OutputPath,
        [Parameter(Mandatory = $true)][int]$Samples,
        [Parameter(Mandatory = $true)][int]$SampleIntervalSeconds,
        [Parameter(Mandatory = $true)][int]$ExpectedPeakMiB,
        [Parameter(Mandatory = $true)][int]$FreeMarginMiB,
        [Parameter(Mandatory = $true)][int]$MaximumUtilizationPercent
    )
    $rows = @()
    $failureReasons = @()
    for ($index = 0; $index -lt $Samples; $index++) {
        $row = Invoke-NvidiaSmiQuery -Gpu $Gpu
        $row['post_training_margin_mib'] = [int]$row.memory_free_mib - $ExpectedPeakMiB
        $rows += $row
        if ($row.post_training_margin_mib -lt $FreeMarginMiB) {
            $failureReasons += (
                "sample $($index + 1): projected free margin " +
                "$($row.post_training_margin_mib) MiB < $FreeMarginMiB MiB"
            )
        }
        if ($row.utilization_gpu_percent -gt $MaximumUtilizationPercent) {
            $failureReasons += (
                "sample $($index + 1): utilization " +
                "$($row.utilization_gpu_percent)% > $MaximumUtilizationPercent%"
            )
        }
        if ($index + 1 -lt $Samples -and $SampleIntervalSeconds -gt 0) {
            Start-Sleep -Seconds $SampleIntervalSeconds
        }
    }
    $fullSnapshot = @(& nvidia-smi -i $Gpu 2>&1 | ForEach-Object { [string]$_ })
    $fullSnapshotExitCode = $LASTEXITCODE
    if ($fullSnapshotExitCode -ne 0) {
        $failureReasons += "full nvidia-smi snapshot failed with exit code $fullSnapshotExitCode"
    }
    $payload = [ordered]@{
        schema_version = 'PHAxis-StageB-train399-queue-GPU-preflight-1.0'
        status = $(if ($failureReasons.Count -eq 0) { 'passed' } else { 'failed' })
        seed = $Seed
        physical_gpu = $Gpu
        expected_training_peak_mib = $ExpectedPeakMiB
        required_free_margin_mib = $FreeMarginMiB
        maximum_gpu_utilization_percent = $MaximumUtilizationPercent
        sample_count = $Samples
        sample_interval_seconds = $SampleIntervalSeconds
        samples = @($rows)
        full_nvidia_smi_exit_code = $fullSnapshotExitCode
        full_nvidia_smi = @($fullSnapshot)
        failure_reasons = @($failureReasons)
        cuda_was_initialized_by_queue_preflight = $false
    }
    Write-AtomicJson -Path $OutputPath -Payload $payload
    if ($failureReasons.Count -gt 0) {
        throw "GPU preflight failed for seed $Seed`: $($failureReasons -join '; ')"
    }
    return $payload
}

function Read-JsonObject {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not [System.IO.File]::Exists($Path)) {
        throw "required JSON file is missing: $Path"
    }
    try {
        return Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json -Depth 64
    }
    catch {
        throw "invalid JSON file $Path`: $($_.Exception.Message)"
    }
}

function Assert-FiniteNumber {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$Description
    )
    $number = [double]$Value
    if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) {
        throw "$Description is non-finite"
    }
}

function Test-OneCompletionReceipt {
    param(
        [Parameter(Mandatory = $true)][System.IO.FileInfo]$ReceiptFile,
        [Parameter(Mandatory = $true)][string]$SeedDirectory,
        [Parameter(Mandatory = $true)][int]$Seed,
        [Parameter(Mandatory = $true)][int]$Gpu,
        [Nullable[DateTime]]$NotBeforeUtc
    )
    if ($null -ne $NotBeforeUtc) {
        # PowerShell unwraps Nullable[DateTime] values supplied to a function;
        # strict mode therefore rejects `.Value` even though the parameter is
        # declared nullable.  Cast the bound value explicitly before applying
        # the two-second filesystem timestamp tolerance.
        $notBeforeValue = ([DateTime]$NotBeforeUtc).ToUniversalTime()
        if ($ReceiptFile.LastWriteTimeUtc -lt $notBeforeValue.AddSeconds(-2)) {
            throw 'receipt predates this training invocation'
        }
    }
    $receipt = Read-JsonObject -Path $ReceiptFile.FullName
    $expectedReceiptFields = [ordered]@{
        schema_version = 'PHAxis-StageB-train399-training-receipt-1.0'
        status = 'completed'
        formal_training = $true
        seed = $Seed
        epochs = $script:ExpectedEpochs
        steps_per_epoch = $script:ExpectedStepsPerEpoch
        global_steps = $script:ExpectedGlobalSteps
        cuda_visible_devices = [string]$Gpu
        internal_device = 'cuda:0'
        nvidia_smi_preflight_status = 'passed'
        nvidia_smi_training_monitor_status = 'passed'
        validation_evaluated_during_training = $false
        blind_images_used = 0
    }
    foreach ($field in $expectedReceiptFields.Keys) {
        if ($receipt.PSObject.Properties.Name -notcontains $field) {
            throw "receipt lacks required field $field"
        }
        if ([string]$receipt.$field -ne [string]$expectedReceiptFields[$field]) {
            throw "receipt field $field mismatch: $($receipt.$field) != $($expectedReceiptFields[$field])"
        }
    }

    $checkpoint = Join-Path $SeedDirectory 'last.pt'
    if (-not [System.IO.File]::Exists($checkpoint)) {
        throw "checkpoint is missing: $checkpoint"
    }
    if ((ConvertTo-CanonicalPath -Path ([string]$receipt.checkpoint)) -ne (ConvertTo-CanonicalPath -Path $checkpoint)) {
        throw 'receipt checkpoint path does not identify this seed last.pt'
    }
    $actualCheckpointHash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
    $receiptCheckpointHash = ([string]$receipt.checkpoint_sha256).ToLowerInvariant()
    if ($receiptCheckpointHash -notmatch '^[0-9a-f]{64}$' -or $actualCheckpointHash -ne $receiptCheckpointHash) {
        throw "checkpoint SHA-256 mismatch: actual=$actualCheckpointHash receipt=$receiptCheckpointHash"
    }

    $historyPath = Join-Path $SeedDirectory 'history.json'
    $history = @(Read-JsonObject -Path $historyPath)
    if ($history.Count -ne $script:ExpectedEpochs) {
        throw "history has $($history.Count) rows instead of $($script:ExpectedEpochs)"
    }
    for ($index = 0; $index -lt $history.Count; $index++) {
        $expectedEpoch = $index + 1
        $expectedStep = $expectedEpoch * $script:ExpectedStepsPerEpoch
        $row = $history[$index]
        if ([int]$row.epoch -ne $expectedEpoch) {
            throw "history epoch sequence fails at row $expectedEpoch"
        }
        if ([int]$row.batches -ne $script:ExpectedStepsPerEpoch) {
            throw "history batch count fails at epoch $expectedEpoch"
        }
        if ([int]$row.global_step -ne $expectedStep) {
            throw "history global step fails at epoch $expectedEpoch"
        }
        if ([bool]$row.validation_evaluated) {
            throw "validation was evaluated during training at epoch $expectedEpoch"
        }
        Assert-FiniteNumber -Value $row.train_loss_total -Description "epoch $expectedEpoch train loss"
    }

    $config = Read-JsonObject -Path (Join-Path $SeedDirectory 'config.json')
    $expectedConfig = [ordered]@{
        epochs = 60
        batch_size = 8
        crops_per_image = 8
        fixed_last_epoch_policy = $true
        amp = $true
        amp_initial_scale = 1024.0
        amp_growth_interval = 1000000
    }
    foreach ($field in $expectedConfig.Keys) {
        if ([string]$config.$field -ne [string]$expectedConfig[$field]) {
            throw "formal config field $field mismatch"
        }
    }

    $contract = Read-JsonObject -Path (Join-Path $SeedDirectory 'training_contract.json')
    $expectedContract = [ordered]@{
        schema_version = 'PHAxis-StageB-train399-checkpoint-contract-1.0'
        formal_training = $true
        seed = $Seed
        member_id = "seed_$Seed"
        blind_images_used = 0
        pyRootHair_called_or_copied = $false
        validation_labels_used_for_gradient_or_early_stopping = $false
    }
    foreach ($field in $expectedContract.Keys) {
        if ([string]$contract.$field -ne [string]$expectedContract[$field]) {
            throw "training contract field $field mismatch"
        }
    }

    $initialization = Read-JsonObject -Path (Join-Path $SeedDirectory 'initialization.json')
    if ([bool]$initialization.historical_stageb_checkpoint_loaded) {
        throw 'historical Stage-B checkpoint was loaded'
    }

    return [ordered]@{
        status = 'verified_complete'
        seed = $Seed
        epochs = $script:ExpectedEpochs
        steps_per_epoch = $script:ExpectedStepsPerEpoch
        global_steps = $script:ExpectedGlobalSteps
        receipt = $ReceiptFile.FullName
        receipt_sha256 = (Get-FileHash -LiteralPath $ReceiptFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        checkpoint = $checkpoint
        checkpoint_sha256 = $actualCheckpointHash
        history = $historyPath
        history_sha256 = (Get-FileHash -LiteralPath $historyPath -Algorithm SHA256).Hash.ToLowerInvariant()
        cuda_visible_devices = [string]$Gpu
        internal_device = 'cuda:0'
        validation_evaluated_during_training = $false
        blind_images_used = 0
    }
}

function Confirm-CompletedSeed {
    param(
        [Parameter(Mandatory = $true)][string]$SeedDirectory,
        [Parameter(Mandatory = $true)][int]$Seed,
        [Parameter(Mandatory = $true)][int]$Gpu,
        [Nullable[DateTime]]$NotBeforeUtc
    )
    $canonicalSeedDirectory = ConvertTo-CanonicalPath -Path $SeedDirectory
    if (-not [System.IO.Directory]::Exists($canonicalSeedDirectory)) {
        throw "seed directory is missing: $canonicalSeedDirectory"
    }
    $receipts = @(
        Get-ChildItem -LiteralPath $canonicalSeedDirectory -File -Filter 'training_receipt*.json' |
            Sort-Object LastWriteTimeUtc -Descending
    )
    if ($receipts.Count -eq 0) {
        throw "no completion receipt exists for seed $Seed"
    }
    $failures = @()
    foreach ($receiptFile in $receipts) {
        try {
            return Test-OneCompletionReceipt `
                -ReceiptFile $receiptFile `
                -SeedDirectory $canonicalSeedDirectory `
                -Seed $Seed `
                -Gpu $Gpu `
                -NotBeforeUtc $NotBeforeUtc
        }
        catch {
            $failures += "$($receiptFile.Name): $($_.Exception.Message)"
        }
    }
    throw "no valid completion receipt exists for seed $Seed`: $($failures -join ' | ')"
}

function Get-SeedLaunchMode {
    param(
        [Parameter(Mandatory = $true)]$PlanEntry,
        [Parameter(Mandatory = $true)][string]$SeedDirectory
    )
    $checkpoint = Join-Path $SeedDirectory 'last.pt'
    if ([System.IO.File]::Exists($checkpoint)) {
        return 'resume'
    }
    if ([string]$PlanEntry.initial_policy -eq 'resume-required-if-incomplete') {
        throw "seed $($PlanEntry.seed) is incomplete but has no last.pt to resume"
    }
    if ([System.IO.Directory]::Exists($SeedDirectory)) {
        $existing = @(Get-ChildItem -LiteralPath $SeedDirectory -Force)
        if ($existing.Count -gt 0) {
            throw "refusing fresh training in non-empty checkpoint-free seed directory: $SeedDirectory"
        }
    }
    return 'fresh'
}

function Start-HiddenTrainingChild {
    param(
        [Parameter(Mandatory = $true)][int]$Seed,
        [Parameter(Mandatory = $true)][int]$Gpu,
        [Parameter(Mandatory = $true)][string]$LaunchMode,
        [Parameter(Mandatory = $true)][string]$StdoutPath,
        [Parameter(Mandatory = $true)][string]$StderrPath
    )
    $arguments = @(
        ('"' + $script:TrainingScript + '"'),
        'train',
        '--seed', [string]$Seed,
        '--device', 'cuda:0'
    )
    if ($LaunchMode -eq 'resume') {
        $arguments += '--resume'
    }
    $oldVisible = [Environment]::GetEnvironmentVariable('CUDA_VISIBLE_DEVICES', 'Process')
    $oldOrder = [Environment]::GetEnvironmentVariable('CUDA_DEVICE_ORDER', 'Process')
    try {
        [Environment]::SetEnvironmentVariable('CUDA_VISIBLE_DEVICES', [string]$Gpu, 'Process')
        [Environment]::SetEnvironmentVariable('CUDA_DEVICE_ORDER', 'PCI_BUS_ID', 'Process')
        $process = Start-Process `
            -FilePath $script:PythonExecutable `
            -ArgumentList ($arguments -join ' ') `
            -WorkingDirectory $script:ProjectRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput $StdoutPath `
            -RedirectStandardError $StderrPath `
            -PassThru
    }
    finally {
        [Environment]::SetEnvironmentVariable('CUDA_VISIBLE_DEVICES', $oldVisible, 'Process')
        [Environment]::SetEnvironmentVariable('CUDA_DEVICE_ORDER', $oldOrder, 'Process')
    }
    return $process
}

function Invoke-Queue {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][int]$Gpu,
        [int]$AttachPid = 0
    )
    if ($AttachPid -lt 0) {
        throw 'AttachExistingPid cannot be negative'
    }
    if ($AttachPid -gt 0 -and $Name -ne 'odd-resume') {
        throw 'AttachExistingPid is permitted only for the odd-resume lane first seed'
    }
    Assert-CoreFiles
    [System.IO.Directory]::CreateDirectory($script:QueueRoot) | Out-Null
    [System.IO.Directory]::CreateDirectory($script:LockRoot) | Out-Null
    $startedUtc = Get-UtcTimestamp
    $queueId = ([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')) + "_${Name}_gpu${Gpu}_pid${PID}"
    $runDirectory = Join-Path $script:QueueRoot $queueId
    [System.IO.Directory]::CreateDirectory($runDirectory) | Out-Null
    $planPayload = Get-QueuePlanPayload -Name $Name -Gpu $Gpu
    $planPayload['queue_id'] = $queueId
    $planPayload['started_utc'] = $startedUtc
    $planPayload['supervisor_pid'] = $PID
    Write-AtomicJson -Path (Join-Path $runDirectory 'queue_plan.json') -Payload $planPayload

    $gpuLockPath = Join-Path $script:LockRoot "physical_gpu_${Gpu}.lock.json"
    $gpuLock = $null
    $queueRecords = @()
    $activeSeedLock = $null
    $activeSeed = $null
    try {
        $gpuLock = Enter-ExclusiveFileLock -Path $gpuLockPath -Metadata ([ordered]@{
            schema_version = 'PHAxis-StageB-train399-exclusive-lock-1.0'
            status = 'held'
            scope = 'physical_gpu'
            physical_gpu = $Gpu
            lane = $Name
            queue_id = $queueId
            supervisor_pid = $PID
            host = [Environment]::MachineName
            acquired_utc = Get-UtcTimestamp
        })

        foreach ($entry in (Get-LanePlan -Name $Name)) {
            $seed = [int]$entry.seed
            $activeSeed = $seed
            $seedDirectory = Join-Path $script:ModelRoot "seed_$seed"
            $seedLockPath = Join-Path $script:LockRoot "seed_${seed}.lock.json"
            $activeSeedLock = Enter-ExclusiveFileLock -Path $seedLockPath -Metadata ([ordered]@{
                schema_version = 'PHAxis-StageB-train399-exclusive-lock-1.0'
                status = 'held'
                scope = 'seed'
                seed = $seed
                physical_gpu = $Gpu
                lane = $Name
                queue_id = $queueId
                supervisor_pid = $PID
                host = [Environment]::MachineName
                acquired_utc = Get-UtcTimestamp
            })

            $alreadyComplete = $null
            try {
                $alreadyComplete = Confirm-CompletedSeed `
                    -SeedDirectory $seedDirectory `
                    -Seed $seed `
                    -Gpu $Gpu
            }
            catch {
                $alreadyComplete = $null
            }
            if ($null -ne $alreadyComplete) {
                $queueRecords += [ordered]@{
                    seed = $seed
                    status = 'verified_complete_skipped'
                    verified_utc = Get-UtcTimestamp
                    verification = $alreadyComplete
                }
                Exit-ExclusiveFileLock -Stream $activeSeedLock -Metadata ([ordered]@{
                    schema_version = 'PHAxis-StageB-train399-exclusive-lock-1.0'
                    status = 'released'
                    scope = 'seed'
                    seed = $seed
                    queue_id = $queueId
                    released_utc = Get-UtcTimestamp
                    outcome = 'verified_complete_skipped'
                })
                $activeSeedLock = $null
                $activeSeed = $null
                Write-AtomicJson -Path (Join-Path $runDirectory 'queue_status.json') -Payload ([ordered]@{
                    schema_version = 'PHAxis-StageB-train399-GPU-queue-status-1.0'
                    status = 'running'
                    queue_id = $queueId
                    lane = $Name
                    physical_gpu = $Gpu
                    records = @($queueRecords)
                    updated_utc = Get-UtcTimestamp
                })
                continue
            }

            if ($AttachPid -gt 0 -and $seed -eq 2026082801) {
                $attached = Confirm-LiveAttachProcess `
                    -ProcessId $AttachPid `
                    -SeedDirectory $seedDirectory `
                    -Seed $seed `
                    -Gpu $Gpu
                $identity = $attached.identity
                $process = $attached.process
                $launchUtcValue = [DateTimeOffset]::Parse(
                    [string]$identity.creation_utc,
                    [System.Globalization.CultureInfo]::InvariantCulture
                ).UtcDateTime
                $attachRecord = [ordered]@{
                    seed = $seed
                    status = 'attached_running'
                    launch_mode = 'attach-existing-resume'
                    attached_utc = Get-UtcTimestamp
                    child_pid = $process.Id
                    child_window_style = 'pre-existing-hidden-process'
                    cuda_visible_devices = [string]$Gpu
                    internal_device = 'cuda:0'
                    identity = $identity
                    process_termination_policy = 'wait-only-never-kill-or-suspend'
                }
                Write-AtomicJson -Path (Join-Path $runDirectory "seed_${seed}_attach.json") -Payload $attachRecord
                Write-AtomicJson -Path (Join-Path $runDirectory 'queue_status.json') -Payload ([ordered]@{
                    schema_version = 'PHAxis-StageB-train399-GPU-queue-status-1.0'
                    status = 'running'
                    queue_id = $queueId
                    lane = $Name
                    physical_gpu = $Gpu
                    completed_records = @($queueRecords)
                    active = $attachRecord
                    updated_utc = Get-UtcTimestamp
                })

                # Attach is observation-only: no process termination, suspension,
                # stdin write, signal, or job-object assignment exists here.
                $process.WaitForExit()
                $process.Refresh()
                if ($process.ExitCode -ne 0) {
                    throw "attached seed $seed PID $($process.Id) exited with code $($process.ExitCode)"
                }
                $requiredReceipt = [string]$identity.trainer_preflight.expected_completion_receipt
                if (-not [System.IO.File]::Exists($requiredReceipt)) {
                    throw "attached seed $seed did not create its suffix-bound completion receipt: $requiredReceipt"
                }
                $receiptFile = Get-Item -LiteralPath $requiredReceipt
                $verification = Test-OneCompletionReceipt `
                    -ReceiptFile $receiptFile `
                    -SeedDirectory $seedDirectory `
                    -Seed $seed `
                    -Gpu $Gpu `
                    -NotBeforeUtc $launchUtcValue
                $queueRecords += [ordered]@{
                    seed = $seed
                    status = 'attached_completed_and_verified'
                    launch_mode = 'attach-existing-resume'
                    attached_utc = $attachRecord.attached_utc
                    finished_utc = Get-UtcTimestamp
                    child_pid = $process.Id
                    child_exit_code = $process.ExitCode
                    identity = $identity
                    verification = $verification
                }
                Exit-ExclusiveFileLock -Stream $activeSeedLock -Metadata ([ordered]@{
                    schema_version = 'PHAxis-StageB-train399-exclusive-lock-1.0'
                    status = 'released'
                    scope = 'seed'
                    seed = $seed
                    queue_id = $queueId
                    released_utc = Get-UtcTimestamp
                    outcome = 'attached_completed_and_verified'
                    checkpoint_sha256 = $verification.checkpoint_sha256
                })
                $activeSeedLock = $null
                $activeSeed = $null
                $AttachPid = 0
                Write-AtomicJson -Path (Join-Path $runDirectory 'queue_status.json') -Payload ([ordered]@{
                    schema_version = 'PHAxis-StageB-train399-GPU-queue-status-1.0'
                    status = 'running'
                    queue_id = $queueId
                    lane = $Name
                    physical_gpu = $Gpu
                    records = @($queueRecords)
                    updated_utc = Get-UtcTimestamp
                })
                continue
            }

            Assert-NoDuplicateSeedProcess -Seed $seed
            $launchMode = Get-SeedLaunchMode -PlanEntry $entry -SeedDirectory $seedDirectory
            $preflightPath = Join-Path $runDirectory "seed_${seed}_nvidia_smi_preflight.json"
            $preflight = Invoke-GpuPreflight `
                -Gpu $Gpu `
                -Seed $seed `
                -OutputPath $preflightPath `
                -Samples $GpuSamples `
                -SampleIntervalSeconds $GpuSampleIntervalSeconds `
                -ExpectedPeakMiB $ExpectedTrainingPeakMiB `
                -FreeMarginMiB $RequiredFreeMarginMiB `
                -MaximumUtilizationPercent $MaximumGpuUtilizationPercent

            # Re-check after the multi-sample preflight to close the startup race.
            Assert-NoDuplicateSeedProcess -Seed $seed
            $launchUtcValue = [DateTime]::UtcNow
            $stdoutPath = Join-Path $runDirectory "seed_${seed}_${launchMode}.stdout.log"
            $stderrPath = Join-Path $runDirectory "seed_${seed}_${launchMode}.stderr.log"
            $process = Start-HiddenTrainingChild `
                -Seed $seed `
                -Gpu $Gpu `
                -LaunchMode $launchMode `
                -StdoutPath $stdoutPath `
                -StderrPath $stderrPath
            $launchRecord = [ordered]@{
                seed = $seed
                status = 'running'
                launch_mode = $launchMode
                launched_utc = $launchUtcValue.ToString('o')
                child_pid = $process.Id
                child_window_style = 'Hidden'
                cuda_visible_devices = [string]$Gpu
                cuda_device_order = 'PCI_BUS_ID'
                internal_device = 'cuda:0'
                stdout = $stdoutPath
                stderr = $stderrPath
                nvidia_smi_preflight = $preflightPath
                nvidia_smi_preflight_status = $preflight.status
            }
            Write-AtomicJson -Path (Join-Path $runDirectory "seed_${seed}_launch.json") -Payload $launchRecord
            Write-AtomicJson -Path (Join-Path $runDirectory 'queue_status.json') -Payload ([ordered]@{
                schema_version = 'PHAxis-StageB-train399-GPU-queue-status-1.0'
                status = 'running'
                queue_id = $queueId
                lane = $Name
                physical_gpu = $Gpu
                completed_records = @($queueRecords)
                active = $launchRecord
                updated_utc = Get-UtcTimestamp
            })

            $process.WaitForExit()
            $process.Refresh()
            if ($process.ExitCode -ne 0) {
                throw "seed $seed training child PID $($process.Id) exited with code $($process.ExitCode); inspect $stderrPath"
            }
            $verification = Confirm-CompletedSeed `
                -SeedDirectory $seedDirectory `
                -Seed $seed `
                -Gpu $Gpu `
                -NotBeforeUtc $launchUtcValue
            $queueRecords += [ordered]@{
                seed = $seed
                status = 'completed_and_verified'
                launch_mode = $launchMode
                launched_utc = $launchUtcValue.ToString('o')
                finished_utc = Get-UtcTimestamp
                child_pid = $process.Id
                child_exit_code = $process.ExitCode
                stdout = $stdoutPath
                stderr = $stderrPath
                nvidia_smi_preflight = $preflightPath
                verification = $verification
            }
            Exit-ExclusiveFileLock -Stream $activeSeedLock -Metadata ([ordered]@{
                schema_version = 'PHAxis-StageB-train399-exclusive-lock-1.0'
                status = 'released'
                scope = 'seed'
                seed = $seed
                queue_id = $queueId
                released_utc = Get-UtcTimestamp
                outcome = 'completed_and_verified'
                checkpoint_sha256 = $verification.checkpoint_sha256
            })
            $activeSeedLock = $null
            $activeSeed = $null
            Write-AtomicJson -Path (Join-Path $runDirectory 'queue_status.json') -Payload ([ordered]@{
                schema_version = 'PHAxis-StageB-train399-GPU-queue-status-1.0'
                status = 'running'
                queue_id = $queueId
                lane = $Name
                physical_gpu = $Gpu
                records = @($queueRecords)
                updated_utc = Get-UtcTimestamp
            })
        }

        $complete = [ordered]@{
            schema_version = 'PHAxis-StageB-train399-GPU-queue-receipt-1.0'
            status = 'completed'
            queue_id = $queueId
            lane = $Name
            physical_gpu = $Gpu
            cuda_visible_devices = [string]$Gpu
            started_utc = $startedUtc
            completed_utc = Get-UtcTimestamp
            expected_global_steps_per_seed = $script:ExpectedGlobalSteps
            stop_on_first_failure = $true
            records = @($queueRecords)
        }
        Write-AtomicJson -Path (Join-Path $runDirectory 'queue_receipt.json') -Payload $complete
        Write-AtomicJson -Path (Join-Path $runDirectory 'queue_status.json') -Payload $complete
        return $complete
    }
    catch {
        $failure = [ordered]@{
            schema_version = 'PHAxis-StageB-train399-GPU-queue-failure-1.0'
            status = 'failed'
            queue_id = $queueId
            lane = $Name
            physical_gpu = $Gpu
            started_utc = $startedUtc
            failed_utc = Get-UtcTimestamp
            active_seed = $activeSeed
            exception_type = $_.Exception.GetType().FullName
            message = $_.Exception.Message
            stop_on_first_failure = $true
            records = @($queueRecords)
        }
        Write-AtomicJson -Path (Join-Path $runDirectory 'queue_failure.json') -Payload $failure
        Write-AtomicJson -Path (Join-Path $runDirectory 'queue_status.json') -Payload $failure
        throw
    }
    finally {
        if ($null -ne $activeSeedLock) {
            Exit-ExclusiveFileLock -Stream $activeSeedLock -Metadata ([ordered]@{
                schema_version = 'PHAxis-StageB-train399-exclusive-lock-1.0'
                status = 'released'
                scope = 'seed'
                seed = $activeSeed
                queue_id = $queueId
                released_utc = Get-UtcTimestamp
                outcome = 'failed_or_interrupted_supervisor'
            })
        }
        if ($null -ne $gpuLock) {
            Exit-ExclusiveFileLock -Stream $gpuLock -Metadata ([ordered]@{
                schema_version = 'PHAxis-StageB-train399-exclusive-lock-1.0'
                status = 'released'
                scope = 'physical_gpu'
                physical_gpu = $Gpu
                lane = $Name
                queue_id = $queueId
                released_utc = Get-UtcTimestamp
            })
        }
    }
}

switch ($PSCmdlet.ParameterSetName) {
    'Plan' {
        Assert-CoreFiles
        Get-QueuePlanPayload -Name $Lane -Gpu $PhysicalGpu | ConvertTo-Json -Depth 32
    }
    'Verify' {
        Confirm-CompletedSeed `
            -SeedDirectory $VerifySeedDirectory `
            -Seed $VerifySeed `
            -Gpu $ExpectedPhysicalGpu `
            -NotBeforeUtc $VerifyReceiptNotBeforeUtc | ConvertTo-Json -Depth 32
    }
    'AuditAttach' {
        $fixture = Read-JsonObject -Path $AuditAttachFixture
        Confirm-AttachIdentityRecord `
            -ProcessRecord $fixture.process `
            -SeedDirectory ([string]$fixture.seed_directory) `
            -Seed $AuditAttachSeed `
            -Gpu $AuditAttachPhysicalGpu | ConvertTo-Json -Depth 32
    }
    'AuditLiveAttach' {
        $liveAudit = Confirm-LiveAttachProcess `
            -ProcessId $AuditLiveAttachPid `
            -SeedDirectory (Join-Path $script:ModelRoot "seed_$AuditLiveAttachSeed") `
            -Seed $AuditLiveAttachSeed `
            -Gpu $AuditLiveAttachPhysicalGpu
        $liveAudit.identity | ConvertTo-Json -Depth 32
    }
    'Run' {
        Invoke-Queue `
            -Name $Lane `
            -Gpu $PhysicalGpu `
            -AttachPid $AttachExistingPid | ConvertTo-Json -Depth 64
    }
    default { throw "unexpected parameter set: $($PSCmdlet.ParameterSetName)" }
}
