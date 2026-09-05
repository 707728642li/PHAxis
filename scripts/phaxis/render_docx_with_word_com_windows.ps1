param(
    [Parameter(Mandatory = $true)][string]$InputDocx,
    [Parameter(Mandatory = $true)][string]$OutputPdf,
    [Parameter(Mandatory = $true)][string]$StatusJson
)

$ErrorActionPreference = 'Stop'
$word = $null
$document = $null
$wordProcessId = $null
$preExistingWordIds = @(Get-Process WINWORD -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
$started = [DateTime]::UtcNow
$payload = [ordered]@{
    schema_version = 'PHAxis-word-com-docx-render-1.0'
    status = 'started'
    input_docx = [System.IO.Path]::GetFullPath($InputDocx)
    output_pdf = [System.IO.Path]::GetFullPath($OutputPdf)
    renderer = 'Microsoft Word COM ExportAsFixedFormat PDF'
    renderer_version = $null
    word_visible = $false
    read_only = $true
    macros_enabled = $false
    pages = $null
    word_process_id = $null
    task_owned_process_only = $true
    forced_task_owned_process_cleanup = $false
    error = $null
    started_utc = $started.ToString('o')
    completed_utc = $null
}

try {
    $inputPath = [System.IO.Path]::GetFullPath($InputDocx)
    $outputPath = [System.IO.Path]::GetFullPath($OutputPdf)
    $statusPath = [System.IO.Path]::GetFullPath($StatusJson)
    if (-not [System.IO.File]::Exists($inputPath)) {
        throw "Input DOCX does not exist: $inputPath"
    }
    if ([System.IO.File]::Exists($outputPath) -or [System.IO.File]::Exists($statusPath)) {
        throw 'Refusing to overwrite PDF or renderer status.'
    }
    [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($outputPath)) | Out-Null
    [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($statusPath)) | Out-Null

    $word = New-Object -ComObject Word.Application
    $newWordProcesses = @(Get-Process WINWORD -ErrorAction SilentlyContinue | Where-Object { $_.Id -notin $preExistingWordIds })
    if ($newWordProcesses.Count -ne 1) {
        throw "Expected one task-owned WINWORD process, found $($newWordProcesses.Count)."
    }
    $wordProcessId = [int]$newWordProcesses[0].Id
    $payload.word_process_id = $wordProcessId
    $payload.renderer_version = [string]$word.Version
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $word.Options.SaveNormalPrompt = $false
    $document = $word.Documents.Open($inputPath, $false, $true, $false)
    $payload.pages = [int]$document.ComputeStatistics(2)
    $document.ExportAsFixedFormat($outputPath, 17)
    if (-not [System.IO.File]::Exists($outputPath)) {
        throw "Word returned without creating PDF: $outputPath"
    }
    $payload.status = 'complete'
}
catch {
    $payload.status = 'failed'
    $payload.error = $_.Exception.ToString()
    throw
}
finally {
    if ($null -ne $document) {
        try { $document.Close($false) } catch {}
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)
    }
    if ($null -ne $word) {
        try { $word.Quit($false) } catch {}
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
    if ($null -ne $wordProcessId) {
        $deadline = [DateTime]::UtcNow.AddSeconds(5)
        while ((Get-Process -Id $wordProcessId -ErrorAction SilentlyContinue) -and [DateTime]::UtcNow -lt $deadline) {
            Start-Sleep -Milliseconds 200
        }
        $taskOwnedProcess = Get-Process -Id $wordProcessId -ErrorAction SilentlyContinue
        if ($null -ne $taskOwnedProcess -and $wordProcessId -notin $preExistingWordIds) {
            Stop-Process -Id $wordProcessId -Force
            Wait-Process -Id $wordProcessId -Timeout 5 -ErrorAction SilentlyContinue
            # Stop-Process can return while Windows still exposes the process
            # object for a short interval.  Poll only the PID that this wrapper
            # proved was absent before launching Word; never broaden cleanup to
            # a process name or to any pre-existing Word instance.
            $forcedCleanupDeadline = [DateTime]::UtcNow.AddSeconds(5)
            while ((Get-Process -Id $wordProcessId -ErrorAction SilentlyContinue) -and [DateTime]::UtcNow -lt $forcedCleanupDeadline) {
                Start-Sleep -Milliseconds 100
            }
            if (Get-Process -Id $wordProcessId -ErrorAction SilentlyContinue) {
                throw "Task-owned WINWORD process $wordProcessId survived cleanup."
            }
            $payload.forced_task_owned_process_cleanup = $true
        }
    }
    $payload.completed_utc = [DateTime]::UtcNow.ToString('o')
    $payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $StatusJson -Encoding UTF8
}
