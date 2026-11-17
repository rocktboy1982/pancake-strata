# win_cuda_queue.ps1 - one-at-a-time CUDA queue for the owner's PC (GTX 1660 Ti 6 GB, sm_75, Windows, PowerShell 5.1).
# Plan: tools\pancake\win_cuda_queue_plan.json (order, needs, timeouts).  Results: bench\results\<date>-gtx1660ti-queue\<item>\.
#   -Plan   print the order and what each step needs; runs and creates nothing
#   -Now    ignore the night window; every other guard stays (idle check, RAM, VRAM, AC power, STOP file, idle wait)
# It never kills a process it did not start, never changes power or system settings, and starts only inside the night
# window (default 23:00-06:30 local) unless -Now.  Only one copy runs (named mutex).  Exit codes: 0 finished or left
# cleanly (outside the window, STOP file, window end), 1 an item failed or timed out, 4 machine never became idle, 5 another copy runs.
# Timeout: the one process this script started for the step (and its children) is stopped by its own id; nothing is
# ever matched by name.  Not registered as a scheduled task: that is a separate step the owner confirms.
param(
    [switch]$Plan,
    [switch]$Now,
    [switch]$WithCtest,
    [switch]$Force,
    [string]$Only = '',
    [string]$WindowStart = '23:00',
    [string]$WindowEnd = '06:30',
    [string]$BuildDir = 'C:\Users\iancu\build-latest\build-sm75',
    [string]$SrcDir = 'C:\Users\iancu\build-latest\src',
    [string]$Model = 'C:\Users\iancu\models\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf',
    [string]$Tokenizer = 'C:\Users\iancu\build-latest\tok\tokenizer.json',
    [string]$Python = 'C:\Users\iancu\build-latest\venv\Scripts\python.exe',
    [string]$CudaBin = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin',
    [string]$Vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
    [string]$StopFile = 'C:\Users\iancu\build-latest\STOP',
    [string]$PlanFile = '',
    [string]$OutRoot = '',
    [int]$IdleMin = 3,
    [int]$MaxGpuPct = 15,
    [int]$MaxCpuPct = 20,
    [int]$MinInputIdleMin = 10,
    [int]$MaxWaitMin = 90
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path (Split-Path $PSScriptRoot)
if ($PlanFile -eq '') { $PlanFile = Join-Path $PSScriptRoot 'win_cuda_queue_plan.json' }
$Q = Get-Content -Raw -Path $PlanFile | ConvertFrom-Json
$MaxFileBytes = 400KB

function Expand-Tok([string]$s, [string]$out) {
    $tokPath = $s.StartsWith('{')
    $r = $s.Replace('{repo}', $Repo).Replace('{src}', $SrcDir).Replace('{build}', $BuildDir).Replace('{model}', $Model)
    $r = $r.Replace('{tok}', $Tokenizer).Replace('{py}', $Python).Replace('{cudabin}', $CudaBin).Replace('{out}', $out)
    if ($r.Contains('{ctest}')) {
        $c = Get-Command ctest -ErrorAction SilentlyContinue
        $ct = 'C:\Program Files\CMake\bin\ctest.exe'
        if ($c) { $ct = $c.Source }
        $r = $r.Replace('{ctest}', $ct)
    }
    if ($tokPath) { return $r.Replace('/', '\') }
    return $r
}

function Test-InWindow {
    $t = Get-Date
    $s = [datetime]::ParseExact($WindowStart, 'HH:mm', $null)
    $e = [datetime]::ParseExact($WindowEnd, 'HH:mm', $null)
    $n = $t.TimeOfDay; $a = $s.TimeOfDay; $b = $e.TimeOfDay
    if ($a -le $b) { return ($n -ge $a -and $n -lt $b) }
    return ($n -ge $a -or $n -lt $b)
}

function Select-Items {
    $ids = @()
    if ($Only -ne '') { $ids = $Only.Split(',') }
    $res = @()
    foreach ($it in $Q.items) {
        if ($it.only_with -eq 'WithCtest' -and -not $WithCtest) { continue }
        if ($ids.Count -gt 0 -and ($ids -notcontains $it.id)) { continue }
        $res += $it
    }
    return $res
}

function Exe-State($it, $step) {
    $p = Expand-Tok $step.exe ''
    if (Test-Path $p) { return 'exists' }
    $name = [IO.Path]::GetFileNameWithoutExtension($p)
    if ($Q.built_by_build -contains $name -and $p.StartsWith($BuildDir)) { return 'built by 00-build' }
    if ($step.optional) { return 'optional, skipped when missing' }
    return 'MISSING'
}

if ($Plan) {
    Write-Host ('win_cuda_queue plan: repo ' + $Repo + ' | build ' + $BuildDir)
    Write-Host ('window ' + $WindowStart + '-' + $WindowEnd + ' (-Now ignores it) | idle wait ' + $IdleMin + ' min before every item | GPU <= ' + $MaxGpuPct + '% | CPU <= ' + $MaxCpuPct + '% | stop file ' + $StopFile)
    $tot = 0; $i = 0
    foreach ($it in (Select-Items)) {
        $i++; $tot += [int]$it.est_min + $IdleMin
        Write-Host ('{0,2}. {1} [{2}] est {3} min, needs RAM {4} GB, VRAM {5} GB, timeout {6} min/step - {7}' -f $i, $it.id, $it.group, $it.est_min, $it.ram_gb, $it.vram_gb, $it.timeout_min, $it.title)
        if ($it.kind -eq 'build') {
            Write-Host ('      builds in ' + $BuildDir + ' from ' + $SrcDir + ' (skip if every exe exists and is newer than the source HEAD)')
            foreach ($n in $Q.built_by_build) { Write-Host ('      target ' + $n) }
        }
        foreach ($s in $it.steps) {
            Write-Host ('      step ' + $s.name + ': ' + (Expand-Tok $s.exe '') + ' ' + ((@($s.args) | ForEach-Object { Expand-Tok $_ '' }) -join ' ') + '  [' + (Exe-State $it $s) + ']')
        }
        foreach ($nd in $it.needs) { $p = Expand-Tok $nd ''; $st = 'MISSING'; if (Test-Path $p) { $st = 'exists' }; Write-Host ('      needs ' + $p + ' [' + $st + ']') }
    }
    Write-Host ('estimated total (not measured): ' + [math]::Round($tot / 60.0, 1) + ' h including the idle waits')
    exit 0
}

# ---- from here on the script may run things ----
if (-not $Now -and -not (Test-InWindow)) { Write-Host ('outside the night window ' + $WindowStart + '-' + $WindowEnd + ': nothing started'); exit 0 }
$mutex = New-Object System.Threading.Mutex($false, 'Global\win_cuda_queue_1660ti')
if (-not $mutex.WaitOne(0)) { Write-Host 'another win_cuda_queue is running: exit'; exit 5 }

if ($OutRoot -eq '') {
    $d = Get-Date
    if (-not $Now -and $d.TimeOfDay -lt [datetime]::ParseExact($WindowEnd, 'HH:mm', $null).TimeOfDay) { $d = $d.AddDays(-1) }
    $OutRoot = Join-Path $Repo ('bench\results\' + $d.ToString('yyyy-MM-dd') + '-gtx1660ti-queue')
}
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$LogFile = Join-Path $OutRoot 'queue.log'
function Log([string]$m) { $l = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + ' ' + $m; Add-Content -Path $LogFile -Value $l; Write-Host $l }

Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class QInput {
  [StructLayout(LayoutKind.Sequential)] struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }
  [DllImport("user32.dll")] static extern bool GetLastInputInfo(ref LASTINPUTINFO p);
  public static double IdleSeconds() { LASTINPUTINFO i = new LASTINPUTINFO(); i.cbSize = (uint)Marshal.SizeOf(i);
    if (!GetLastInputInfo(ref i)) return 1e9; return ((uint)Environment.TickCount - i.dwTime) / 1000.0; }
}
'@

function Get-GpuUtil { try { $v = & nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>$null; return [int](($v | Select-Object -First 1).Trim()) } catch { return 100 } }
function Get-VramFreeMB { try { $v = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null; return [int](($v | Select-Object -First 1).Trim()) } catch { return 0 } }
function Get-AvailRamMB { try { return [int](Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes } catch { return [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024) } }
function Get-CpuPct { try { return [int](Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'").PercentProcessorTime } catch { return 100 } }
function Test-Ac {
    $b = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
    if (-not $b) { return $true }
    foreach ($x in @($b)) { if (@(2, 3, 6, 7, 8, 9) -contains [int]$x.BatteryStatus) { return $true } }
    return $false
}
function Get-Blockers {
    # report only; nothing is stopped.  Another agent's build or benchmark, an engine, a compiler.
    $names = @('strata', 'strata-q36', 'strata-poly', 'strata-split', 'ninja', 'cmake', 'cl', 'link', 'nvcc', 'cicc', 'ptxas', 'ctest', 'MSBuild')
    $hit = @()
    foreach ($p in (Get-Process -ErrorAction SilentlyContinue)) {
        if ($names -contains $p.ProcessName -or $p.ProcessName -like 'poly_*') { $hit += $p.ProcessName }
    }
    foreach ($w in (Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue)) {
        if ($w.CommandLine -match 'bench_win|abtest|replay_turns|ttft_bench') { $hit += ('python:' + $w.ProcessId) }
    }
    return $hit
}

function Wait-Idle($it) {
    # returns OK, STOP, WINDOW or BUSY.  Every condition must hold for IdleMin minutes in a row.
    $okSince = $null; $t0 = Get-Date; $why = ''
    while ($true) {
        if (Test-Path $StopFile) { return 'STOP' }
        if (-not $Now -and -not (Test-InWindow)) { return 'WINDOW' }
        if (((Get-Date) - $t0).TotalMinutes -gt $MaxWaitMin) { Log ('BUSY: not idle for ' + $MaxWaitMin + ' min, last reason: ' + $why); return 'BUSY' }
        $why = ''
        $bl = @(Get-Blockers); if ($bl.Count -gt 0) { $why = 'process ' + (($bl | Select-Object -Unique) -join ',') }
        if ($why -eq '') { $g = Get-GpuUtil; if ($g -gt $MaxGpuPct) { $why = 'GPU ' + $g + '%' } }
        if ($why -eq '') { $c = Get-CpuPct; if ($c -gt $MaxCpuPct) { $why = 'CPU ' + $c + '%' } }
        if ($why -eq '') { $r = Get-AvailRamMB; if ($r -lt ([double]$it.ram_gb * 1024)) { $why = 'free RAM ' + $r + ' MB < ' + $it.ram_gb + ' GB' } }
        if ($why -eq '') { $v = Get-VramFreeMB; if ($v -lt ([double]$it.vram_gb * 1024)) { $why = 'free VRAM ' + $v + ' MB < ' + $it.vram_gb + ' GB' } }
        if ($why -eq '' -and -not (Test-Ac)) { $why = 'not on AC power' }
        if ($why -eq '' -and -not $Now) { $ii = [QInput]::IdleSeconds(); if ($ii -lt ($MinInputIdleMin * 60)) { $why = 'owner input ' + [int]$ii + ' s ago' } }
        if ($why -ne '') { if ($okSince) { Log ('idle broken: ' + $why) }; $okSince = $null }
        elseif (-not $okSince) { $okSince = Get-Date }
        elseif (((Get-Date) - $okSince).TotalMinutes -ge $IdleMin) { return 'OK' }
        Start-Sleep -Seconds 20
    }
}

function Quote-Arg([string]$a) { if ($a -match '[\s"]') { return '"' + $a.Replace('"', '\"') + '"' } return $a }

function Stop-OwnTree([int]$id) {
    # only the process this script started, found by id, and its descendants
    & taskkill.exe /PID $id /T /F 2>&1 | Out-Null
}

function Run-Step($it, $step, [string]$dir) {
    $exe = Expand-Tok $step.exe $dir
    if (-not (Test-Path $exe) -and -not ($exe -notmatch '\\')) {
        if ($step.optional) { Log ('SKIP step ' + $step.name + ': ' + $exe + ' does not exist (optional)'); return 0 }
        Log ('FAIL step ' + $step.name + ': ' + $exe + ' does not exist'); return 127
    }
    $argl = @(); foreach ($a in @($step.args)) { $argl += (Quote-Arg (Expand-Tok $a $dir)) }
    $saved = @{}
    $envs = @{}
    if ($step.env) { foreach ($p in $step.env.PSObject.Properties) { $envs[$p.Name] = (Expand-Tok ([string]$p.Value) $dir) } }
    $envs['PATH'] = $CudaBin + ';' + $env:PATH
    foreach ($k in $envs.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k, 'Process'); [Environment]::SetEnvironmentVariable($k, $envs[$k], 'Process') }
    $so = Join-Path $dir ($step.name + '.stdout.txt'); $se = Join-Path $dir ($step.name + '.stderr.txt')
    Log ('START ' + $it.id + '/' + $step.name + ': ' + $exe + ' ' + ($argl -join ' '))
    $rc = 1
    try {
        if ($argl.Count -gt 0) { $pr = Start-Process -FilePath $exe -ArgumentList ($argl -join ' ') -RedirectStandardOutput $so -RedirectStandardError $se -NoNewWindow -PassThru }
        else { $pr = Start-Process -FilePath $exe -RedirectStandardOutput $so -RedirectStandardError $se -NoNewWindow -PassThru }
        $null = $pr.Handle
        if (-not $pr.WaitForExit([int]$it.timeout_min * 60000)) {
            Log ('TIMEOUT ' + $it.id + '/' + $step.name + ' after ' + $it.timeout_min + ' min: stopping own pid ' + $pr.Id)
            Stop-OwnTree $pr.Id
            $rc = 124
        } else { $rc = $pr.ExitCode }
    } catch { Log ('ERROR starting ' + $exe + ': ' + $_.Exception.Message); $rc = 126 }
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k], 'Process') }
    Log ('END ' + $it.id + '/' + $step.name + ' exit ' + $rc)
    return $rc
}

function Trim-Big([string]$dir) {
    foreach ($f in (Get-ChildItem -Path $dir -Recurse -File -ErrorAction SilentlyContinue)) {
        if ($f.Length -gt $MaxFileBytes) {
            if ($f.Name -like '*.stdout.txt' -or $f.Name -like '*.stderr.txt' -or $f.Name -like '*.log') {
                $lines = Get-Content -Path $f.FullName -Tail 400
                Set-Content -Path ($f.FullName + '.tail.txt') -Value $lines
            }
            Log ('SKIP big file ' + $f.FullName + ' (' + $f.Length + ' bytes > 400 KB)')
            Remove-Item -Path $f.FullName -Force
        }
    }
}

function Build-Fresh {
    $head = (& git -C $SrcDir log -1 --format=%ct 2>$null)
    if (-not $head) { return $false }
    $headT = [DateTimeOffset]::FromUnixTimeSeconds([long]$head).LocalDateTime
    foreach ($n in $Q.built_by_build) {
        $p = Join-Path $BuildDir ($n + '.exe')
        if ($n -in @('head_argmax_parity', 'head_gather_parity', 'head_fmt_test', 'q5k_head_test')) { continue }   # optional targets
        if (-not (Test-Path $p)) { return $false }
        if ((Get-Item $p).LastWriteTime -lt $headT) { return $false }
    }
    return $true
}

function Run-Build($it, [string]$dir) {
    if (Build-Fresh) { Log ('SKIP build: every exe exists and is newer than the HEAD of ' + $SrcDir); return 0 }
    $cmd = Join-Path $dir 'build_step.cmd'
    $targets = ($Q.built_by_build -join ' ')
    $lines = @(
        '@echo off',
        ('call "' + $Vcvars + '"'),
        ('set "CUDA_PATH=' + (Split-Path $CudaBin) + '"'),
        'set "PATH=%CUDA_PATH%\bin;%PATH%"',
        ('cd /d "' + $SrcDir + '"'),
        ('cmake -S . -B "' + $BuildDir + '" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared'),
        'if errorlevel 1 exit /b 1',
        'set RC=0',
        ('for %%T in (' + $targets + ') do ( cmake --build "' + $BuildDir + '" -j 4 --target %%T || ( echo BUILD-FAIL %%T & set RC=1 ) )'),
        'exit /b %RC%')
    Set-Content -Path $cmd -Value $lines -Encoding ASCII
    $pseudo = New-Object PSObject -Property @{ id = $it.id; timeout_min = $it.timeout_min }
    $step = New-Object PSObject -Property @{ name = 'build'; exe = 'cmd.exe'; args = @('/c', $cmd); optional = $false; env = $null }
    return (Run-Step $pseudo $step $dir)
}

# ---- main loop ----
Log ('QUEUE START repo ' + $Repo + ' out ' + $OutRoot + ' now=' + [bool]$Now + ' window ' + $WindowStart + '-' + $WindowEnd)
$failed = 0
$exitCode = 0
foreach ($it in (Select-Items)) {
    $dir = Join-Path $OutRoot $it.id
    if ((Test-Path (Join-Path $dir 'exit.txt')) -and -not $Force) { Log ('SKIP ' + $it.id + ': exit.txt exists (done earlier; -Force reruns)'); continue }
    if (Test-Path $StopFile) { Log ('STOP file ' + $StopFile + ' found: leaving before ' + $it.id); break }
    if (-not $Now -and -not (Test-InWindow)) { Log ('window closed: leaving before ' + $it.id); break }
    $w = Wait-Idle $it
    if ($w -eq 'STOP') { Log ('STOP file found while waiting: leaving before ' + $it.id); break }
    if ($w -eq 'WINDOW') { Log ('window closed while waiting: leaving before ' + $it.id); break }
    if ($w -eq 'BUSY') { $exitCode = 4; break }
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Copy-Item -Path $PlanFile -Destination (Join-Path $dir 'plan_used.json') -Force
    Log ('START ITEM ' + $it.id + ' [' + $it.group + '] ' + $it.title)
    $worst = 0
    if ($it.kind -eq 'build') { $worst = Run-Build $it $dir }
    else {
        $missing = $false
        foreach ($nd in $it.needs) { if (-not (Test-Path (Expand-Tok $nd $dir))) { Log ('FAIL ' + $it.id + ': needs ' + (Expand-Tok $nd $dir) + ' which does not exist'); $missing = $true } }
        if ($missing) { $worst = 127 }
        else {
            foreach ($s in $it.steps) {
                if (Test-Path $StopFile) { Log 'STOP file found between steps'; break }
                $rc = Run-Step $it $s $dir
                if ($rc -ne 0 -and $rc -ne 77 -and $rc -ne 3) { $worst = $rc }
            }
        }
    }
    Trim-Big $dir
    Set-Content -Path (Join-Path $dir 'exit.txt') -Value $worst
    Log ('END ITEM ' + $it.id + ' exit ' + $worst)
    if ($worst -ne 0) { $failed++ }
    if ($it.kind -eq 'build' -and $worst -ne 0) { Log 'build failed: later items run only what the build produced (others skip as missing)' }
}
Log ('QUEUE END failed items ' + $failed + ' exit ' + $exitCode)
$mutex.ReleaseMutex()
if ($exitCode -ne 0) { exit $exitCode }
if ($failed -gt 0) { exit 1 }
exit 0
