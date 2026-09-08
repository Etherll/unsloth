# SPDX-License-Identifier: AGPL-3.0-only
$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Force native-evidence | Out-Null
. .codex/pr9666/read-debug-environment.ps1
function Protect-Log([string]$value) {
  if ($null -eq $value) { return '' }
  $value=$value -replace '(?im)^.*(?:password|authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|cookie|secret|bootstrap).*$','[sensitive line redacted]'
  $value=$value -replace 'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+','[JWT redacted]'
  $value=$value.Replace($env:USERPROFILE,'<profile>').Replace((Get-Location).Path,'<target>')
  return $value
}
Add-Type @'
using System;
using System.Text;
using System.Collections.Generic;
using System.Runtime.InteropServices;
public static class NativeWindows {
 public delegate bool EnumProc(IntPtr h,IntPtr p);
 [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc callback,IntPtr data);
 [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h,out uint pid);
 [DllImport("user32.dll",CharSet=CharSet.Unicode)] static extern int GetWindowText(IntPtr h,StringBuilder s,int n);
 [DllImport("user32.dll",CharSet=CharSet.Unicode)] static extern int GetClassName(IntPtr h,StringBuilder s,int n);
 [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr h);
 public static string[] ForProcess(uint owner) {
  var rows=new List<string>();
  EnumWindows((h,p)=>{uint pid;GetWindowThreadProcessId(h,out pid);if(pid==owner){var t=new StringBuilder(512);var c=new StringBuilder(512);GetWindowText(h,t,512);GetClassName(h,c,512);rows.Add(h.ToInt64()+" | "+IsWindowVisible(h)+" | "+c+" | "+t);}return true;},IntPtr.Zero);
  return rows.ToArray();
 }
}
'@
function Snapshot-App([string]$label) {
  $all=@(Get-CimInstance Win32_Process)
  $owned=@([uint32]$app.Id)
  do {
    $new=@($all | Where-Object {$_.ParentProcessId -in $owned -and $_.ProcessId -notin $owned} | Select-Object -ExpandProperty ProcessId)
    $owned+=$new
  } while($new.Count)
  $rows=@($all | Where-Object {$_.ProcessId -in $owned -or $_.Name -eq 'msedgewebview2.exe'} | ForEach-Object {@{pid=$_.ProcessId;parent=$_.ParentProcessId;name=$_.Name;command=(Protect-Log $_.CommandLine)}})
  $app.Refresh()
  $flag=if(-not $app.HasExited){[DebugEnvironment]::ReadFlag($app.Id)}else{'<exited>'}
  @{label=$label;inheritedDebugArguments=$flag} | ConvertTo-Json | Set-Content "native-evidence/child-environment-$label.json"
  $modules=@();$threads=@();$windows=@()
  if(-not $app.HasExited) {
    try { $modules=@($app.Modules | ForEach-Object {$_.ModuleName}) } catch { $modules=@('module enumeration unavailable') }
    try { $threads=@($app.Threads | ForEach-Object {@{id=$_.Id;state=[string]$_.ThreadState}}) } catch { $threads=@('thread enumeration unavailable') }
    $windows=@([NativeWindows]::ForProcess($app.Id) | ForEach-Object {Protect-Log $_})
  }
  $ports=@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object {$_.OwningProcess -in $owned -or $_.LocalPort -eq 19266} | Select-Object LocalAddress,LocalPort,OwningProcess)
  @{label=$label;time=[DateTime]::UtcNow.ToString('o');appExited=$app.HasExited;exitCode=$(if($app.HasExited){$app.ExitCode}else{$null});processes=$rows;modules=$modules;threads=$threads;windows=$windows;listeners=$ports} | ConvertTo-Json -Depth 8 | Set-Content "native-evidence/snapshot-$label.json"
}
$runtimeRoots=@("${env:ProgramFiles(x86)}/Microsoft/EdgeWebView/Application", "$env:LOCALAPPDATA/Microsoft/EdgeWebView/Application")
$versions=@($runtimeRoots | Where-Object {Test-Path -LiteralPath $_} | ForEach-Object {Get-ChildItem -LiteralPath $_ -Directory | Select-Object -ExpandProperty Name})
@{runtimeVersions=$versions;sessionId=(Get-Process -Id $PID).SessionId;target=(git rev-parse HEAD);debugArguments='--remote-debugging-port=19266';binary=(Get-FileHash studio/src-tauri/target/debug/unsloth-studio.exe -Algorithm SHA256).Hash} | ConvertTo-Json | Set-Content native-evidence/environment.json
$env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS='--remote-debugging-port=19266'
$server=Start-Process node -ArgumentList '.codex/pr9666/serve-harness.mjs' -WindowStyle Hidden -PassThru -RedirectStandardOutput vite.log -RedirectStandardError vite.err
$app=$null
try {
  $ready=$false
  for($attempt=0;$attempt -lt 60;$attempt++){try{$null=Invoke-WebRequest http://localhost:5173/review.html;$ready=$true;break}catch{Start-Sleep -Seconds 1}}
  if(-not $ready){throw 'Harness HTTP endpoint failed to start'}
  $registry='HKCU:\Software\Policies\Microsoft\Edge\WebView2\AdditionalBrowserArguments'
  foreach($mode in @('explicit-env','registry-only')) {
    if($mode -eq 'registry-only') {
      Remove-Item Env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS -ErrorAction SilentlyContinue
      New-Item -Path $registry -Force | Out-Null
      New-ItemProperty -Path $registry -Name 'unsloth-studio.exe' -PropertyType String -Value '--remote-debugging-port=19266' -Force | Out-Null
      $childEnv=@{WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=$null}
    } else {$childEnv=@{WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS='--remote-debugging-port=19266'}}
    $app=Start-Process './studio/src-tauri/target/debug/unsloth-studio.exe' -Environment $childEnv -WindowStyle Hidden -PassThru -RedirectStandardOutput app.log -RedirectStandardError app.err
    Start-Sleep -Seconds 5
    Snapshot-App "$mode-5s"
    $cdp=$false
    for($i=0;$i -lt 30;$i++) {try{$version=Invoke-RestMethod http://127.0.0.1:19266/json/version;$cdp=$true;break}catch{Start-Sleep -Seconds 1}}
    Snapshot-App "$mode-final"
    @{mode=$mode;cdp=$cdp} | ConvertTo-Json | Set-Content "native-evidence/result-$mode.json"
    if($cdp){break}
    Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
  }
  node .codex/pr9666/probe.mjs
  if($LASTEXITCODE){throw 'Native probe failed'}
} finally {
  if($registry -and (Test-Path $registry)){Remove-ItemProperty -Path $registry -Name 'unsloth-studio.exe' -ErrorAction SilentlyContinue}
  if($app){try{Snapshot-App 'final'}catch{Protect-Log $_.Exception.Message | Set-Content native-evidence/snapshot-error.txt}finally{Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue}}
  Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
  foreach($log in @('app.log','app.err','vite.log','vite.err')) {
    if(Test-Path -LiteralPath $log){Protect-Log (Get-Content -LiteralPath $log -Raw) | Set-Content "native-evidence/$log.sanitized.txt"}
  }
}
