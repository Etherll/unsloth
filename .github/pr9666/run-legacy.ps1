# SPDX-License-Identifier: AGPL-3.0-only
$ErrorActionPreference='Stop'
foreach($phase in @('existing','upgrade','rollback','re-upgrade')) {
  $env:PR9666_PHASE=$phase
  $old=$phase -in @('existing','rollback')
  $env:PR9666_LEGACY=if($old){'1'}else{'0'}
  $env:PR9666_FRONTEND=if($old){(Resolve-Path '../baseline/studio/frontend').Path}else{(Resolve-Path 'studio/frontend').Path}
  $binary=if($old){'../native-binary/unsloth-studio-base.exe'}else{'studio/src-tauri/target/debug/unsloth-studio.exe'}
  $server=$null;$app=$null
  try {
    $server=Start-Process node -ArgumentList '.codex/pr9666/serve-harness.mjs' -WindowStyle Hidden -PassThru -RedirectStandardOutput "legacy-$phase-vite.log" -RedirectStandardError "legacy-$phase-vite.err"
    $ready=$false
    for($i=0;$i -lt 30;$i++){try{$null=Invoke-WebRequest http://localhost:5173/review.html -TimeoutSec 2;$ready=$true;break}catch{Start-Sleep -Seconds 1}}
    if(-not $ready){throw 'Legacy harness did not start'}
    $app=Start-Process $binary -WindowStyle Hidden -PassThru -RedirectStandardOutput "legacy-$phase-app.log" -RedirectStandardError "legacy-$phase-app.err"
    node .codex/pr9666/legacy-probe.mjs
    if($LASTEXITCODE){throw "Legacy phase failed: $phase"}
  } finally {
    if($app){Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue}
    if($server){Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue}
    foreach($kind in @('app.log','app.err','vite.log','vite.err')) {
      $file="legacy-$phase-$kind"
      if(Test-Path $file){
        $value=(Get-Content $file -Raw) -replace '(?im)^.*(?:password|authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|cookie|secret|bootstrap).*$','[sensitive line redacted]'
        $value=$value -replace 'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+','[JWT redacted]'
        $value.Replace($env:USERPROFILE,'<profile>').Replace((Get-Location).Path,'<target>') | Set-Content "native-evidence/$file.sanitized.txt"
      }
    }
    Start-Sleep -Seconds 3
  }
}
