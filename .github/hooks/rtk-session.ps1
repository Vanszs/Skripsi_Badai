# rtk-session.ps1
# VS Code SessionStart hook: injects RTK context into every agent session
param()

$rtkVersion = (rtk --version 2>&1) -replace 'rtk ', ''
$msg = "RTK $rtkVersion active (Rust Token Killer). Prefix commands with rtk for 60-90% token savings. Examples: rtk git status, rtk read file.py, rtk grep pattern ., rtk pytest, rtk docker ps, rtk ls. RTK passes unknown commands through unchanged - always safe to use. Run: rtk gain to see cumulative savings."

$out = [ordered]@{
    hookSpecificOutput = [ordered]@{
        hookEventName   = 'SessionStart'
        additionalContext = $msg
    }
} | ConvertTo-Json -Compress

Write-Output $out
exit 0
