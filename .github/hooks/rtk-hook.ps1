# rtk-hook.ps1
# VS Code PreToolUse hook: transparently rewrites shell commands with rtk prefix
# RTK reduces LLM token usage by 60-90% by compressing command output.
param()

$raw = [Console]::In.ReadToEnd()
try {
    $data = $raw | ConvertFrom-Json -ErrorAction Stop
} catch {
    Write-Output '{"continue":true}'
    exit 0
}

# Terminal tool names across VS Code and Claude Code
$terminalTools = @('runInTerminal', 'Bash', 'bash', 'run_in_terminal',
                   'execute_command', 'terminal', 'shell')

if ($data.tool_name -notin $terminalTools) {
    Write-Output '{"continue":true}'
    exit 0
}

# Get command string from tool input
$cmd = $null
if ($null -ne $data.tool_input.command) { $cmd = $data.tool_input.command }
elseif ($null -ne $data.tool_input.cmd) { $cmd = $data.tool_input.cmd }

if ([string]::IsNullOrWhiteSpace($cmd)) {
    Write-Output '{"continue":true}'
    exit 0
}

# Skip already-prefixed or complex chained commands
if ($cmd -match '^rtk ' -or $cmd -match '\|' -or $cmd -match '&&' -or $cmd -match ';') {
    Write-Output '{"continue":true}'
    exit 0
}

# Patterns that benefit from rtk prefix
$rtkPatterns = @(
    '^git (status|diff|log|add|commit|push|pull|branch|stash|show)',
    '^cat ',
    '^grep ',
    '^rg ',
    '^ls( |$)',
    '^pytest',
    '^python -m pytest',
    '^docker (ps|images|logs|compose)',
    '^npm test',
    '^yarn test',
    '^cargo (test|build|clippy)',
    '^tsc',
    '^eslint',
    '^pip (list|outdated)',
    '^kubectl (get|logs|describe)'
)

$matched = $rtkPatterns | Where-Object { $cmd -match $_ } | Select-Object -First 1

if ($null -ne $matched) {
    $field = if ($null -ne $data.tool_input.command) { 'command' } else { 'cmd' }
    $updatedInput = [ordered]@{}
    $data.tool_input.PSObject.Properties | ForEach-Object { $updatedInput[$_.Name] = $_.Value }
    $updatedInput[$field] = "rtk $cmd"

    $output = [ordered]@{
        hookSpecificOutput = [ordered]@{
            hookEventName = 'PreToolUse'
            updatedInput  = $updatedInput
        }
    } | ConvertTo-Json -Depth 10 -Compress

    Write-Output $output
} else {
    Write-Output '{"continue":true}'
}
exit 0
