$Root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$env:PYTHONPATH = "$Root\python;$env:PYTHONPATH"
python -m pytest "$Root\tests" -q @args
