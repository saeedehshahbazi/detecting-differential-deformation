setlocal EnableDelayedExpansion
@echo off
set start_time=%TIME%
set current_dir=%~dp0

set "fileListPs="
set "fileListPoly="

for %%f in (../*) do (
	set tmp=%%f
    :: if start with "_inside_" then add to list
    if "!tmp:~0,9!"=="_inside_" (
        set "fileListPs=!fileListPs! !tmp!"
    )
)

for %%F in (%fileListPs%) do (
    :: set a new name add "includedPolygons_" instead of "_inside_"
    set "newName=%%F"
    set "newName=!newName:__inside_SpainC_=includedPolygons_!"
    set "tmp=%%F"
    echo Processing files
    echo PsPoint: !tmp!
    echo Polygon: !newName!

    :: call python script
    python gradientCalculationInsidePoints_sepPoly.py !tmp! !newName!
    set exit_code=!errorlevel!
    if not "!exit_code!"=="0" (
        echo The Python script exited with a non-zero status.
        exit /b %exit_code%
    )
)

set end_time=%TIME%
set elapsed=!start_time!-!end_time!
echo "Elapsed time:" !elapsed!
echo Task completed successfully...
