@echo off
rem Double-click this to propose the next round.
rem
rem It opens a small window: choose the campaign workbook, press "Check
rem workbook", then press "Propose R1" (or R2). The proposed conditions are
rem written to a NEW file beside the workbook; the workbook itself is never
rem modified.
rem
rem If the window does not appear, the message left in this console says why.

setlocal
cd /d "%~dp0"

set "MOBO_PYTHON=.venv\Scripts\python.exe"
if not exist "%MOBO_PYTHON%" set "MOBO_PYTHON=python"

"%MOBO_PYTHON%" -m mobo_kit.launcher %*

if errorlevel 1 (
  echo.
  echo The launcher stopped with an error. The workbook was not modified.
  echo.
  echo If it says "No module named mobo_kit", the environment is not installed
  echo yet. From this folder, run:
  echo.
  echo     py -3.12 -m venv .venv
  echo     .venv\Scripts\python -m pip install -r requirements\dev.txt
  echo.
  pause
)

endlocal
