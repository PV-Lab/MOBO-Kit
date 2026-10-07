#!/bin/sh
# Double-click this to propose the next round (macOS).
#
# It opens a small window: choose the campaign workbook, press "Check workbook",
# then press "Propose R1" (or R2). The proposed conditions are written to a NEW
# file beside the workbook; the workbook itself is never modified.
#
# macOS will not run a .command file until it is marked executable. Once, in
# Terminal, from this folder:
#
#     chmod +x launch_mobo_kit.command

cd "$(dirname "$0")" || exit 1

MOBO_PYTHON=.venv/bin/python
if [ ! -x "$MOBO_PYTHON" ]; then
  MOBO_PYTHON=python3
fi

"$MOBO_PYTHON" -m mobo_kit.launcher "$@"
status=$?

if [ "$status" -ne 0 ]; then
  echo
  echo "The launcher stopped with an error. The workbook was not modified."
  echo
  echo 'If it says "No module named mobo_kit", the environment is not installed'
  echo "yet. From this folder, run:"
  echo
  echo "    python3 -m venv .venv"
  echo "    .venv/bin/python -m pip install -r requirements/dev.txt"
  echo
  echo "Press return to close."
  read -r _
fi

exit "$status"
