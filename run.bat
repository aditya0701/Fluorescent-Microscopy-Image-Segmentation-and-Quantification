@echo off
REM BoutonViewer launcher for Windows.
REM Run this file from the BoutonViewer directory after activating
REM the correct conda environment, for example:
REM   conda activate bouton_viewer
REM   run.bat

cd /d "%~dp0"

REM Ensure pyvistaqt uses the PyQt6 binding rather than auto-detecting
REM an older one that may also be present in the environment.
set QT_API=pyqt6

python main.py

pause
