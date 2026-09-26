@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo .venv belum ada. Jalankan: python -m venv .venv
  pause
  exit /b 1
)
".venv\Scripts\python.exe" main.py
pause
