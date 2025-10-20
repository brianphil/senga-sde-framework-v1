@echo off
REM Senga SDE Demo Launcher - Windows Batch Version
REM Simple version that works on all Windows systems

echo ========================================
echo   Senga SDE Demo Launcher
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python found

REM Create/activate virtual environment
@REM if not exist "venv\" (
@REM     echo [SETUP] Creating virtual environment...
@REM     python -m venv venv
@REM     call venv\Scripts\activate.bat
@REM     python -m pip install --upgrade pip
@REM     pip install -r requirements.txt
@REM     echo [OK] Virtual environment created
@REM ) else (
@REM     echo [OK] Virtual environment found
@REM     call venv\Scripts\activate.bat
@REM )

REM Initialize databases if needed
if not exist "data\senga_config.db" (
    echo [SETUP] Initializing databases...
    python scripts\initialize_system.py
    echo [OK] Databases initialized
) else (
    echo [OK] Databases found
)

REM Start API in background
echo [STARTING] Launching FastAPI backend...
start "Senga API" cmd /k "python src\api\main.py"

REM Wait for API to be ready
echo [WAIT] Waiting for API to start (15 seconds)...
timeout /t 15 /nobreak >nul

REM Start Streamlit
echo [STARTING] Launching Streamlit demo...
echo.
echo ========================================
echo   Demo URLs:
echo   - Streamlit: http://localhost:8501
echo   - API: http://localhost:8000
echo   - API Docs: http://localhost:8000/docs
echo ========================================
echo.
echo Press Ctrl+C in this window to stop the demo
echo.

streamlit run streamlit_demo.py

REM Cleanup happens when user closes the window