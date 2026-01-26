@echo off
REM CodeInsight Setup Script (Batch Wrapper)
REM Calls the PowerShell setup script for easy double-click execution

echo ========================================
echo CodeInsight Setup
echo ========================================
echo.

REM Check if PowerShell is available
where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PowerShell is not available
    echo.
    echo Please run the Python setup script instead:
    echo   python setup.py
    echo.
    pause
    exit /b 1
)

REM Check if setup.ps1 exists
if not exist "setup.ps1" (
    echo ERROR: setup.ps1 not found
    echo.
    echo Please run the Python setup script instead:
    echo   python setup.py
    echo.
    pause
    exit /b 1
)

REM Run PowerShell script
echo Running PowerShell setup script...
echo.

powershell.exe -ExecutionPolicy Bypass -File "setup.ps1"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Setup failed. Please check the error messages above.
    echo.
    echo Alternative: Run the Python setup script:
    echo   python setup.py
    echo.
    pause
    exit /b 1
)

echo.
pause
