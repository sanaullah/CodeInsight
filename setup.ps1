# CodeInsight v3 Setup Script
# Creates a virtual environment and installs dependencies

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CodeInsight v3 Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
    
    # Check Python version (3.10+)
    $versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
    if ($versionMatch) {
        $majorVersion = [int]$matches[1]
        $minorVersion = [int]$matches[2]
        
        if ($majorVersion -lt 3 -or ($majorVersion -eq 3 -and $minorVersion -lt 10)) {
            Write-Host "ERROR: Python 3.10 or higher is required. Found Python $majorVersion.$minorVersion" -ForegroundColor Red
            exit 1
        }
    }
}
catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.10 or higher from https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Check if .venv already exists
if (Test-Path ".venv") {
    Write-Host "Virtual environment '.venv' already exists." -ForegroundColor Yellow
    Write-Host "Skipping virtual environment creation." -ForegroundColor Yellow
    Write-Host ""
}
else {
    Write-Host "Creating virtual environment '.venv'..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "Virtual environment created successfully!" -ForegroundColor Green
    Write-Host ""
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
$activateScript = ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $activateScript)) {
    Write-Host "ERROR: Virtual environment activation script not found" -ForegroundColor Red
    exit 1
}

# Handle PowerShell execution policy
$executionPolicy = Get-ExecutionPolicy
if ($executionPolicy -eq "Restricted") {
    Write-Host "WARNING: PowerShell execution policy is Restricted" -ForegroundColor Yellow
    Write-Host "Attempting to activate virtual environment..." -ForegroundColor Yellow
    Write-Host "If activation fails, run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    Write-Host ""
}

try {
    & $activateScript
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Virtual environment activation may have failed" -ForegroundColor Yellow
        Write-Host "You may need to activate manually: .venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "WARNING: Could not activate virtual environment automatically" -ForegroundColor Yellow
    Write-Host "Please activate manually: .venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host ""
}

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Failed to upgrade pip" -ForegroundColor Yellow
}
else {
    Write-Host "pip upgraded successfully!" -ForegroundColor Green
}
Write-Host ""

# Install requirements
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
if (-not (Test-Path "requirements.txt")) {
    Write-Host "ERROR: requirements.txt not found" -ForegroundColor Red
    exit 1
}

python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
if ($env:VIRTUAL_ENV) {
    Write-Host "1. Virtual Environment:" -ForegroundColor White
    Write-Host "   Active ($env:VIRTUAL_ENV)" -ForegroundColor Green
}
else {
    Write-Host "1. Activate the virtual environment:" -ForegroundColor White
    Write-Host "   .venv\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host "   (Note: If this script was not run with '. .\setup.ps1', the environment is not active in your current shell)" -ForegroundColor DarkGray
}
Write-Host ""
# Copy Config Files
Write-Host "Checking configuration files..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Write-Host "Copying .env.example to .env..." -ForegroundColor Yellow
        Copy-Item ".env.example" ".env"
        Write-Host ".env created successfully!" -ForegroundColor Green
    }
    else {
        Write-Host "WARNING: .env.example not found. Please create .env manually." -ForegroundColor Red
    }
}
else {
    Write-Host ".env already exists. Skipping." -ForegroundColor Yellow
}

if (-not (Test-Path "config.yaml")) {
    if (Test-Path "config.yaml.example") {
        Write-Host "Copying config.yaml.example to config.yaml..." -ForegroundColor Yellow
        Copy-Item "config.yaml.example" "config.yaml"
        Write-Host "config.yaml created successfully!" -ForegroundColor Green
    } 
}
Write-Host ""

# Setup Langfuse (Docker)
Write-Host "Setting up Langfuse..." -ForegroundColor Yellow
if (Get-Command "docker" -ErrorAction SilentlyContinue) {
    if (Test-Path "langfuse/docker-compose.yml") {
        Write-Host "Starting Langfuse services..." -ForegroundColor Yellow
        docker compose -f langfuse/docker-compose.yml up -d
        if ($LASTEXITCODE -ne 0) {
            Write-Host "WARNING: Failed to start Langfuse services. Please check Docker Desktop." -ForegroundColor Red
        }
        else {
            Write-Host "Langfuse services started!" -ForegroundColor Green
        }
    }
    else {
        Write-Host "WARNING: langfuse/docker-compose.yml not found." -ForegroundColor Red
    }
}
else {
    Write-Host "WARNING: Docker is not installed or not in PATH. Skipping Langfuse setup." -ForegroundColor Red
}
Write-Host ""

# Initialize Database
Write-Host "Initializing database..." -ForegroundColor Yellow
$venvPython = ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    if (Test-Path "scripts/init_database.py") {
        & $venvPython scripts/init_database.py
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Database initialization failed." -ForegroundColor Red
            exit 1
        }
        Write-Host "Database initialized successfully!" -ForegroundColor Green
    }
    else {
        Write-Host "WARNING: scripts/init_database.py not found." -ForegroundColor Red
    }
}
else {
    Write-Host "ERROR: Virtual environment python not found at $venvPython" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
if ($env:VIRTUAL_ENV) {
    Write-Host "1. Virtual Environment:" -ForegroundColor White
    Write-Host "   Active ($env:VIRTUAL_ENV)" -ForegroundColor Green
}
else {
    Write-Host "1. Activate the virtual environment:" -ForegroundColor White
    Write-Host "   .venv\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host "   (Note: If this script was not run with '. .\setup.ps1', the environment is not active in your current shell)" -ForegroundColor DarkGray
}
Write-Host ""
Write-Host "2. Start the application:" -ForegroundColor White
Write-Host "   streamlit run ui/app.py" -ForegroundColor Cyan
Write-Host ""
