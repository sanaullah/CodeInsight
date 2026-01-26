#!/usr/bin/env python3
"""
CodeInsight v3 Setup Script
Creates a virtual environment and installs dependencies
Cross-platform: Works on Windows, Linux, and macOS
"""

import sys
import subprocess
import platform
from pathlib import Path
import shutil
import os

# Colors for terminal output (cross-platform compatible)
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

def print_colored(message: str, color: str = Colors.WHITE) -> None:
    """Print colored message if terminal supports it, otherwise plain text."""
    if sys.stdout.isatty() and platform.system() != "Windows":
        print(f"{color}{message}{Colors.RESET}")
    else:
        print(message)

def check_python_version() -> bool:
    """Check if Python version is 3.10 or higher."""
    if sys.version_info < (3, 10):
        print_colored(
            f"ERROR: Python 3.10 or higher is required. Found Python {sys.version_info.major}.{sys.version_info.minor}",
            Colors.RED
        )
        return False
    print_colored(f"Found: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}", Colors.GREEN)
    return True

def create_venv(venv_path: Path) -> bool:
    """Create virtual environment if it doesn't exist."""
    if venv_path.exists():
        print_colored("Virtual environment '.venv' already exists.", Colors.YELLOW)
        print_colored("Skipping virtual environment creation.", Colors.YELLOW)
        print()
        return True
    
    print_colored("Creating virtual environment '.venv'...", Colors.YELLOW)
    try:
        import venv
        venv.create(venv_path, with_pip=True)
        print_colored("Virtual environment created successfully!", Colors.GREEN)
        print()
        return True
    except Exception as e:
        print_colored(f"ERROR: Failed to create virtual environment: {e}", Colors.RED)
        return False

def get_pip_command(venv_path: Path) -> list:
    """Get the pip command for the virtual environment."""
    if platform.system() == "Windows":
        pip_exe = venv_path / "Scripts" / "pip.exe"
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        pip_exe = venv_path / "bin" / "pip"
        python_exe = venv_path / "bin" / "python"
    
    # Use python -m pip for better compatibility
    return [str(python_exe), "-m", "pip"]

def upgrade_pip(venv_path: Path) -> bool:
    """Upgrade pip in the virtual environment."""
    print_colored("Upgrading pip...", Colors.YELLOW)
    pip_cmd = get_pip_command(venv_path)
    
    try:
        result = subprocess.run(
            pip_cmd + ["install", "--upgrade", "pip", "--quiet"],
            check=True,
            capture_output=True,
            text=True
        )
        print_colored("pip upgraded successfully!", Colors.GREEN)
        print()
        return True
    except subprocess.CalledProcessError as e:
        print_colored("WARNING: Failed to upgrade pip", Colors.YELLOW)
        print(f"Error: {e.stderr}")
        print()
        return False

def install_requirements(venv_path: Path) -> bool:
    """Install requirements from requirements.txt."""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print_colored("ERROR: requirements.txt not found", Colors.RED)
        return False
    
    print_colored("Installing dependencies from requirements.txt...", Colors.YELLOW)
    pip_cmd = get_pip_command(venv_path)
    
    try:
        result = subprocess.run(
            pip_cmd + ["install", "-r", str(requirements_file)],
            check=True
        )
        print_colored("Dependencies installed successfully!", Colors.GREEN)
        print()
        return True
    except subprocess.CalledProcessError as e:
        print_colored("ERROR: Failed to install dependencies", Colors.RED)
        print(f"Error: {e.stderr if hasattr(e, 'stderr') else str(e)}")
        return False

        return "source .venv/bin/activate"

def setup_configs() -> None:
    """Copy configuration files if they don't exist."""
    print_colored("Checking configuration files...", Colors.YELLOW)
    
    # .env
    if not Path(".env").exists():
        if Path(".env.example").exists():
            print_colored("Copying .env.example to .env...", Colors.YELLOW)
            shutil.copy(".env.example", ".env")
            print_colored(".env created successfully!", Colors.GREEN)
        else:
            print_colored("WARNING: .env.example not found. Please create .env manually.", Colors.RED)
    else:
        print_colored(".env already exists. Skipping.", Colors.YELLOW)
    
    # config.yaml
    if not Path("config.yaml").exists():
        if Path("config.yaml.example").exists():
            print_colored("Copying config.yaml.example to config.yaml...", Colors.YELLOW)
            shutil.copy("config.yaml.example", "config.yaml")
            print_colored("config.yaml created successfully!", Colors.GREEN)
    print()

def setup_docker() -> None:
    """Start Langfuse services using Docker."""
    print_colored("Setting up Langfuse...", Colors.YELLOW)
    
    if shutil.which("docker") is None:
        print_colored("WARNING: Docker is not installed or not in PATH. Skipping Langfuse setup.", Colors.RED)
        print()
        return

    docker_compose_file = Path("langfuse/docker-compose.yml")
    if not docker_compose_file.exists():
        print_colored("WARNING: langfuse/docker-compose.yml not found.", Colors.RED)
        print()
        return

    print_colored("Starting Langfuse services...", Colors.YELLOW)
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(docker_compose_file), "up", "-d"],
            check=True
        )
        print_colored("Langfuse services started!", Colors.GREEN)
    except subprocess.CalledProcessError:
        print_colored("WARNING: Failed to start Langfuse services. Please check Docker Desktop.", Colors.RED)
    print()

def init_database(venv_path: Path) -> bool:
    """Initialize the database using the virtual environment's Python."""
    print_colored("Initializing database...", Colors.YELLOW)
    
    if platform.system() == "Windows":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"
    
    if not python_exe.exists():
        print_colored(f"ERROR: Virtual environment python not found at {python_exe}", Colors.RED)
        return False
        
    init_script = Path("scripts/init_database.py")
    if not init_script.exists():
        print_colored("WARNING: scripts/init_database.py not found.", Colors.RED)
        return True # Non-fatal if script is missing, just warn
        
    try:
        subprocess.run(
            [str(python_exe), str(init_script)],
            check=True
        )
        print_colored("Database initialized successfully!", Colors.GREEN)
        print()
        return True
    except subprocess.CalledProcessError:
        print_colored("ERROR: Database initialization failed.", Colors.RED)
        return False

def main() -> int:
    """Main setup function."""
    print_colored("========================================", Colors.CYAN)
    print_colored("CodeInsight v3 Setup", Colors.CYAN)
    print_colored("========================================", Colors.CYAN)
    print()
    
    # Check Python version
    print_colored("Checking Python installation...", Colors.YELLOW)
    if not check_python_version():
        return 1
    print()
    
    # Create virtual environment
    venv_path = Path(".venv")
    if not create_venv(venv_path):
        return 1
    
    # Upgrade pip
    if not upgrade_pip(venv_path):
        # Non-fatal, continue anyway
        pass
    
    # Install requirements
    if not install_requirements(venv_path):
        return 1
    
    # Setup Configs
    setup_configs()
    
    # Setup Docker
    setup_docker()
    
    # Init Database
    if not init_database(venv_path):
        return 1

    # Success message
    print_colored("========================================", Colors.CYAN)
    print_colored("Setup completed successfully!", Colors.GREEN)
    print_colored("========================================", Colors.CYAN)
    print()
    
    activation_cmd = get_activation_command()
    
    print_colored("Next steps:", Colors.YELLOW)
    
    # Check if venv is active
    current_venv = os.environ.get("VIRTUAL_ENV")
    if current_venv:
        print_colored("1. Virtual Environment:", Colors.WHITE)
        print_colored(f"   Active ({current_venv})", Colors.GREEN)
    else:
        print_colored("1. Activate the virtual environment:", Colors.WHITE)
        print_colored(f"   {activation_cmd}", Colors.CYAN)
        print_colored("   (Note: The environment is not active in your current shell)", Colors.WHITE)  # DarkGray not available in standard ANSI easily without more code, White is fine.
    
    print()
    print_colored("2. Start the application:", Colors.WHITE)
    print_colored("   streamlit run ui/app.py", Colors.CYAN)
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
