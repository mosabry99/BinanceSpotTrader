"""
Neural Pulse Scalper - Installation Helper

This script guides you through the initial setup process for the trading bot.
It performs the following actions:
1. Checks for required system dependencies (Python version, pip).
2. Creates the necessary directory structure for logs, data, and models.
3. Installs the required Python packages from requirements.txt.
4. Sets up the environment configuration file (.env) from the example.

Please run this script from the root directory of the project.
"""

import sys
import os
import subprocess
import shutil
import platform

# --- Helper for Colored Output ---

class Colors:
    """A class to hold ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_color(text, color):
    """Prints text in a specified color."""
    # Disable colors on Windows if colorama is not installed, to avoid printing raw ANSI codes.
    if platform.system() == "Windows":
        os.system('') # Enables ANSI escape characters in Windows 10/11 CMD
    print(f"{color}{text}{Colors.ENDC}")

# --- Installation Steps ---

def check_system_dependencies():
    """
    Verifies that essential system-level dependencies are installed.
    """
    print_color("\n[1/4] Checking system dependencies...", Colors.HEADER)
    all_good = True

    # 1. Check Python version
    min_py_version = (3, 8)
    if sys.version_info >= min_py_version:
        print_color(f"  ✔ Python version {sys.version_info.major}.{sys.version_info.minor} is sufficient (>= 3.8).", Colors.OKGREEN)
    else:
        print_color(f"  ✖ Python version is too old. Please use Python {min_py_version[0]}.{min_py_version[1]} or newer.", Colors.FAIL)
        all_good = False

    # 2. Check if pip is available
    if shutil.which("pip") or shutil.which("pip3"):
        print_color("  ✔ pip is available.", Colors.OKGREEN)
    else:
        print_color("  ✖ pip is not installed. Please install pip for your Python distribution.", Colors.FAIL)
        all_good = False

    if not all_good:
        print_color("\nCritical dependencies are missing. Please resolve the issues above and run the script again.", Colors.FAIL)
        sys.exit(1)
    
    print_color("System dependencies check passed.", Colors.OKGREEN)

def create_directories():
    """
    Creates the necessary directory structure for the bot to operate.
    """
    print_color("\n[2/4] Creating directory structure...", Colors.HEADER)
    
    dirs_to_create = [
        'logs',
        'data',
        'data/historical',
        'models',
        'results',
        'results/backtest'
    ]
    
    for directory in dirs_to_create:
        try:
            os.makedirs(directory, exist_ok=True)
            print_color(f"  ✔ Directory '{directory}/' is ready.", Colors.OKCYAN)
        except OSError as e:
            print_color(f"  ✖ Error creating directory '{directory}/': {e}", Colors.FAIL)
            sys.exit(1)
            
    print_color("Directory structure created successfully.", Colors.OKGREEN)

def install_requirements():
    """
    Installs Python packages from the requirements.txt file.
    """
    print_color("\n[3/4] Installing Python packages...", Colors.HEADER)
    
    requirements_file = 'requirements.txt'
    if not os.path.exists(requirements_file):
        print_color(f"  ✖ '{requirements_file}' not found. Cannot install dependencies.", Colors.FAIL)
        sys.exit(1)

    print_color("This will install all required Python packages from 'requirements.txt'.", Colors.WARNING)
    
    # Ask for user confirmation
    try:
        response = input("  Do you want to proceed? (y/n): ").lower().strip()
    except (EOFError, KeyboardInterrupt):
        print_color("\nInstallation cancelled by user.", Colors.FAIL)
        sys.exit(1)

    if response != 'y':
        print_color("  Skipping package installation.", Colors.WARNING)
        return

    print_color("  Installing packages... This may take a few minutes.", Colors.OKBLUE)
    
    try:
        # Use sys.executable to ensure pip from the correct environment is used
        process = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            capture_output=True,
            text=True,
            check=True
        )
        print_color(process.stdout, Colors.OKCYAN)
        print_color("✔ All packages installed successfully.", Colors.OKGREEN)
    except subprocess.CalledProcessError as e:
        print_color("  ✖ An error occurred during package installation.", Colors.FAIL)
        print_color(e.stderr, Colors.FAIL)
        print_color("\nPlease try running 'pip install -r requirements.txt' manually to diagnose the issue.", Colors.WARNING)
        sys.exit(1)

def setup_env_file():
    """
    Sets up the .env file for storing API keys and secrets.
    """
    print_color("\n[4/4] Setting up environment file...", Colors.HEADER)
    
    env_file = '.env'
    example_env_file = '.env.example'

    if os.path.exists(env_file):
        print_color(f"  ✔ '{env_file}' already exists. Skipping creation.", Colors.OKGREEN)
        print_color("  Please ensure your API keys are correctly set in the file.", Colors.WARNING)
        return

    if not os.path.exists(example_env_file):
        print_color(f"  ✖ '{example_env_file}' not found. Cannot create '{env_file}'.", Colors.FAIL)
        print_color("  Please restore the example file or create '.env' manually.", Colors.WARNING)
        return

    try:
        shutil.copy(example_env_file, env_file)
        print_color(f"  ✔ Successfully created '{env_file}' from '{example_env_file}'.", Colors.OKGREEN)
        print_color("\n" + "="*60, Colors.WARNING + Colors.BOLD)
        print_color("  IMPORTANT: You must now edit the '.env' file!", Colors.WARNING + Colors.BOLD)
        print_color("  Open the file and add your Binance API keys.", Colors.WARNING + Colors.BOLD)
        print_color("="*60 + "\n", Colors.WARNING + Colors.BOLD)
    except Exception as e:
        print_color(f"  ✖ An error occurred while creating '{env_file}': {e}", Colors.FAIL)
        sys.exit(1)

def main():
    """
    Main function to run all installation steps.
    """
    print_color("="*60, Colors.HEADER)
    print_color("  Welcome to the Neural Pulse Scalper Installation  ", Colors.HEADER)
    print_color("="*60, Colors.HEADER)
    
    check_system_dependencies()
    create_directories()
    install_requirements()
    setup_env_file()
    
    print_color("\n🎉 Setup Complete! 🎉", Colors.OKGREEN + Colors.BOLD)
    print_color("You are now ready to use the Neural Pulse Scalper.", Colors.OKGREEN)
    print_color("\nNext Steps:", Colors.HEADER)
    print("  1. " + Colors.WARNING + "Edit the '.env' file with your Binance API keys." + Colors.ENDC)
    print("  2. To start the bot with its graphical interface, run: " + Colors.OKCYAN + "python run.py gui" + Colors.ENDC)
    print("  3. To train the AI model (optional), run: " + Colors.OKCYAN + "python run.py train" + Colors.ENDC)

if __name__ == "__main__":
    main()
