"""
Neural Pulse Scalper - Main Runner Script

This script provides a simple and convenient way to run the different components
of the Neural Pulse Scalper bot from the command line.

Usage:
    - To launch the Streamlit GUI:
      python run.py gui

    - To start the bot in trading mode (headless):
      python run.py trade

    - To begin the AI model training process:
      python run.py train
      
    - To run a backtest of the strategy:
      python run.py backtest
"""

import argparse
import subprocess
import sys
import os

def main():
    """
    Parses command-line arguments and executes the appropriate application mode.
    """
    parser = argparse.ArgumentParser(
        description="Neural Pulse Scalper - Main Runner",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'mode',
        choices=['gui', 'trade', 'train', 'backtest'],
        help="""Choose the mode to run:
- gui:      Launch the Streamlit web interface for monitoring and control.
- trade:    Run the bot in live or paper trading mode (headless).
- train:    Start the AI/ML model training process.
- backtest: Run a backtest of the strategy.
"""
    )

    args = parser.parse_args()

    # Check if the script is being run from the project's root directory
    if not os.path.exists('src') or not os.path.exists('requirements.txt'):
        print("Error: This script must be run from the root directory of the project.")
        print("Please 'cd' into the project's root folder and try again.")
        sys.exit(1)

    command = []
    if args.mode == 'gui':
        print("🚀 Launching Streamlit GUI...")
        print("   Open your browser and navigate to the URL provided.")
        command = [sys.executable, '-m', 'streamlit', 'run', 'src/gui.py']
    elif args.mode == 'trade':
        print("🤖 Starting trading bot in headless mode...")
        print("   Press Ctrl+C to stop the bot.")
        command = [sys.executable, '-m', 'src.main', 'trade']
    elif args.mode == 'train':
        print("🧠 Starting AI model training process...")
        command = [sys.executable, '-m', 'src.main', 'train']
    elif args.mode == 'backtest':
        print("📈 Running strategy backtest...")
        command = [sys.executable, '-m', 'src.main', 'backtest']

    try:
        # The `subprocess.run` command will execute the specified command.
        # `sys.executable` ensures that we use the same Python interpreter
        # (and virtual environment) that is running this script.
        subprocess.run(command, check=True)
    except FileNotFoundError:
        if args.mode == 'gui':
            print("\nError: 'streamlit' command not found.")
            print("Please ensure Streamlit is installed by running: pip install -r requirements.txt")
        else:
            print(f"\nError: Could not find the required script to run mode '{args.mode}'.")
            print("Please ensure the project structure is correct.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nAn error occurred while running the '{args.mode}' mode. The process exited with code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\nProcess for '{args.mode}' mode interrupted by user. Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()
