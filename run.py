#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runner script for the Crypto Trading Analysis Application.

This script provides a simple and robust way to launch the Streamlit application.
It ensures that the application is run with the correct Python environment and
provides clear feedback and error handling.

To run the application:
1. Make sure you have installed the required dependencies:
   pip install -r requirements.txt
2. Execute this script from your terminal:
   python run.py
"""

import os
import sys
import subprocess
import logging
import time

# Configure logging for clear console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def find_app_path() -> str:
    """
    Finds the absolute path to the main Streamlit application file (app.py).

    Returns:
        The absolute path to app.py if found, otherwise None.
    """
    # Get the directory where this runner script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, 'app.py')

    if not os.path.exists(app_path):
        logging.error(f"'app.py' not found in the script directory: {script_dir}")
        logging.error("Please ensure 'run.py' and 'app.py' are in the same directory.")
        return None
    return app_path

def run_streamlit_app():
    """
    Launches the Streamlit application using a subprocess.
    This function constructs and executes the command to run the app,
    handling common errors and user interruptions.
    """
    app_path = find_app_path()
    if not app_path:
        sys.exit(1)

    # Use sys.executable to ensure we use the streamlit from the current python env
    command = [sys.executable, '-m', 'streamlit', 'run', app_path]

    logging.info(f"Starting Streamlit application with command: {' '.join(command)}")
    logging.info("The application should open in your default web browser shortly.")
    logging.info("To stop the application, press Ctrl+C in this terminal.")

    try:
        # subprocess.run will block until the Streamlit server is stopped.
        # It's a cleaner way to run and wait for a process to complete.
        subprocess.run(command, check=True)

    except FileNotFoundError:
        logging.error("Error: 'streamlit' command not found.")
        logging.error("It seems Streamlit is not installed or not in your system's PATH.")
        logging.error("Please install the required dependencies by running: pip install -r requirements.txt")
        sys.exit(1)

    except subprocess.CalledProcessError as e:
        # This error is raised if the Streamlit app exits with a non-zero status (i.e., crashes)
        logging.error(f"Streamlit application exited with an error (code {e.returncode}).")
        logging.error("Please check the output above for specific error messages from the application.")
        sys.exit(1)

    except KeyboardInterrupt:
        # Handle the user pressing Ctrl+C
        logging.info("\nStreamlit application stopped by user. Exiting.")
        sys.exit(0)

    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"An unexpected error occurred while trying to launch the application: {e}")
        sys.exit(1)

def main():
    """
    Main function to orchestrate the application launch.
    """
    logging.info("==============================================")
    logging.info("  Crypto Trading Analysis Application Runner  ")
    logging.info("==============================================")
    logging.info("Preparing to launch the application...")

    # Check for requirements.txt to remind the user to install dependencies
    if not os.path.exists('requirements.txt'):
        logging.warning("Warning: 'requirements.txt' not found.")
        logging.warning("Please ensure all dependencies (streamlit, pandas, etc.) are installed.")
    else:
        logging.info("Found 'requirements.txt'. Ensure dependencies are installed via 'pip install -r requirements.txt'")

    time.sleep(1)  # A brief pause for readability

    # Launch the application
    run_streamlit_app()

if __name__ == "__main__":
    main()
