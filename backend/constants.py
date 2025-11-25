"""
Constants for GPS Track Analysis

This module defines path constants used throughout the GPS analysis system.
"""

from pathlib import Path

# GPS Data folder is one level up from backend/
DATA_DIR = Path(__file__).parent.parent / "GPS Data"
HEADER_FILE = DATA_DIR / "headers.txt"
DEFAULT_DATA_FILE = DATA_DIR / "Parsable Data.txt"

