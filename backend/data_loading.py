"""
Data Loading and Parsing for GPS Track Analysis

This module handles loading and parsing raw GPS data files into structured
dictionaries organized by time steps and header types.
"""

from pathlib import Path
from typing import Dict, List, Optional
from . import constants
from . import utils


def get_headers(header_file: Path = constants.HEADER_FILE) -> Dict[str, List[str]]:
    """
    Load and parse the header definitions file.
    
    Reads a file where each line contains a header type followed by comma-separated
    field names. Returns a dictionary mapping header types to their field lists.
    
    Args:
        header_file: Path to the headers definition file. Defaults to HEADER_FILE.
        
    Returns:
        Dictionary mapping header type names (e.g., "Fix", "Accel") to lists
        of field names for that header type.
    """
    headers_dict = {}
    
    with header_file.open("r", encoding="utf-8") as file:
        headers = file.readlines()
    
    for header in headers:
        header = header.strip()
        header_list = header.split(',')
        headers_dict[header_list[0]] = header_list[1:]
    
    return headers_dict


def get_header(line: str, headers: List[str]) -> Optional[str]:
    """
    Identify which header type a line belongs to.
    
    Checks if a line starts with any of the known header types.
    
    Args:
        line: The line of text to check.
        headers: List of known header type names.
        
    Returns:
        The matching header type name, or None if no match is found.
    """
    for header in headers:
        if line.startswith(header):
            return header
    return None


def parse_timestep_data(time_step: List[str], headers_dict: Dict[str, List[str]]) -> Dict:
    """
    Parse a single time step's data into a structured dictionary.
    
    Processes all lines in a time step, grouping data by header type and
    handling multiple entries of the same header type.
    
    Args:
        time_step: List of lines belonging to a single time step.
        headers_dict: Dictionary mapping header types to their field names.
        
    Returns:
        Nested dictionary structure: {header_type: {entry_index: {field: value}}}
        Example: {"Fix": {0: {"LatitudeDegrees": "40.123", ...}}, ...}
    """
    headers = list(headers_dict.keys())
    time_step_dict = {}
    
    for line in time_step:
        header_type = get_header(line, headers)
        if not header_type:
            continue
            
        split_line = line.rstrip().split(",")
        
        # Check if there are multiple entries of the same header type
        n = 0
        if header_type in time_step_dict:
            n = len(time_step_dict[header_type].keys())
        else:
            time_step_dict[header_type] = {}
        
        # Add the data for this entry
        time_step_dict[header_type][n] = {}
        
        header_type_keys = headers_dict[header_type]
        for i in range(1, len(header_type_keys)):
            # Note: i-1 because split_line[0] is the header type itself
            time_step_dict[header_type][n][header_type_keys[i-1]] = split_line[i]
    
    return time_step_dict


def load_gps_data(file_path: Path = constants.DEFAULT_DATA_FILE) -> Dict[int, Dict]:
    """
    Load and parse GPS data file into a time-indexed structure.
    
    Splits the raw data file into time steps (delimited by "Fix" lines) and
    parses each time step into a structured dictionary.
    
    Args:
        file_path: Path to the GPS data file. Defaults to DEFAULT_DATA_FILE.
        
    Returns:
        Dictionary mapping time step indices to parsed time step dictionaries.
        Each time step contains header-type-organized data (Fix, Accel, Gyro, etc.).
    """
    header_dict = get_headers(constants.HEADER_FILE)
    data = {}
    
    with Path(file_path).open("r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # Split data into time steps (each time step starts with a "Fix" line)
    time_step_data = []
    cur_data = []
    last_was_fix = True
    
    for line in lines:
        if line.rstrip().startswith('Fix'):
            if not last_was_fix:
                # End of previous time step
                time_step_data.append(cur_data)
                cur_data = []
            last_was_fix = True
        else:
            last_was_fix = False
        cur_data.append(line)
    
    # Add the last time step
    time_step_data.append(cur_data)
    
    # Parse each time step
    for t in range(len(time_step_data)):
        data[t] = parse_timestep_data(time_step_data[t], header_dict)
    
    return data

