"""
Utility Functions for GPS Track Analysis

This module provides helper functions for data conversion, rounding, and
mathematical operations used throughout the analysis pipeline.
"""

import numpy as np
from typing import List, Optional


def safe_float(value) -> float:
    """
    Safely convert a value to float, returning NaN on failure.
    
    Args:
        value: Value to convert (string, number, etc.).
        
    Returns:
        Float value, or np.nan if conversion fails.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def mean_or_nan(values: List[float]) -> float:
    """
    Compute mean of values, ignoring None and NaN.
    
    Args:
        values: List of numeric values (may contain None or NaN).
        
    Returns:
        Mean of valid values, or np.nan if no valid values exist.
    """
    cleaned = [v for v in values if v is not None and not np.isnan(v)]
    if not cleaned:
        return np.nan
    return float(np.mean(cleaned))


def round_float(value, digits: int = 3) -> Optional[float]:
    """
    Round a float value, handling None, NaN, and Inf.
    
    Args:
        value: Value to round.
        digits: Number of decimal places. Default 3.
        
    Returns:
        Rounded float, or None if value is None, NaN, or Inf.
    """
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return None
    return round(float(value), digits)


def preserve_precision(value, digits: Optional[int] = None) -> Optional[float]:
    """
    Preserve or round precision of a float value.
    
    Args:
        value: Value to process.
        digits: Number of decimal places. If None, preserves original precision.
        
    Returns:
        Float value (rounded if digits specified), or None if value is None or NaN.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if digits is None:
        return float(value)
    return round(float(value), digits)

