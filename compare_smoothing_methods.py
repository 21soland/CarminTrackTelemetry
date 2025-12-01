"""
Compare smoothing method performance for GPS data validation.

This script runs OSM validation with different smoothing methods and displays
a side-by-side comparison of error metrics.

Usage:
    python3 compare_smoothing_methods.py --data-file "GPS Data/Cheswick Laps.txt"
    python3 compare_smoothing_methods.py --data-file "GPS Data/Cheswick Laps.txt" --window 7
    python3 compare_smoothing_methods.py --data-file "GPS Data/Cheswick Laps.txt" --use-existing
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_osm_validation(data_file: Path, smooth_method: str, smooth_window: int, 
                       dist: float, output_dir: Path) -> bool:
    """
    Run OSM validation for a specific smoothing method.
    
    Args:
        data_file: Path to GPS data file
        smooth_method: Smoothing method to test
        smooth_window: Window size for smoothing
        dist: OSM query radius
        output_dir: Output directory
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Running OSM validation with {smooth_method} filter (window={smooth_window})...")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable, "osm.py",
        "--data-file", str(data_file),
        "--smooth-method", smooth_method,
        "--smooth-window", str(smooth_window),
        "--dist", str(dist),
        "--output-dir", str(output_dir)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running validation: {e}")
        print(e.stderr)
        return False


def load_metrics_from_csv(csv_path: Path) -> Optional[Dict[str, float]]:
    """
    Load metrics from a metrics summary CSV file.
    
    Args:
        csv_path: Path to metrics summary CSV
        
    Returns:
        Dictionary of metrics or None if file doesn't exist
    """
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        # Extract smoothed GPS metrics
        metrics = {}
        for _, row in df.iterrows():
            metric_name = row['Metric']
            smoothed_value = row['Smoothed_GPS']
            metrics[metric_name] = float(smoothed_value) if pd.notna(smoothed_value) else np.nan
        return metrics
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def collect_metrics(data_file: Path, methods: List[str], smooth_window: int,
                    dist: float, output_dir: Path, use_existing: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Collect metrics for all smoothing methods.
    
    Args:
        data_file: Path to GPS data file
        methods: List of smoothing methods to test
        smooth_window: Window size for smoothing
        dist: OSM query radius
        output_dir: Output directory
        use_existing: If True, only load existing results, don't run new validations
        
    Returns:
        Dictionary mapping method name to metrics dictionary
    """
    all_metrics = {}
    file_stem = data_file.stem
    
    for method in methods:
        # Check if results already exist
        metrics_path = output_dir / f"{file_stem}_{method}_metrics_summary.csv"
        
        if use_existing and metrics_path.exists():
            print(f"Loading existing results for {method}...")
            metrics = load_metrics_from_csv(metrics_path)
            if metrics:
                all_metrics[method] = metrics
            else:
                print(f"Warning: Could not load metrics for {method}")
        else:
            # Run validation
            if not use_existing:
                success = run_osm_validation(data_file, method, smooth_window, dist, output_dir)
                if success:
                    metrics = load_metrics_from_csv(metrics_path)
                    if metrics:
                        all_metrics[method] = metrics
                else:
                    print(f"Warning: Validation failed for {method}")
            else:
                print(f"Warning: No existing results found for {method}, skipping...")
    
    return all_metrics


def print_comparison_table(all_metrics: Dict[str, Dict[str, float]], 
                          data_file: Path) -> None:
    """
    Print a formatted comparison table of all metrics.
    
    Args:
        all_metrics: Dictionary mapping method name to metrics
        data_file: Path to the data file being analyzed
    """
    if not all_metrics:
        print("No metrics to display!")
        return
    
    print(f"\n{'='*90}")
    print(f"COMPARISON OF SMOOTHING METHODS: {data_file.name}")
    print(f"{'='*90}")
    
    # Define metrics to display (in order of importance)
    metric_names = ['mean_m', 'rms_m', 'median_m', 'std_m', 'p95_m', 'p99_m', 'max_m']
    metric_labels = {
        'mean_m': 'Mean Error [m]',
        'rms_m': 'RMS Error [m]',
        'median_m': 'Median Error [m]',
        'std_m': 'Std Dev [m]',
        'p95_m': '95th %ile [m]',
        'p99_m': '99th %ile [m]',
        'max_m': 'Max Error [m]'
    }
    
    # Build comparison table
    methods = list(all_metrics.keys())
    
    # Print header
    header = f"{'Metric':<20}"
    for method in methods:
        header += f"{method.upper():>15}"
    print(header)
    print("-" * len(header))
    
    # Print each metric
    for metric in metric_names:
        row = f"{metric_labels[metric]:<20}"
        values = []
        for method in methods:
            value = all_metrics[method].get(metric, np.nan)
            values.append(value)
            if np.isnan(value):
                row += f"{'N/A':>15}"
            else:
                row += f"{value:>15.2f}"
        print(row)
        
        # Highlight best (lowest) value
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            best_idx = np.argmin(valid_values)
            best_method = methods[best_idx]
            print(f"  → Best: {best_method.upper()} ({valid_values[best_idx]:.2f} m)")
    
    print(f"{'='*90}\n")


def create_comparison_plot(all_metrics: Dict[str, Dict[str, float]], 
                          data_file: Path, output_path: Path) -> None:
    """
    Create a comparison visualization of metrics.
    
    Args:
        all_metrics: Dictionary mapping method name to metrics
        data_file: Path to the data file being analyzed
        output_path: Path to save the plot
    """
    if not all_metrics:
        print("No metrics to plot!")
        return
    
    methods = list(all_metrics.keys())
    
    # Key metrics to plot
    metrics_to_plot = {
        'mean_m': 'Mean Error [m]',
        'rms_m': 'RMS Error [m]',
        'median_m': 'Median Error [m]',
        'p95_m': '95th Percentile [m]',
        'max_m': 'Max Error [m]'
    }
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot.items()):
        ax = axes[idx]
        
        values = []
        labels = []
        bar_colors = []
        
        for i, method in enumerate(methods):
            value = all_metrics[method].get(metric_key, np.nan)
            if not np.isnan(value):
                values.append(value)
                labels.append(method.upper())
                bar_colors.append(colors[i % len(colors)])
        
        if values:
            bars = ax.bar(labels, values, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Highlight best (lowest) value
            best_idx = np.argmin(values)
            bars[best_idx].set_alpha(1.0)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
            ax.set_title(metric_label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_axisbelow(True)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric_label, fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Smoothing Method Comparison: {data_file.name}', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    plt.close()


def save_comparison_csv(all_metrics: Dict[str, Dict[str, float]], 
                       data_file: Path, output_path: Path) -> None:
    """
    Save comparison metrics to CSV.
    
    Args:
        all_metrics: Dictionary mapping method name to metrics
        data_file: Path to the data file being analyzed
        output_path: Path to save the CSV
    """
    if not all_metrics:
        print("No metrics to save!")
        return
    
    # Get all unique metric keys
    all_metric_keys = set()
    for metrics in all_metrics.values():
        all_metric_keys.update(metrics.keys())
    
    all_metric_keys = sorted(all_metric_keys)
    
    # Build DataFrame
    data = {'Method': list(all_metrics.keys())}
    for metric_key in all_metric_keys:
        data[metric_key] = [all_metrics[method].get(metric_key, np.nan) 
                           for method in all_metrics.keys()]
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved comparison CSV to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare smoothing method performance for GPS validation"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to GPS data file"
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help="Window size for smoothing filter (default: 7)"
    )
    parser.add_argument(
        "--dist",
        type=float,
        default=2000.0,
        help="OSM query radius in meters (default: 2000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="osm_validation",
        help="Output directory for OSM validation results (default: osm_validation)"
    )
    parser.add_argument(
        "--comparison-dir",
        type=str,
        default="smoothing_comparison",
        help="Output directory for comparison plots and CSV (default: smoothing_comparison)"
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Only load existing results, don't run new validations"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["gaussian", "savgol", "ma"],
        choices=["gaussian", "savgol", "ma"],
        help="Smoothing methods to compare (default: all three)"
    )
    
    args = parser.parse_args()
    
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    comparison_dir = Path(args.comparison_dir)
    comparison_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Comparing Smoothing Methods")
    print(f"{'='*70}")
    print(f"Data file: {data_file}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Window size: {args.smooth_window}")
    print(f"OSM validation output: {output_dir}")
    print(f"Comparison output: {comparison_dir}")
    print(f"Use existing results: {args.use_existing}")
    print(f"{'='*70}")
    
    # Collect metrics for all methods
    all_metrics = collect_metrics(
        data_file, 
        args.methods, 
        args.smooth_window,
        args.dist,
        output_dir,
        args.use_existing
    )
    
    if not all_metrics:
        print("\nError: No metrics collected. Check that validations completed successfully.")
        sys.exit(1)
    
    # Print comparison table
    print_comparison_table(all_metrics, data_file)
    
    # Create comparison plot (save to separate comparison directory)
    plot_path = comparison_dir / f"{data_file.stem}_smoothing_comparison.png"
    create_comparison_plot(all_metrics, data_file, plot_path)
    
    # Save comparison CSV (save to separate comparison directory)
    csv_path = comparison_dir / f"{data_file.stem}_smoothing_comparison.csv"
    save_comparison_csv(all_metrics, data_file, csv_path)
    
    print(f"\n{'='*70}")
    print("Comparison Complete!")
    print(f"{'='*70}")
    print(f"Results saved to:")
    print(f"  - Plot: {plot_path}")
    print(f"  - CSV: {csv_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

