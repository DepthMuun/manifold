"""
Hyperparameter Analysis Tool
============================

Reads the optimization results CSV and generates a report on the best hyperparameters.
"""

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_results(csv_path: str, output_report: str = "optimization_report.md"):
    print(f"Analyzing results from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    # Basic stats
    total_trials = len(df)
    completed_trials = len(df[df["status"] == "COMPLETED"])
    failed_trials = len(df[df["status"] == "ERROR"])
    diverged_trials = len(df[df["status"] == "FAILED"]) # Assuming diverged logic

    print(f"Total Trials: {total_trials}")
    print(f"Completed: {completed_trials}")
    print(f"Failed (Error): {failed_trials}")
    
    if completed_trials == 0:
        print("No completed trials to analyze.")
        return

    # Sort by final loss (ascending)
    # We want low loss, low steps, stable
    
    # Filter only completed
    success_df = df[df["status"] == "COMPLETED"].copy()
    
    # Sort by primary metric: Final Loss
    success_df = success_df.sort_values(by="final_loss")
    
    best_run = success_df.iloc[0]
    
    # Generate Markdown Report
    with open(output_report, "w") as f:
        f.write("# Hyperparameter Optimization Report\n\n")
        f.write(f"**Total Trials:** {total_trials}\n")
        f.write(f"**Completed:** {completed_trials}\n\n")
        
        f.write("## Best Configuration\n")
        f.write("The configuration with the lowest final loss:\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|---|---|\n")
        
        # Exclude metrics from config listing
        metrics = ["trial_id", "status", "final_loss", "initial_loss", "steps", "duration", "diverged", "success", "error"]
        for col in success_df.columns:
            if col not in metrics:
                f.write(f"| {col} | {best_run[col]} |\n")
                
        f.write("\n## Performance Metrics\n")
        f.write(f"- **Final Loss:** {best_run['final_loss']:.4f}\n")
        f.write(f"- **Convergence Steps:** {best_run['steps']}\n")
        f.write(f"- **Duration:** {best_run['duration']:.2f}s\n")
        
        f.write("\n## Top 5 Configurations\n")
        f.write(success_df.head(5).to_markdown(index=False))
        
    print(f"Report generated: {output_report}")
    print("Top 5 configurations:")
    print(success_df.head(5)[["trial_id", "final_loss", "steps"]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze optimization results")
    parser.add_argument("csv_file", type=str, help="Path to optimization_results.csv")
    parser.add_argument("--output", type=str, default="optimization_report.md", help="Output report file")
    
    args = parser.parse_args()
    
    analyze_results(args.csv_file, args.output)
