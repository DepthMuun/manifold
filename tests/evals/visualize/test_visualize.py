#!/usr/bin/env python3
"""
Test script for VisualizeEvaluator
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from visualize_evaluator import VisualizeEvaluator

def test_visualize_evaluator():
    """Test the VisualizeEvaluator with sample data"""
    
    print("Testing VisualizeEvaluator...")
    
    # Create evaluator
    evaluator = VisualizeEvaluator(output_dir="visualization_outputs/test")
    
    # Sample evaluation results
    sample_results = {
        'total_tasks': 100,
        'successful_tasks': 85,
        'total_time': 120.5,
        'avg_task_time': 1.205,
        'metrics': {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85,
            'latency_ms': 45.2,
            'throughput_samples_per_sec': 150.0,
            'memory_usage_mb': 512.0,
            'energy_consumption_j': 25.0
        }
    }
    
    print("Generating visualizations...")
    
    # Generate visualizations
    visualizations = evaluator.generate_evaluation_visualizations(sample_results)
    
    print(f"Generated {len(visualizations)} visualizations:")
    for viz_type, viz_path in visualizations.items():
        print(f"  - {viz_type}: {viz_path}")
    
    # Create evaluation report
    report = evaluator.create_evaluation_report(sample_results, visualizations)
    
    # Save report
    report_path = "visualization_outputs/test/evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_path}")
    print("\nTest completed successfully!")
    
    return visualizations, report

if __name__ == "__main__":
    test_visualize_evaluator()