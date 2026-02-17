import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

class VizEngine:
    """
    Unified Visualization Engine for MANIFOLD Evaluation Suites.
    Implements a standard professional scientific style.
    """
    
    COLORS = {
        'primary': '#1f77b4',    # Standard Blue
        'secondary': '#ff7f0e',  # Standard Orange
        'background': '#ffffff',
        'grid': '#e0e0e0',
        'text': '#333333',
        'success': '#2ca02c',    # Standard Green
        'warning': '#d62728',    # Standard Red (for errors/warnings)
        'error': '#d62728'
    }

    @staticmethod
    def apply_style():
        """Apply global matplotlib/seaborn styling."""
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'text.color': VizEngine.COLORS['text'],
            'axes.labelcolor': VizEngine.COLORS['text'],
            'xtick.color': VizEngine.COLORS['text'],
            'ytick.color': VizEngine.COLORS['text'],
            'font.family': 'sans-serif',
            'font.weight': 'normal',
            'figure.facecolor': VizEngine.COLORS['background']
        })

    @staticmethod
    def create_dashboard(title, num_plots=(2, 2), figsize=(18, 14)):
        """Create a stylized figure and axes."""
        VizEngine.apply_style()
        fig, axes = plt.subplots(num_plots[0], num_plots[1], figsize=figsize)
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        fig.suptitle(title, fontsize=24, fontweight='bold', y=0.98, color=VizEngine.COLORS['text'])
        return fig, axes

    @staticmethod
    def save_dashboard(fig, output_path):
        """Save fig to output_path and ensure directory exists."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot_curve(ax, x, y, label, color_key='primary', linewidth=2, moving_average=None):
        """Plot a standard metrics curve with optional smoothing."""
        color = VizEngine.COLORS.get(color_key, color_key)
        
        if moving_average and len(y) > moving_average:
            y_smooth = np.convolve(y, np.ones(moving_average)/moving_average, mode='valid')
            x_smooth = x[moving_average-1:]
            ax.plot(x_smooth, y_smooth, color=color, label=f"{label} (SMA)", linewidth=linewidth)
            ax.plot(x, y, color=color, alpha=0.3, linewidth=1)
        else:
            ax.plot(x, y, color=color, label=label, linewidth=linewidth)
            
        ax.legend(frameon=True)

    @staticmethod
    def plot_heatmap(ax, data, xlabel, ylabel, title, cmap='viridis'):
        """Plot a stylized heatmap (e.g., for NIAH)."""
        sns.heatmap(data, ax=ax, cmap=cmap, annot=True, fmt=".1f", 
                    cbar_kws={'label': 'Recall Success %'},
                    xticklabels=True, yticklabels=True)
        ax.set_title(title, fontweight='bold', fontsize=16)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
