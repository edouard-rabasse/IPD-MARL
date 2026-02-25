"""Centralized plotting style configuration for IPD-MARL."""

import matplotlib.pyplot as plt
import seaborn as sns


def set_style() -> None:
    """Apply the project's standard plotting style.

    Uses seaborn's 'paper' context and 'darkgrid' style, with a custom color
    palette and font settings for publication-quality figures.
    """
    sns.set_theme(context="paper", style="darkgrid", font="sans-serif", font_scale=1.2)

    # Customise matplotlib params further if needed
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "lines.linewidth": 2,
        }
    )

    ## set colors palette to tab20
    sns.set_palette("tab20")
