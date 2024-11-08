import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd

def plot_bar(value, ranges, colors):
    """
    Plots a horizontal bar with color-coded ranges and a value indicator line.

    Args:
        value (float): The score to indicate on the bar (e.g., a correlation score between 0 and 100).
        ranges (dict): A dictionary where keys are labels and values are tuples indicating start and end of the range.
                       Example: {'Poor': (0, 30), 'Moderate': (30, 60), 'Good': (60, 80), 'Excellent': (80, 100)}
        colors (dict): A dictionary where keys match range labels and values are color codes for each range.
                       Example: {'Poor': '#ff9999', 'Moderate': '#ff6666', 'Good': '#ff3333', 'Excellent': '#ff4b4b'}
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_alpha(0)  # Make the background transparent

    # Plot each range as a separate bar
    for level, (start, end) in ranges.items():
        ax.barh(0, end - start, left=start, color=colors.get(level, '#000000'), edgecolor='none')

    # Add the value indicator (a vertical line at the specified value)
    ax.axvline(value, color='white', linestyle='-', linewidth=6)  # Thicker line

    # Set the limits of the x-axis
    ax.set_xlim(min(start for start, end in ranges.values()), max(end for start, end in ranges.values()))

    # Remove y-axis ticks and labels
    ax.set_yticks([])

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Remove title and x label
    ax.set_title('')
    ax.set_xlabel('')

    # Set x-ticks and labels based on the ranges
    x_ticks = [start for start, end in ranges.values()] + [max(end for start, end in ranges.values())]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{x:.2f}' for x in x_ticks], color='white', fontweight='bold', fontsize=22)

    # Remove the minor ticks
    ax.tick_params(axis='x', which='both', length=0)

    # Show the plot with transparent background
    st.pyplot(fig, transparent=True)

    # Determine the correlation level based on the value
    correlation_level = next((level for level, (start, end) in ranges.items() if start <= value <= end), "Unknown")

    # Display the correlation level as a centered, styled label
    st.markdown(
        f"<div style='text-align: center; color: white; font-size: 20px; background-color: {colors.get(correlation_level, '#ff4b4b')}; "
        f"padding: 10px; border-radius: 5px; font-weight: bold; margin-top: 0px;'>{correlation_level}</div>",
        unsafe_allow_html=True
    )

################################################################################################################
def plot_cumulative_returns(Analysis_df):
    # Calculate cumulative returns
    cumulative_returns = (1 + Analysis_df.pct_change()).cumprod() - 1

    # Plot cumulative returns
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots()

        # Plot cumulative returns
        cumulative_returns.plot(ax=ax)

        # Add horizontal grey lines
        ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=0.5)

        # Set labels and title with white color
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Cumulative Return', color='white')
        ax.title.set_color('white')

        # Set tick parameters with white color
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Set legend background color to transparent and position it at the top left
        legend = ax.legend(loc='upper left')
        legend.get_frame().set_alpha(0)

        # Hide the figure
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        # Collapse the graph to show only when expanded
        with st.expander("Cumulative Returns Graph"):
            st.pyplot(fig)
