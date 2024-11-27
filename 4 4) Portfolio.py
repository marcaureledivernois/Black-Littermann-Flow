import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

weights = st.session_state['portfolio_weights']
weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True))

#####################################################################################à
# Set Streamlit page title
st.title("Optimized Portfolio")
colors = ['#f2e6d9','#ffb8b8','#ff9999', '#ff7a7a', '#ff4b4b']
colors = colors[::-1]
custom_cmap = LinearSegmentedColormap.from_list("custom_red", colors)

if all(weight >= 0 for weight in weights.values()):
    # If no negative weights, create a pie chart
    with st.expander("View Portfolio Allocation"):
        fig, ax = plt.subplots(figsize=(6, 4))  # Smaller figure size

        labels = list(weights.keys())
        sizes = list(weights.values())
        cmap = plt.get_cmap(custom_cmap)
        norm = plt.Normalize(vmin=0, vmax=len(sizes))

        labels_with_pct = [f'{label}: {size*100:.1f}%' for label, size in zip(labels, sizes)]

        wedges, texts = ax.pie(
            sizes,
            startangle=140,
            colors=cmap(norm(np.arange(len(sizes)))),
            wedgeprops=dict(width=0.3),  # Smaller wedge width
            textprops={'fontsize': 8}  # Smaller font size
        )

        legend = ax.legend(wedges, labels_with_pct, title="Assets:", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(legend.get_title(), fontsize=18, color="#f2e6d9")  # Increase the font size of the title
        plt.setp(legend.get_texts(), color="#f2e6d9")

        plt.tight_layout()

        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        legend.get_frame().set_alpha(0.0)

        st.pyplot(fig)
else:
    # If there are negative weights, create a bar chart
    with st.expander("View Portfolio Allocation"):
        df = pd.DataFrame(list(weights.items()), columns=["Asset", "Weight"])
        fig, ax = plt.subplots(figsize=(6, 4))  # Smaller figure size

        bars = ax.bar(
            df["Asset"], df["Weight"], color=colors[:len(df)], edgecolor="white"
        )
        ax.set_ylabel("Weight", color="white")
        ax.set_xlabel("Asset", color="white")
        ax.set_title("Portfolio Weights", color="white")
        plt.xticks(rotation=45, fontsize=8, color="white")
        plt.yticks(fontsize=8, color="white")
        ax.tick_params(axis="x", colors="white", which="both")
        ax.tick_params(axis="y", colors="white", which="both")

        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y * 100)}%"))

        for bar, weight in zip(bars, df["Weight"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # Center the label horizontally
                bar.get_height(),  # Position the label above the bar
                f"{weight * 100:.1f}%",  # Format as percentage without decimals
                ha="center", va="bottom", fontsize=8, color="white"  # Label styling
            )

        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        plt.tight_layout()

        st.pyplot(fig)

#####################################################################################

st.title("Performance Metrics")

Analysis_df = pd.read_csv('Resources/Analysis_df.csv')

# Separate Asset prices and Benchmark prices
Asset_Prices = Analysis_df.drop(columns=['Benchmark'])
Benchmark_Prices = Analysis_df[['Date', 'Benchmark']]

# Set 'Date' as index for both Asset_Prices and Benchmark_Prices
Asset_Prices.set_index('Date', inplace=True)
Benchmark_Prices.set_index('Date', inplace=True)

with st.expander("View Portfolio Equity curve"):
    fig1, ax1 = plt.subplots(figsize=(6, 4))  # Define ax1
    # Format y-axis ticks to add "x" at the end of each label
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.2f}x'))

    # Calculate the portfolio returns using Asset_Prices
    returns = Asset_Prices.pct_change().dropna()

    # Align weights with the columns of returns (i.e., the assets)
    weight_series = pd.Series(weights).reindex(returns.columns).fillna(0)  # Ensure correct alignment and handle missing weights

    # Calculate the portfolio returns
    portfolio_returns = returns.dot(weight_series)  # Direct dot product of returns and aligned weights
    equity_curve = (1 + portfolio_returns).cumprod()

    # Calculate equity curve with max equity
    max_equity = equity_curve.max()
    max_equity_date = equity_curve.idxmax()  # This returns a timestamp (date)

    # Plot the equity curve
    equity_curve.plot(ax=ax1, label="Portfolio Equity Curve", color='#FF6666')

    # Plot max equity point (ensure date format is correctly interpreted on x-axis)
    ax1.scatter(equity_curve.index.get_loc(max_equity_date), max_equity, color='red', marker='o', label="Max Equity")

    # Calculate and plot the benchmark equity curve
    benchmark_returns = Benchmark_Prices['Benchmark'].pct_change().dropna()
    benchmark_equity_curve = (1 + benchmark_returns).cumprod()
    benchmark_equity_curve.plot(ax=ax1, label="Benchmark Equity Curve", linestyle='--', color='#4682B4')

    # Calculate max equity point for the benchmark
    benchmark_max_equity = benchmark_equity_curve.max()
    benchmark_max_equity_date = benchmark_equity_curve.idxmax()

    # Plot max equity point for the benchmark (ensure date format is correctly interpreted on x-axis)
    ax1.scatter(benchmark_equity_curve.index.get_loc(benchmark_max_equity_date), benchmark_max_equity, color='blue', marker='o', label="Benchmark Max Equity")

    # Rotate x-axis labels for better readability
    ax1.tick_params(axis='x', rotation=15)
    ax1.xaxis.label.set_color("white")
    ax1.yaxis.label.set_color("white")

    ax1.tick_params(axis='x', colors="white")
    ax1.tick_params(axis='y', colors="white")

    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['right'].set_color('white')

    # Remove the "Date" label from the x-axis
    ax1.set_xlabel("")

    # Set background transparency
    fig1.patch.set_alpha(0.0)
    ax1.patch.set_alpha(0.0)

    # Set legend background transparency and move it to the top left
    legend = ax1.legend(loc='upper left', fontsize=7)
    plt.setp(legend.get_texts(), color="#f2e6d9")
    legend.get_frame().set_alpha(0.0)

    st.pyplot(fig1)

with st.expander("View Portfolio Drawdown"):
    fig2, ax2 = plt.subplots(figsize=(6, 4))  # Define ax2

    # Calculate drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max

    # Plot the drawdown
    drawdown.plot(ax=ax2, label="Portfolio Drawdown", color='#FF6666')

    # Calculate and plot the benchmark drawdown
    benchmark_rolling_max = benchmark_equity_curve.cummax()
    benchmark_drawdown = (benchmark_equity_curve - benchmark_rolling_max) / benchmark_rolling_max
    benchmark_drawdown.plot(ax=ax2, label="Benchmark Drawdown", linestyle='--', color='#4682B4')

    # Find and plot the max drawdown point
    max_drawdown = drawdown.min()
    max_drawdown_date = drawdown.idxmin()
    ax2.scatter(drawdown.index.get_loc(max_drawdown_date), max_drawdown, color='red', marker='o', label="Max Drawdown")

    # Find and plot the max drawdown point for the benchmark
    benchmark_max_drawdown = benchmark_drawdown.min()
    benchmark_max_drawdown_date = benchmark_drawdown.idxmin()
    ax2.scatter(benchmark_drawdown.index.get_loc(benchmark_max_drawdown_date), benchmark_max_drawdown, color='blue', marker='o', label="Benchmark Max Drawdown")

    # Rotate x-axis labels for better readability
    ax2.tick_params(axis='x', rotation=15)

    # Remove the "Date" label from the x-axis
    ax2.set_xlabel("")

    # Format y-axis as percentages
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax2.tick_params(axis='y', colors='white')
    ax2.tick_params(axis='x', colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax2.spines['right'].set_color('white')

    # Set y-axis max to 0
    ax2.set_ylim(top=0)

    # Set background transparency
    fig2.patch.set_alpha(0.0)
    ax2.patch.set_alpha(0.0)

    # Set legend background transparency and move it to the bottom left
    legend = ax2.legend(loc='best', fontsize=7)
    plt.setp(legend.get_texts(), color='white')
    legend.get_frame().set_alpha(0.0)

    st.pyplot(fig2)


#####################################################################################

# Calculate performance metrics for the benchmark
returns_BNK = Benchmark_Prices.pct_change().dropna()
equity_curve_BNK = (1 + returns_BNK).cumprod()
cumulative_return_BNK = equity_curve_BNK.iloc[-1] - 1
cumulative_return_BNK = cumulative_return_BNK.iloc[0]
annualized_return_BNK = np.mean(returns_BNK) * 252
annualized_volatility_BNK = np.std(returns_BNK) * np.sqrt(252)
annualized_volatility_BNK = annualized_volatility_BNK.iloc[0]

# Benchmark drawdown and max equity
peak_BNK = equity_curve_BNK.cummax()
drawdown_BNK = (equity_curve_BNK - peak_BNK) / peak_BNK
max_dd_BNK = drawdown_BNK.min().iloc[0]
max_equity_BNK = equity_curve_BNK.max().iloc[0]

# Calculate performance metrics for the portfolio
returns = Asset_Prices.pct_change().dropna()
weight_series = pd.Series(weights).reindex(returns.columns)
portfolio_returns = returns.dot(weight_series)
equity_curve = (1 + portfolio_returns).cumprod()

cumulative_return = equity_curve.iloc[-1] - 1
annualized_return = np.mean(portfolio_returns) * 252
annualized_volatility = np.std(portfolio_returns) * np.sqrt(252)

peak = equity_curve.cummax()
drawdown = (equity_curve - peak) / peak
max_dd = drawdown.min()
max_equity = equity_curve.max()

# Data for the histogram
metrics = ['Cumulative Return', 'Annualized Return', 'Annualized Volatility', 'Max Equity', 'Max Drawdown']
portfolio_values = [cumulative_return, annualized_return, annualized_volatility, max_equity, max_dd]
benchmark_values = [cumulative_return_BNK, annualized_return_BNK, annualized_volatility_BNK, max_equity_BNK, max_dd_BNK]

with st.expander("View Performance Metrics"):
    fig3, ax3 = plt.subplots(figsize=(8, 6))  # Define ax3

    # Define the bar width
    bar_width = 0.25

    # Define the positions of the bars
    index = np.arange(len(metrics))

    # Plot the bars for the portfolio and benchmark without colormap
    bars1 = ax3.bar(index, portfolio_values, bar_width, label='Portfolio', color='#FF6666')
    bars2 = ax3.bar(index + bar_width, benchmark_values, bar_width, label='Benchmark', color='#4682B4')

    # Add labels, title, and legend
    ax3.set_xlabel('Metrics', color='white')
    ax3.set_ylabel('Values', color='white')
    ax3.set_xticks(index + bar_width / 2)
    ax3.set_xticklabels(metrics, rotation=15, color='white')
    ax3.legend()

    # Set background transparency
    fig3.patch.set_alpha(0.0)
    ax3.patch.set_alpha(0.0)

    # Set legend background transparency and color to white
    legend = ax3.legend(loc='best', fontsize=11)
    legend.get_frame().set_alpha(0.0)
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    # Set tick parameters color to white
    ax3.tick_params(axis='y', colors='white')
    ax3.tick_params(axis='x', colors='white')

    # Set the frame color to white
    for spine in ax3.spines.values():
        spine.set_edgecolor('white')

    # Annotate the values on top of the bars for both portfolio and benchmark
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.2f}',  # You can change the format as needed (e.g., .2f or .2%)
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', color='white')

    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.2f}',  # You can change the format as needed
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', color='white')

    st.pyplot(fig3)

#####################################################################################

# Calculate Sharpe Ratio for the portfolio and benchmark
risk_free_rate = 0.02  # Example risk-free rate

sharpe_ratio_portfolio = (annualized_return - risk_free_rate) / annualized_volatility
sharpe_ratio_benchmark = (annualized_return_BNK - risk_free_rate) / annualized_volatility_BNK

# Calculate Sortino Ratio for the portfolio and benchmark
downside_risk_portfolio = np.std(portfolio_returns[portfolio_returns < 0]) * np.sqrt(252)
sortino_ratio_portfolio = (annualized_return - risk_free_rate) / downside_risk_portfolio

downside_risk_benchmark = np.std(returns_BNK[returns_BNK < 0]) * np.sqrt(252)
sortino_ratio_benchmark = (annualized_return_BNK - risk_free_rate) / downside_risk_benchmark
sortino_ratio_benchmark = sortino_ratio_benchmark.iloc[0]

# Calculate Calmar Ratio for the portfolio and benchmark
calmar_ratio_portfolio = annualized_return / abs(max_dd)
calmar_ratio_benchmark = annualized_return_BNK / abs(max_dd_BNK)

# Data for the ratios
ratios = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
portfolio_ratios = [sharpe_ratio_portfolio, sortino_ratio_portfolio, calmar_ratio_portfolio]
benchmark_ratios = [sharpe_ratio_benchmark, sortino_ratio_benchmark, calmar_ratio_benchmark]

with st.expander("View Performance Ratios"):
    fig4, ax4 = plt.subplots(figsize=(8, 6))  # Define ax4

    # Define the bar width
    bar_width = 0.35

    # Define the positions of the bars
    index = np.arange(len(ratios))

    # Plot the bars for the portfolio and benchmark without colormap
    bars1 = ax4.bar(index, portfolio_ratios, bar_width, label='Portfolio', color='#FF6666')
    bars2 = ax4.bar(index + bar_width, benchmark_ratios, bar_width, label='Benchmark', color='#4682B4')

    # Add labels, title, and legend
    ax4.set_xlabel('Ratios', color='white')
    ax4.set_ylabel('Values', color='white')
    ax4.set_xticks(index + bar_width / 2)
    ax4.set_xticklabels(ratios, rotation=0, color='white')
    ax4.legend()

    # Set background transparency
    fig4.patch.set_alpha(0.0)
    ax4.patch.set_alpha(0.0)

    # Set legend background transparency and color to white
    legend = ax4.legend(loc='best', fontsize=11)
    legend.get_frame().set_alpha(0.0)
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    # Set tick parameters color to white
    ax4.tick_params(axis='y', colors='white')
    ax4.tick_params(axis='x', colors='white')

    ax4.set_xlabel("")

    # Set the frame color to white
    for spine in ax4.spines.values():
        spine.set_edgecolor('white')

    # Annotate the values on top of the bars for both portfolio and benchmark
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}',  # You can change the format as needed (e.g., .2f or .2%)
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', color='white')

    for bar in bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}',  # You can change the format as needed
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', color='white')

    st.pyplot(fig4)