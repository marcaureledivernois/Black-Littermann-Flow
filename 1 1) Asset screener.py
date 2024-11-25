# Libraries#############################################################################
import streamlit as st
import pandas as pd
import numpy as np
from numpy import pi, cos
import matplotlib.pyplot as plt
from pypfopt import risk_models

from Usefull_Functions import plot_bar, calculate_redundancy_score

# Page Inizialization####################################################################
Constituents = pd.read_csv(r"Resources/constituents.csv")
SP500_Asset_df = pd.read_csv(r"Resources/sp500_Companies_AdjClose.csv", index_col=0, parse_dates=True)
BenchMark_df = pd.read_csv(r"Resources/Benchmarks_Adj_Close.csv", index_col=0, parse_dates=True)

constituents_dict = pd.Series(Constituents['Security'].values, index=Constituents['Symbol']).to_dict()

# Session State Inizialization####################################################################

def update_asset_list():
    st.session_state['Asset_List'] = st.session_state['temp_Asset_List']
def update_data_frequency():
    st.session_state['Data_Frequency'] = st.session_state['temp_Data_Frequency']
    
if 'Asset_List' not in st.session_state:
    st.session_state['Asset_List'] = []
if 'temp_Asset_List' not in st.session_state:
    st.session_state['temp_Asset_List'] = st.session_state['Asset_List']
if 'BenchMark' not in st.session_state:
    st.session_state['BenchMark'] = 'S&P 500'
if 'Start_Date' not in st.session_state:
    st.session_state['Start_Date'] = SP500_Asset_df.index[-1] - pd.DateOffset(years=5)
if 'End_Date' not in st.session_state:
    st.session_state['End_Date'] = SP500_Asset_df.index[-1]
if 'Data_Frequency' not in st.session_state:
    st.session_state['Data_Frequency'] = 'Monthly'

# Dictionary for benchmark names
benchmark_names = {
    '^FTSE': 'FTSE 100',
    '^NDX': 'NASDAQ 100',
    '^RUT': 'Russell 2000',
    '^NYA': 'NYSE Composite',
    '^GSPC': 'S&P 500'
}

# Dictionary for Risk Models
risk_model_names = {
    "sample_cov": "Sample Covariance",
    "semicovariance": "Semicovariance",
    "exp_cov": "Exponential Covariance",
    "ledoit_wolf": "Ledoit Wolf",
    "ledoit_wolf_constant_variance": "Ledoit Wolf Constant Variance",
    "ledoit_wolf_single_factor": "Ledoit Wolf Single Factor",
    "ledoit_wolf_constant_correlation": "Ledoit Wolf Constant Correlation",
    "oracle_approximating": "Oracle Approximating"
}
# Dictionary for Porfolio types
portfolio_beta = 1
portfolio_type = (
    'Anti-cyclical' if portfolio_beta < -0.25 else
    'Defensive' if -0.25 <= portfolio_beta < 0.75 else
    'Neutral' if 0.75 <= portfolio_beta <= 1.25 else
    'Aggressive'
)
ranges_Beta = {
    'Anti-cyclical': (-1, -0.25),
    'Defensive': (-0.25, 0.75),
    'Neutral': (0.75, 1.25),
    'Aggressive': (1.25, 2)
}
colors_Beta = {
    'Anti-cyclical': '#ffb8b8',
    'Defensive': '#ff9999',
    'Neutral': '#ff7a7a',
    'Aggressive': '#ff4b4b'
}
# Dictionary for diversification level
diversification_score = 1
diversification_level = (
    'Poor' if diversification_score < 30 else
    'Moderate' if 30 <= diversification_score < 60 else
    'Good' if 60 <= diversification_score < 80 else
    'Excellent'
)
ranges_Div = {
    'Poor': (0, 30),
    'Moderate': (30, 60),
    'Good': (60, 80),
    'Excellent': (80, 100)
}
colors_Div = {
    'Poor': '#ffb8b8',
    'Moderate': '#ff9999',
    'Good': '#ff7a7a',
    'Excellent': '#ff4b4b'
}
# Dictionary for correlation level
correlation_score = 1
correlation_level = (
    'Excellent' if correlation_score < 0.1 else
    'Good' if 0.1 <= correlation_score < 0.3 else
    'Moderate' if 0.3 <= correlation_score < 0.5 else
    'Poor'
)
ranges_C = {
    'Excellent': (0, 30),
    'Good': (30, 60),
    'Modeate': (60, 80),
    'Poor': (80, 100)
}
colors_C = {
    'Poor': '#ffb8b8',
    'Moderate': '#ff9999',
    'Good': '#ff7a7a',
    'Excellent': '#ff4b4b'
}
# Sidebar##########################################################################

st.sidebar.markdown("<h2 style='text-align: center;'>Advanced Settings</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='text-align: left;'>Data Selection Options</h4>", unsafe_allow_html=True)

# Start Date and End Date input
col1, col2 = st.sidebar.columns(2)
with col1:
    Start_Date = st.date_input(
        "Start Date", 
        value=st.session_state['Start_Date'], 
        min_value=SP500_Asset_df.index[0], 
        max_value=st.session_state['End_Date']
    )
with col2:
    End_Date = st.date_input(
        "End Date", 
        value=st.session_state['End_Date'], 
        min_value=Start_Date, 
        max_value=SP500_Asset_df.index[-1]
    )
st.session_state['Start_Date'] = Start_Date
st.session_state['End_Date'] = End_Date

data_frequency = st.sidebar.selectbox(
    "Data Frequency",
    ["Daily", "Monthly", "Annual"],
    index=["Daily", "Monthly", "Annual"].index(st.session_state['Data_Frequency']),
    on_change=update_data_frequency,
    key='temp_Data_Frequency'
)

st.sidebar.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='text-align: left;'>Benchmark Selection</h4>", unsafe_allow_html=True)

# User input for Benchmark
BenchMark = st.sidebar.selectbox(
    "Benchmark",
    [benchmark_names.get(b, b) for b in BenchMark_df.columns.tolist()],
    index=[benchmark_names.get(b, b) for b in BenchMark_df.columns.tolist()].index(st.session_state['BenchMark']),
    label_visibility='collapsed'
)
st.session_state['BenchMark'] = BenchMark
BenchMark = {v: k for k, v in benchmark_names.items()}.get(BenchMark, BenchMark)
st.session_state['BenchMarkTicker'] = BenchMark

st.sidebar.markdown("<h4 style='text-align: left;'>Risk Model Selection</h4>", unsafe_allow_html=True)

# User input for Risk Model
Risk_Model = st.sidebar.selectbox(
    "Risk Model",
    [risk_model_names.get(r, r) for r in risk_model_names.keys()],
    index=[risk_model_names.get(r, r) for r in risk_model_names.keys()].index(st.session_state.get('Risk_Model', "Sample Covariance")),
    label_visibility='collapsed'
)
st.session_state['Risk_Model'] = Risk_Model

# Reverse lookup to get the risk model key from the selected risk model name
Risk_Model = {v: k for k, v in risk_model_names.items()}.get(Risk_Model, "sample_cov")

# Debug Mode Button
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# Asset Selection##############################################################################################
col1, col2 = st.columns([0.8, 0.5])
with col1:
    st.title("Asset Selection")
with col2:
    st.markdown("<div style='text-align: center; width: 100%; padding-top: 22px;'>", unsafe_allow_html=True)
    if st.button('Select Random Company', key='random_company_button', use_container_width=True):
        random_company = Constituents.sample(1)['Symbol'].values[0]
        st.session_state['Asset_List'].append(random_company)
        st.session_state['temp_Asset_List'] = st.session_state['Asset_List']
        update_asset_list()
    st.markdown("</div>", unsafe_allow_html=True)

# User input Asset List
Asset_List = st.multiselect(
    "Select Assets",
    options=list(constituents_dict.keys()),
    format_func=lambda x: f"{x} - {constituents_dict[x]}",
    on_change=update_asset_list,
    key='temp_Asset_List',
    label_visibility='collapsed'
)

# Database Filtering and Data Preparation - Only computation##############################################################################################################
filtered_SP500_Asset_df = SP500_Asset_df.loc[Start_Date.strftime('%Y-%m-%d'):End_Date.strftime('%Y-%m-%d'), Asset_List].dropna(how='any')
filtered_BenchMark_df = BenchMark_df.loc[Start_Date.strftime('%Y-%m-%d'):End_Date.strftime('%Y-%m-%d'), BenchMark].dropna(how='any')

# Combine the filtered dataframes into a single dataframe
Analysis_df = pd.concat([filtered_SP500_Asset_df, filtered_BenchMark_df], axis=1).dropna(how='any')

# Rename the benchmark column for clarity
Analysis_df.rename(columns={BenchMark: 'Benchmark'}, inplace=True)

# Resample the dataframe based on the selected data frequency
if data_frequency == "Daily":
    Analysis_df = Analysis_df.resample('D').last().dropna(how='any')
elif data_frequency == "Monthly":
    Analysis_df = Analysis_df.resample('ME').last().dropna(how='any')
elif data_frequency == "Annual":
    Analysis_df = Analysis_df.resample('YE').last().dropna(how='any')

Analysis_df.index = Analysis_df.index.strftime('%Y-%m-%d').dropna()
Analysis_df.to_csv("Resources/Analysis_df.csv")

# Screening Tool##################################################################################################
try:
    earliest_dates = [SP500_Asset_df[asset].dropna().index[0] for asset in Asset_List]
    earliest_date = max(earliest_dates).strftime('%Y-%m-%d')
except Exception as e:
    earliest_date = Start_Date.strftime('%Y-%m-%d')

if len(Asset_List) >= 2 and Start_Date.strftime('%Y-%m-%d') > earliest_date:
    

    st.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
    # Create columns & Headers
    col1, col2, col5 ,col4, col3,col6 = st.columns([0.9, 0.5, 0.8 ,0.27, 0.4, 1])
    col1.markdown("**Company - Sector**", unsafe_allow_html=True)
    col2.markdown("**Price**", unsafe_allow_html=True)
    col5.markdown("**52-Week range**", unsafe_allow_html=True)
    col4.markdown("**Beta**", unsafe_allow_html=True)
    col3.markdown("**Mk Cap**", unsafe_allow_html=True)
    col6.markdown("**Monthly price chart**", unsafe_allow_html=True)
    
    # Display data for selected assets
    for asset in Asset_List:
        current_price = SP500_Asset_df[asset].iloc[-1]
        sector_abbr = Constituents.loc[Constituents['Symbol'] == asset, 'GICS Sector'].values[0][:3].upper()
        company_name = f"{constituents_dict[asset].split()[0]} - {sector_abbr}"
        if len(SP500_Asset_df[asset]) >= 252:
            week_52_range = f"{SP500_Asset_df[asset].rolling(window=252).min().iloc[-1]:.2f} - {SP500_Asset_df[asset].rolling(window=252).max().iloc[-1]:.2f}"
        else:
            week_52_range = f"{SP500_Asset_df[asset].min():.2f} - {SP500_Asset_df[asset].max():.2f}"
        
        # Calculate the beta of the asset
        asset_returns = Analysis_df[asset].pct_change(fill_method=None).dropna()
        benchmark_returns = Analysis_df['Benchmark'].pct_change(fill_method=None).dropna()
        covariance = asset_returns.cov(benchmark_returns)
        variance = benchmark_returns.var()
        beta = covariance / variance

        #col1.write(company_name)
        yahoo_finance_url = f"https://finance.yahoo.com/quote/{asset}"
        col1.markdown(f"<a href='{yahoo_finance_url}' target='_blank' style='display: block; background-color: #ff4b4b; padding: 5px; border-radius: 5px; text-align: center; color: white; text-decoration: none;'>{company_name}</a>", unsafe_allow_html=True)
        col2.markdown(f"<div style='height: 100%; padding-top: 6px;padding-bottom: 20px; display: flex; align-items: left; justify-content: left;'>${current_price:.2f}</div>", unsafe_allow_html=True)
        col5.markdown(f"<div style='height: 100%; padding-top: 6px;padding-bottom: 20px; display: flex; align-items: left; justify-content: left;'>${week_52_range.replace(' - ', ' - $')}</div>", unsafe_allow_html=True)
        beta_color = '#ff4b4b'
        col4.markdown(f"<div style='height: 100%; padding-top: 6px;padding-bottom: 20px; display: flex; align-items: center; justify-content: center; color: {beta_color};'>{beta:.2f}</div>", unsafe_allow_html=True)
        
        market_cap = Constituents.loc[Constituents['Symbol'] == asset, 'Market Cap'].values[0]
        
        # Format market cap
        if market_cap >= 1_000_000_000_000:
            market_cap_str = f"{market_cap / 1_000_000_000_000:.2f}T"
        elif market_cap >= 1_000_000_000:
            market_cap_str = f"{market_cap / 1_000_000_000:.2f}B"
        elif market_cap >= 1_000_000:
            market_cap_str = f"{market_cap / 1_000_000:.2f}M"
        else:
            market_cap_str = f"{market_cap:.2f}"
        
        col3.markdown(f"<div style='height: 100%; padding-top: 6px;padding-bottom: 20px; display: flex; align-items: centre; justify-content: centre;'>{market_cap_str}</div>", unsafe_allow_html=True)
        # Plot this week's prices
        one_month_ago = SP500_Asset_df.index[-1] - pd.DateOffset(months=1)
        this_week_prices = SP500_Asset_df.loc[one_month_ago:]

        fig, ax = plt.subplots(figsize=(2, 0.22))  # Set fixed width and height
        fig.patch.set_alpha(0.0)  # Make the background transparent
        ax.patch.set_alpha(0.0)  # Make the plot area transparent
        ax.plot(this_week_prices.index, this_week_prices[asset], color='#ff4b4b', alpha=1)  # Change line color to #ff4b4b

        # Add horizontal white line with the average
        avg_price = this_week_prices[asset].mean()
        ax.axhline(avg_price, color='white', linestyle='--', linewidth=1)

        ax.set_xlabel("")  # Remove x label
        ax.set_ylabel("")  # Remove y label
        ax.xaxis.set_visible(False)  # Remove x axis
        ax.spines['top'].set_visible(False)  # Remove top spine
        ax.spines['right'].set_visible(False)  # Remove right spine
        ax.spines['left'].set_visible(True)  # Remove left spine
        ax.spines['bottom'].set_visible(False)  # Remove bottom spine

        # Set spine and y-tick colors to white
        ax.spines['left'].set_color('white')
        ax.yaxis.set_tick_params(colors='white')

        # Set y-ticks to only show max and min
        y_min, y_max = this_week_prices[asset].min(), this_week_prices[asset].max()
        ax.set_yticks([y_min, y_max])
        ax.set_yticklabels([f"${y_min:.1f}", f"${y_max:.1f}"], color='white')

        col6.pyplot(fig)
elif len(Asset_List) < 2:
    st.markdown(
        "<div style='text-align: center; color: white; font-size: 20px; background-color: #ff4b4b; padding: 10px; border-radius: 5px; font-weight: bold;'>Select at least 2 assets</div>",
        unsafe_allow_html=True
    )
else:
    problematic_stocks = [asset for asset in Asset_List if SP500_Asset_df[asset].dropna().index[0].strftime('%Y-%m-%d') >= Start_Date.strftime('%Y-%m-%d')]
    if problematic_stocks:
        earliest_problematic_date = (max(SP500_Asset_df[asset].dropna().index[0] for asset in problematic_stocks) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        st.markdown(
            f"<div style='text-align: center; color: white; font-size: 20px; background-color: #ff4b4b; padding: 10px; border-radius: 5px; font-weight: bold;'>No data available for the selected date range.<br> Problematic stock(s): {', '.join(problematic_stocks)}.<br> Either remove the problematic stocks or <br> reduce the start date to {earliest_problematic_date} to include all stocks.</div>",
            unsafe_allow_html=True
        )

########################################################################################################################################
# Compute the beta of the market portfolio resulted from the selected assets
if len(Asset_List) >= 2 and Start_Date.strftime('%Y-%m-%d') > earliest_date:
    st.header("Summary - The market portfolio")

    # Calculate the Beta Score
    asset_returns = SP500_Asset_df[Asset_List].pct_change(fill_method=None).dropna()
    weights = [1 / len(Asset_List)] * len(Asset_List)
    portfolio_returns = (asset_returns * weights).sum(axis=1)
    benchmark_returns = BenchMark_df[BenchMark].pct_change(fill_method=None).dropna()
    covariance = portfolio_returns.cov(benchmark_returns)
    variance = benchmark_returns.var()
    portfolio_beta = covariance / variance

    # Calculate portfolio diversification score
    sector_counts = Constituents[Constituents['Symbol'].isin(Asset_List)]['GICS Sector'].value_counts()
    num_sectors = len(sector_counts)
    num_stocks = len(Asset_List)
    stock_points = min(50, (num_stocks / 10) * 50)
    max_sector_points = 50
    sector_points = max_sector_points * (1 - (sector_counts.max() / num_stocks))
    diversification_score = stock_points + sector_points

    # Drop the benchmark column from the original dataframe
    Analysis_df_no_benchmark = Analysis_df.drop(columns='Benchmark')

    if Risk_Model == "sample_cov":
        risk_model_matrix = risk_models.sample_cov(Analysis_df_no_benchmark)
    elif Risk_Model == "semicovariance":
        risk_model_matrix = risk_models.semicovariance(Analysis_df_no_benchmark)
    elif Risk_Model == "exp_cov":
        risk_model_matrix = risk_models.exp_cov(Analysis_df_no_benchmark)
    elif Risk_Model == "ledoit_wolf":
        risk_model_matrix = risk_models.CovarianceShrinkage(Analysis_df_no_benchmark).ledoit_wolf()
    elif Risk_Model == "ledoit_wolf_constant_variance":
        risk_model_matrix = risk_models.CovarianceShrinkage(Analysis_df_no_benchmark).ledoit_wolf(shrinkage_target="constant_variance")
    elif Risk_Model == "ledoit_wolf_single_factor":
        risk_model_matrix = risk_models.CovarianceShrinkage(Analysis_df_no_benchmark).ledoit_wolf(shrinkage_target="single_factor")
    elif Risk_Model == "ledoit_wolf_constant_correlation":
        risk_model_matrix = risk_models.CovarianceShrinkage(Analysis_df_no_benchmark).ledoit_wolf(shrinkage_target="constant_correlation")
    elif Risk_Model == "oracle_approximating":
        risk_model_matrix = risk_models.CovarianceShrinkage(Analysis_df_no_benchmark).oracle_approximating()

    st.session_state['risk_model_matrix'] = risk_model_matrix

    # Calculate the correlation Score
    correlation_matrix = risk_model_matrix.corr()
    correlation_score, Pairs = calculate_redundancy_score(correlation_matrix)

    # Start of the formatting 
    col1, col2, col3 = st.columns(3)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold;'><strong>Portfolio type (Beta)</strong></div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
        plot_bar(portfolio_beta, ranges_Beta, colors_Beta)

    with col2:
        st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold;'><strong>Portfolio diversification</strong></div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
        plot_bar(diversification_score, ranges_Div, colors_Div)

    with col3:
        st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold;'><strong>Portfolio redundancy</strong></div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
        plot_bar(correlation_score, ranges_C, colors_C)


    st.header("Tips")
    
    col1, col2, col3 = st.columns(3)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold;'><strong>Portfolio type</strong></div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
        if portfolio_beta < -0.25:
            st.markdown("<div style='text-align: center;'>This portfolio is classified as <strong>Anti-cyclical</strong>. It tends to perform well during economic downturns and may underperform during economic expansions.</div>", unsafe_allow_html=True)
        elif -0.25 <= portfolio_beta < 0.75:
            st.markdown("<div style='text-align: center;'>This portfolio is classified as <strong>Defensive</strong>. It is less sensitive to market movements and tends to provide more stable returns.</div>", unsafe_allow_html=True)
        elif 0.75 <= portfolio_beta <= 1.25:
            st.markdown("<div style='text-align: center;'>This portfolio is classified as <strong>Neutral</strong>. It has a beta close to 1, indicating that it moves in line with the market.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: center;'>This portfolio is classified as <strong>Aggressive</strong>. It tends to outperform during market upswings but may underperform during downturns.</div>", unsafe_allow_html=True)
    with col2:
        if diversification_score < 30:
            st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold;'><strong>Diversify more</strong></div>", unsafe_allow_html=True)
            st.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center;'>This portfolio has <strong>Poor</strong> diversification. Consider adding more stocks and diversifying across different sectors to improve the score.</div>", unsafe_allow_html=True)
        elif 30 <= diversification_score < 60:
            st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold;'><strong>Diversify more</strong></div>", unsafe_allow_html=True)
            st.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center;'>This portfolio has <strong>Moderate</strong> diversification. Adding more stocks and ensuring they are from different sectors can help improve diversification.</div>", unsafe_allow_html=True)
        elif 60 <= diversification_score < 80:
            st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold;'><strong>Diversify more</strong></div>", unsafe_allow_html=True)
            st.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center;'>This portfolio has <strong>Good</strong> diversification. To achieve an excellent score, consider adding a few more stocks from different sectors.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold;'><strong>Good Diversification</strong></div>", unsafe_allow_html=True)
            st.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center;'>This portfolio has <strong>Excellent</strong> diversification. Keep monitoring and maintaining the balance across different sectors.</div>", unsafe_allow_html=True)

    with col3:
        if Pairs:
            st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold;'><strong>Remove high corr. pairs</strong></div>", unsafe_allow_html=True)

        else:
            st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold;'><strong>Low redundancy</strong></div>", unsafe_allow_html=True)

        st.markdown("<hr style='margin: 0; padding: 0;'>", unsafe_allow_html=True)

        if Pairs:
            # Create a single line string with bullet-separated pairs
            pairs_string = "Pairs: " + ", ".join([f"<span style='color: #ff4b4b; font-size: 17px;'>{pair}</span>" for pair in Pairs])
            # Display the pairs on a single line
            st.markdown(
            f"<h3 style='text-align: left; font-size: 16px;'>"
            f"{pairs_string}"
            f"</h3>", 
            unsafe_allow_html=True
            )

        # Displaying correlation score analysis with a single color
        if correlation_score < 30:
            st.markdown("<div style='text-align: center;'>The portfolio is well-diversified with minimal redundancy.</div>", unsafe_allow_html=True)
        elif 30 <= correlation_score < 60:
            st.markdown("<div style='text-align: center;'>Some redundancy, but decent diversification.</div>", unsafe_allow_html=True)
        elif 60 <= correlation_score < 80:
            st.markdown("<div style='text-align: center;'>Noticeable redundancy, room for improvement.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: center;'>High redundancy, poor diversification.</div>", unsafe_allow_html=True)


# Debug Mode#######################################################################################################################################
if debug_mode:
    st.header("Debug Screen")
    st.write("Selected Assets:", Asset_List)
    st.write("Start Date:", Start_Date)
    st.write("End Date:", End_Date)
    st.write("Data Frequency:", data_frequency)
    st.write("Benchmark:", BenchMark)
    st.write("Risk Model:", Risk_Model)
    with st.expander("DataFrame"):
        st.write(Analysis_df)
    with st.expander("Corr Martix"):
        st.write(Analysis_df.drop(columns='Benchmark').corr())