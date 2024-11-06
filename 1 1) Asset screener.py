# Libraries#############################################################################
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# Page Inizialization####################################################################
Constituents = pd.read_csv(r"Resources/constituents.csv")
SP500_Asset_df = pd.read_csv(r"Resources/sp500_Companies_AdjClose.csv", index_col=0, parse_dates=True)
BenchMark_df = pd.read_csv(r"Resources/Benchmarks_Adj_Close.csv", index_col=0, parse_dates=True)

# Create a dictionary from Constituents with ticker as key and company name as value
constituents_dict = pd.Series(Constituents['Security'].values, index=Constituents['Symbol']).to_dict()

# Create a list to store the assets
if 'Asset_List' not in st.session_state:
    st.session_state['Asset_List'] = []

if 'BenchMark' not in st.session_state:
    st.session_state['BenchMark'] = 'S&P 500'

# User input for Asset List with search by company name
def update_asset_list():
    st.session_state['Asset_List'] = st.session_state['temp_Asset_List']

if 'temp_Asset_List' not in st.session_state:
    st.session_state['temp_Asset_List'] = st.session_state['Asset_List']

# BenchMark Selection##########################################################################
# Sidebar title
st.sidebar.markdown("<h2 style='text-align: center;'>Advanced Asset Selection</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)

# Dictionary with names of benchmarks
benchmark_names = {
    '^FTSE': 'FTSE 100',
    '^NDX': 'NASDAQ 100',
    '^RUT': 'Russell 2000',
    '^NYA': 'NYSE Composite',
    '^GSPC': 'S&P 500'
}

# User input for Benchmark
BenchMark = st.sidebar.selectbox(
    "Benchmark",
    [benchmark_names.get(b, b) for b in BenchMark_df.columns.tolist()],
    index=[benchmark_names.get(b, b) for b in BenchMark_df.columns.tolist()].index(st.session_state['BenchMark'])
)
st.session_state['BenchMark'] = BenchMark

# Reverse lookup to get the ticker symbol from the selected benchmark name
BenchMark = {v: k for k, v in benchmark_names.items()}.get(BenchMark, BenchMark)

st.session_state['BenchMarkTicker'] = BenchMark

# Button to select a random company
if st.sidebar.button('Select Random Company'):
    random_company = Constituents.sample(1)['Symbol'].values[0]
    st.session_state['Asset_List'].append(random_company)
    st.session_state['temp_Asset_List'] = st.session_state['Asset_List']
    update_asset_list()
# Asset Selection##########################################################################
st.title("Asset Selection")

Asset_List = st.multiselect(
    "Select Assets",
    options=list(constituents_dict.keys()),
    format_func=lambda x: f"{x} - {constituents_dict[x]}",
    on_change=update_asset_list,
    key='temp_Asset_List',
    label_visibility='collapsed'
)

# Screening Tool##########################################################################

if len(Asset_List) >= 2:
    st.header("Screening Tool")
    # Create columns
    col1, col2, col5 ,col4, col3,col6 = st.columns([0.9, 0.5, 0.8 ,0.25, 0.4, 1])

    # Display headers
    col1.markdown("**Company name**", unsafe_allow_html=True)
    col2.markdown("**Price**", unsafe_allow_html=True)
    col5.markdown("**52-Week range**", unsafe_allow_html=True)
    col4.markdown("**Beta**", unsafe_allow_html=True)
    col3.markdown("**Mk Cap**", unsafe_allow_html=True)
    col6.markdown("**Monthly price chart**", unsafe_allow_html=True)

    # Display data for selected assets
    for asset in Asset_List:
        company_name = constituents_dict[asset].split()[0]
        current_price = SP500_Asset_df[asset].iloc[-1]
        if len(SP500_Asset_df[asset]) >= 252:
            week_52_range = f"{SP500_Asset_df[asset].rolling(window=252).min().iloc[-1]:.2f} - {SP500_Asset_df[asset].rolling(window=252).max().iloc[-1]:.2f}"
        else:
            week_52_range = f"{SP500_Asset_df[asset].min():.2f} - {SP500_Asset_df[asset].max():.2f}"
        # Calculate the beta of the asset
        asset_returns = SP500_Asset_df[asset].pct_change(fill_method=None).dropna()
        benchmark_returns = BenchMark_df[BenchMark].pct_change(fill_method=None).dropna()
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

    # Find the company with the fewest datapoints among selected assets
    fewest_datapoints_company = SP500_Asset_df[Asset_List].count().idxmin()
    fewest_datapoints_name = constituents_dict[fewest_datapoints_company]

    # Display the company with the fewest datapoints
    st.sidebar.markdown(f"**Company with the fewest datapoints:** {fewest_datapoints_name} ({fewest_datapoints_company})")

    # Compute the amount of years of data based on today
    today = pd.Timestamp('2024-10-18')
    data_start_date = SP500_Asset_df[Asset_List].dropna().index.min()

    years_of_data = (today - data_start_date).days / 365.25

    # Display the amount of years of data
    st.sidebar.markdown(f"**Years of data available:** {years_of_data:.2f} years")
else:
    st.markdown(
        "<div style='text-align: center; color: white; font-size: 20px; background-color: #ff4b4b; padding: 10px; border-radius: 5px; font-weight: bold;'>Select at least 2 assets</div>",
        unsafe_allow_html=True
    )