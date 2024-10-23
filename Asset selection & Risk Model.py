##Libraries###############################################################################################
# Data handling libraries
import pandas as pd
import numpy as np
from datetime import datetime

# Graphics libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Optimization Libraries
from pypfopt import risk_models
import streamlit as st

##Asset Selection###############################################################################################

# Databases Initialization
Constituents = pd.read_csv(r"Resources\constituents.csv")
SP500_Asset_df = pd.read_csv(r"Resources\sp500_Companies_AdjClose.csv", index_col=0, parse_dates=True)
BenchMark_df = pd.read_csv(r"Resources\Benchmarks_Adj_Close.csv", index_col=0, parse_dates=True)

# Initialize session state
if 'Asset_List' not in st.session_state:
    st.session_state['Asset_List'] = []
if 'Start_Date' not in st.session_state:
    st.session_state['Start_Date'] = SP500_Asset_df.index[-1] - pd.DateOffset(years=5)
if 'End_Date' not in st.session_state:
    st.session_state['End_Date'] = SP500_Asset_df.index[-1]
if 'BenchMark' not in st.session_state:
    st.session_state['BenchMark'] = 'S&P 500'

st.title("Asset Selection")

# Create a dictionary from Constituents with ticker as key and company name as value
constituents_dict = pd.Series(Constituents['Security'].values, index=Constituents['Symbol']).to_dict()

# User input for Asset List with search by company name
def update_asset_list():
    st.session_state['Asset_List'] = st.session_state['temp_Asset_List']

if 'temp_Asset_List' not in st.session_state:
    st.session_state['temp_Asset_List'] = st.session_state['Asset_List']

Asset_List = st.multiselect(
    "Select Assets",
    options=list(constituents_dict.keys()),
    format_func=lambda x: f"{x} - {constituents_dict[x]}",
    on_change=update_asset_list,
    key='temp_Asset_List',
    label_visibility='collapsed'
)

##Sidebar###############################################################################################

# Sidebar title
st.sidebar.markdown("<h2 style='text-align: center;'>Advanced Settings</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
# Section title for Data Selection options
st.sidebar.markdown("<h4 style='text-align: left;'>Data Selection Options</h4>", unsafe_allow_html=True)

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

# User input for Start Date and End Date
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

# User input for Data Frequency
def update_data_frequency():
    st.session_state['Data_Frequency'] = st.session_state['temp_Data_Frequency']

if 'Data_Frequency' not in st.session_state:
    st.session_state['Data_Frequency'] = 'Monthly'

data_frequency = st.sidebar.selectbox(
    "Data Frequency",
    ["Daily", "Monthly", "Annual"],
    index=["Daily", "Monthly", "Annual"].index(st.session_state['Data_Frequency']),
    on_change=update_data_frequency,
    key='temp_Data_Frequency'
)

st.sidebar.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='text-align: left;'>Risk Model Selection</h4>", unsafe_allow_html=True)

# User input for Risk Model
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

#style##############################################################################################################
# Output Header
st.header("Output Data Selection")
#Resulting Dataframe##############################################################################################################

# Filter the dataframes based on the selected dates
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

# Ensure the date is in yyyy-mm-dd format
Analysis_df.index = Analysis_df.index.strftime('%Y-%m-%d').dropna()

# Save the Analysis dataframe to a CSV file
Analysis_df.to_csv("Resources\Analysis_df.csv")
# Display the Analysis dataframe

# Collapse the dataframe to show only the first few rows
with st.expander("DataFrame"):
    st.write(Analysis_df)

#Cumulative returns##############################################################################################################

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

#Risk Model##############################################################################################################
st.header("Output Risk Model & Correlation Matrixes")

# Exclude the benchmark column from the Analysis dataframe
Analysis_df_no_benchmark = Analysis_df.drop(columns=['Benchmark'])

if not Analysis_df_no_benchmark.empty and len(Analysis_df_no_benchmark.columns) > 1:
    # Calculate the risk model matrix based on the user selection
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
        
    # Save the risk model matrix to session state for use in other pages
    st.session_state['risk_model_matrix'] = risk_model_matrix

    # Plot the risk model matrix
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots()
        
        sns.heatmap(risk_model_matrix, annot=True, fmt=".2f", cmap="magma", ax=ax)
        
        # Set labels and title with white color
        ax.set_xlabel('Assets', color='white')
        ax.set_ylabel('Assets', color='white')
        ax.title.set_color('white')
        
        # Set tick parameters with white color
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        # Set the figure background to transparent
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
    # Collapse the graph to show only when expanded
    with st.expander("Risk Model Covariance Matrix Heatmap"):
        st.pyplot(fig)

    # Calculate the correlation matrix
    correlation_matrix = Analysis_df_no_benchmark.corr()

    # Plot the correlation matrix
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots()
        
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="magma", ax=ax)
        
        # Set labels and title with white color
        ax.set_xlabel('Assets', color='white')
        ax.set_ylabel('Assets', color='white')
        ax.title.set_color('white')
        
        # Set tick parameters with white color
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        # Set the figure background to transparent
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
    # Collapse the graph to show only when expanded
    with st.expander("Correlation Matrix Heatmap"):
        st.pyplot(fig)
else:
    st.write("Not enough data to display the risk model and correlation matrix.")
