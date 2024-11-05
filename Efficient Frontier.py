import streamlit as st
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
import numpy as np
import pandas as pd
from copy import deepcopy

returns = st.session_state['returns'] 
risk_model_matrix = st.session_state['risk_model_matrix']

# Callback function to update session state
def update_session_state(key, value):
    st.session_state[key] = value

# Sidebar title
st.sidebar.markdown("<h2 style='text-align: center;'>Advanced Settings</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='text-align: left;'>Constraints</h4>", unsafe_allow_html=True)

# Checkbox to allow short selling
allow_short_selling = st.sidebar.checkbox(
    'Allow Short Selling', 
    value=st.session_state.get('allow_short_selling', False),
    on_change=update_session_state,
    args=('allow_short_selling', not st.session_state.get('allow_short_selling', False))
)

# Determine weight bounds based on short selling allowance
weight_bounds = (-1, 1) if st.session_state.get('allow_short_selling', False) else (0, 1)

# Checkbox to allow sector constraints
allow_sector_constraints = st.sidebar.checkbox(
    'Allow Sector Constraints', 
    value=st.session_state.get('allow_sector_constraints', False),
    on_change=update_session_state,
    args=('allow_sector_constraints', not st.session_state.get('allow_sector_constraints', False))
)

st.sidebar.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='text-align: left;'>Objective</h4>", unsafe_allow_html=True)

# Checkbox to allow custom constraints
custom_constraints = st.sidebar.checkbox(
    'Custom Objective', 
    value=st.session_state.get('custom_constraints', False),
    on_change=update_session_state,
    args=('custom_constraints', not st.session_state.get('custom_constraints', False))
)
#Main Page############################################################################################################
# Create a sector mapper using the constituents DataFrame
if allow_sector_constraints:
    st.title("Sector constraints")

    constituents = pd.read_csv('Resources//constituents.csv')
    sector_mapper = dict(zip(constituents['Symbol'], constituents['GICS Sector']))

    # Filter the sector mapper with the asset list
    asset_list = returns.index.tolist()
    filtered_sector_mapper = {asset: sector for asset, sector in sector_mapper.items() if asset in asset_list}

    # Initialize sector constraints if not in session state
    if 'sector_lower' not in st.session_state:
        st.session_state['sector_lower'] = {}
    if 'sector_upper' not in st.session_state:
        st.session_state['sector_upper'] = {}

    # Function to update sector constraints
    def update_sector_constraints(sector, lower, upper):
        st.session_state['sector_lower'][sector] = lower
        st.session_state['sector_upper'][sector] = upper

    with st.expander("Show Sector Constraints"):
        # User input for sector constraints
        for sector in set(filtered_sector_mapper.values()):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**{sector}**")
                tickers = [ticker for ticker, sec in filtered_sector_mapper.items() if sec == sector]
                st.markdown(f"Tickers: {', '.join(tickers)}")
            with col2:
                lower = st.number_input(f"Min {sector}", min_value=0.0, max_value=1.0, step=0.05, value=st.session_state['sector_lower'].get(sector, 0.0), key=f"lower_{sector}")
            with col3:
                upper = st.number_input(f"Max {sector}", min_value=0.0, max_value=1.0, step=0.05, value=st.session_state['sector_upper'].get(sector, 1.0), key=f"upper_{sector}")
            update_sector_constraints(sector, lower, upper)
            if sector != list(set(filtered_sector_mapper.values()))[-1]:
                st.markdown("<hr>", unsafe_allow_html=True)  # Add a line between each row except the last one

        sector_lower = st.session_state['sector_lower']
        sector_upper = st.session_state['sector_upper']
#############################################################################################################
st.title("Objective")
# Dropdown menu for objective selection
objective_options = ["Max Sharpe", "Min Volatility", "Max Quadratic Utility", "Max Return"]
if custom_constraints:
    objective_options = ["Target Return", "Target Variance"]

if custom_constraints:
    col1, col2 = st.columns(2)
    with col1:
        selected_objective = st.selectbox("Select Objective", objective_options, key="selected_objective")
    with col2:
        target_value = st.number_input(f"Input {selected_objective}", min_value=0.0, step=0.05, key="target_value")
else:
    selected_objective = st.selectbox("Select Objective", objective_options, key="selected_objective")

#############################################################################################################
st.title("Efficient Frontier")

if allow_sector_constraints:
    ef = EfficientFrontier(expected_returns=returns, cov_matrix=risk_model_matrix, weight_bounds=weight_bounds)
    ef.add_sector_constraints(sector_mapper, st.session_state['sector_lower'], st.session_state['sector_upper'])
else:
    ef = EfficientFrontier(expected_returns=returns, cov_matrix=risk_model_matrix, weight_bounds=weight_bounds)

# Plot efficient frontier within an expandable box
with st.expander("Show Efficient Frontier"):
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Manually plot the efficient frontier
        mus = np.linspace(-0.01, 0.35, 100)
        frontier = []
        for mu in mus:
            ef.efficient_return(mu)
            ret, std, _ = ef.portfolio_performance()
            frontier.append((std, ret))
        frontier = np.array(frontier)
        ax.plot(frontier[:, 0], frontier[:, 1], linestyle='--', color='cyan', label='Efficient Frontier')
        
        # Calculate and plot tangency, min vol, max ret and max quadratic utility portfolios
        if allow_sector_constraints:
            ef_max_sharpe = EfficientFrontier(expected_returns=returns, cov_matrix=risk_model_matrix, weight_bounds=weight_bounds)
            ef_min_vol = EfficientFrontier(expected_returns=returns, cov_matrix=risk_model_matrix, weight_bounds=weight_bounds)
            ef_max_ret = EfficientFrontier(expected_returns=returns, cov_matrix=risk_model_matrix, weight_bounds=weight_bounds)
            ef_max_quadratic = EfficientFrontier(expected_returns=returns, cov_matrix=risk_model_matrix, weight_bounds=weight_bounds)

            ef_max_sharpe.add_sector_constraints(sector_mapper, st.session_state['sector_lower'], st.session_state['sector_upper'])
            ef_min_vol.add_sector_constraints(sector_mapper, st.session_state['sector_lower'], st.session_state['sector_upper'])
            ef_max_ret.add_sector_constraints(sector_mapper, st.session_state['sector_lower'], st.session_state['sector_upper'])
            ef_max_quadratic.add_sector_constraints(sector_mapper, st.session_state['sector_lower'], st.session_state['sector_upper'])
        else:
            ef_max_sharpe = EfficientFrontier(expected_returns=returns, cov_matrix=risk_model_matrix, weight_bounds=weight_bounds)
            ef_min_vol = EfficientFrontier(expected_returns=returns, cov_matrix=risk_model_matrix, weight_bounds=weight_bounds)
            ef_max_ret = EfficientFrontier(expected_returns=returns, cov_matrix=risk_model_matrix, weight_bounds=weight_bounds)
            ef_max_quadratic = EfficientFrontier(expected_returns=returns, cov_matrix=risk_model_matrix, weight_bounds=weight_bounds)


        ef_max_sharpe.max_sharpe()
        ef_min_vol.min_volatility()
        ef_max_ret._max_return()
        ef_max_quadratic.max_quadratic_utility()
        
        ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
        ret_min, std_min, _ = ef_min_vol.portfolio_performance()
        ret_max, std_max, _ = ef_max_ret.portfolio_performance()
        ret_qua, std_qua, _ = ef_max_quadratic.portfolio_performance()
        
        # Plot optimized portfoliosf     '{x:.1f}%'
        ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label=f"Max Sharpe ({ret_tangent:.1f}%, {std_tangent:.1f}%)")
        ax.scatter(std_min, ret_min, marker="*", s=100, c="b", label=f"Min Vol ({ret_min:.1f}%, {std_min:.1f}%)")
        ax.scatter(std_max, ret_max, marker="*", s=100, c="g", label=f"Max Return ({ret_max:.1f}%, {std_max:.1f}%)")
        ax.scatter(std_qua, ret_qua, marker="*", s=100, c="y", label=f"Max Quadratic Utility ({ret_qua:.1f}%, {std_qua:.1f}%)")
        
        # Generate random portfolios
        n_samples = 10000
        w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
        rets = w.dot(ef.expected_returns)
        stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
        sharpes = rets / stds
        scatter = ax.scatter(stds, rets, marker=".", c=sharpes, cmap="magma", alpha=0.6)
        
        # Add color bar for Sharpe ratios
        cbar = fig.colorbar(scatter, ax=ax, label="Sharpe Ratio", pad=0.01)
        
        ax.set_xlabel("Portfolio Risk (Standard Deviation)", color='white')
        ax.set_ylabel("Portfolio Return", color='white')
        
        # Customize the plot
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        # Set the figure background to transparent
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
        # Set legend background to transparent
        leg = ax.legend(loc='lower right')
        leg.get_frame().set_alpha(0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Format x and y axis labels to show percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
        
        # Project a line based on the selected objective
        if selected_objective == "Max Sharpe":
            ax.axvline(x=std_tangent, color='r', linestyle='--')
            ax.axhline(y=ret_tangent, color='r', linestyle='--')
            ax.scatter(std_tangent, ret_tangent, marker="o", s=100, c="r", label="Real Portfolio")
        elif selected_objective == "Min Volatility":
            ax.axvline(x=std_min, color='b', linestyle='--')
            ax.axhline(y=ret_min, color='b', linestyle='--')
            ax.scatter(std_min, ret_min, marker="o", s=100, c="b", label="Real Portfolio")
        elif selected_objective == "Max Return":
            ax.axvline(x=std_max, color='g', linestyle='--')
            ax.axhline(y=ret_max, color='g', linestyle='--')
            ax.scatter(std_max, ret_max, marker="o", s=100, c="g", label="Real Portfolio")
        elif selected_objective == "Max Quadratic Utility":
            ax.axvline(x=std_qua, color='y', linestyle='--')
            ax.axhline(y=ret_qua, color='y', linestyle='--')
            ax.scatter(std_qua, ret_qua, marker="o", s=100, c="y", label="Real Portfolio")
        
                # Display the plot
        st.pyplot(fig)
        
# Propagate the weights of the selected portfolio and store in session state
if selected_objective == "Max Sharpe":
    weights = ef_max_sharpe.clean_weights()
elif selected_objective == "Min Volatility":
    weights = ef_min_vol.clean_weights()
elif selected_objective == "Max Return":
    weights = ef_max_ret.clean_weights()
elif selected_objective == "Max Quadratic Utility":
    weights = ef_max_quadratic.clean_weights()
else:
    weights = ef.clean_weights()

# Store the weights in session state
st.session_state['portfolio_weights'] = weights
        
