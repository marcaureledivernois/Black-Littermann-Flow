import pandas as pd
import streamlit as st
from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Callback from another page
Risk_Matrix = st.session_state['risk_model_matrix']
Asset_List = st.session_state['Asset_List']

# Load Data
Constituents = pd.read_csv('Resources/constituents.csv')
Analysis_df = pd.read_csv('Resources/Analysis_df.csv')

# Filter Constituents based on Asset_List
filtered_constituents = Constituents[Constituents['Symbol'].isin(Asset_List)]
MkCap_Sector = filtered_constituents[['Symbol', 'GICS Sector', 'Market Cap']]

# Benchmark Prices
Benchmark_Prices = Analysis_df.set_index(Analysis_df.columns[0]).iloc[:, [-1]]

# Implied Risk Aversion
Risk_Adversion = float(black_litterman.market_implied_risk_aversion(Benchmark_Prices).iloc[0])

# Sidebar title
st.sidebar.markdown("<h2 style='text-align: center;'>Advanced Settings</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)

# Section title for Data Selection options
st.sidebar.markdown("<h4 style='text-align: left;'>Black Littermann options</h4>", unsafe_allow_html=True)

# Callback function to update session state
def update_session_state(key, value):
    st.session_state[key] = value

# Checkbox to allow modification of the risk aversion coefficient
modify_risk_aversion = st.sidebar.checkbox(
    f'Modify Risk Aversion Coefficient ({Risk_Adversion:.1f})', 
    value=st.session_state.get('modify_risk_aversion', False),
    on_change=update_session_state,
    args=('modify_risk_aversion', not st.session_state.get('modify_risk_aversion', False))
)

# Display the risk aversion coefficient with an option to modify it if the checkbox is selected
if st.session_state.get('modify_risk_aversion', False):
    risk_aversion = st.sidebar.slider(
        'New Risk Aversion Coefficient', 
        min_value=0.0, 
        max_value=10.0, 
        value=st.session_state.get('risk_aversion', float(Risk_Adversion)), 
        step=0.1, 
        format="%.2f",
        label_visibility='collapsed'
    )
    st.session_state['risk_aversion'] = risk_aversion
else:
    risk_aversion = Risk_Adversion
    st.session_state['risk_aversion'] = risk_aversion

# Checkbox to allow insertion of the analyst views
insert_analyst_views = st.sidebar.checkbox(
    'Insert Analyst Views', 
    value=st.session_state.get('insert_analyst_views', False),
    on_change=update_session_state,
    args=('insert_analyst_views', not st.session_state.get('insert_analyst_views', False))
)

# Checkbox to allow modification of the tau parameter if analyst views are inserted
if st.session_state.get('insert_analyst_views', False):
    modify_tau = st.sidebar.checkbox(
        f'Modify Tau Parameter (0.05)', 
        value=st.session_state.get('modify_tau', False),
        on_change=update_session_state,
        args=('modify_tau', not st.session_state.get('modify_tau', False))
    )

    # Display the tau parameter with an option to modify it if the checkbox is selected
    if st.session_state.get('modify_tau', False):
        tau = st.sidebar.slider(
            'New Tau Parameter', 
            min_value=0.01, 
            max_value=0.10, 
            value=st.session_state.get('tau', 0.05), 
            step=0.01, 
            format="%.2f",
            label_visibility='collapsed'
        )
        st.session_state['tau'] = tau
    else:
        tau = 0.05
        st.session_state['tau'] = tau
else:
    tau = 0.05
    st.session_state['tau'] = tau

# Main page#############################################################################################################
# Define the custom color palette
colors = ['#f2e6d9','#ff4b4b','#ff7a7a','#ffb8b8', '#ff9999']
custom_cmap = LinearSegmentedColormap.from_list("custom_red", colors)


st.title("Black Littermann")

# Calculate market capitalization
market_caps = dict(zip(MkCap_Sector['Symbol'], MkCap_Sector['Market Cap']))

# Calculate the market implied prior returns using the updated risk aversion coefficient
BlPrior = black_litterman.market_implied_prior_returns(market_caps, risk_aversion, Risk_Matrix)

# Input for the vector of analyst views and confidence
if st.session_state.get('insert_analyst_views', False):
    st.subheader("Analyst Views and Confidence")

    num_assets = len(Asset_List)
    views = []
    confidences = []

    for i in range(0, num_assets, 2):
        cols = st.columns(4)
        for j in range(2):
            if i + j < num_assets:
                with cols[j * 2]:
                    view = st.number_input(f'View on {Asset_List[i + j]}', value=0.0, step=0.1, format="%.2f", key=f'view_{i + j}', min_value=0.0)
                    views.append(view)
                with cols[j * 2 + 1]:
                    confidence = st.number_input(f'Confidence on {Asset_List[i + j]}', value=0.0, step=0.1, format="%.2f", key=f'confidence_{i + j}', min_value=0.0, max_value=1.0)
                    confidences.append(confidence)

    st.session_state['views'] = views
    st.session_state['confidences'] = confidences

    # Infer the picking matrix given that all the views are absolute
    if st.session_state.get('insert_analyst_views', False):
        # Initialize the picking matrix with zeros
        P = pd.DataFrame(0, index=Asset_List, columns=Asset_List)

        # Convert views and confidences to numpy arrays
        views = np.array(st.session_state['views'])
        confidences = np.array(st.session_state['confidences'])

        # Update the picking matrix for non-zero views and confidences
        for i, asset in enumerate(Asset_List):
            if views[i] != 0.0 and confidences[i] != 0.0:
                P.at[asset, asset] = 1

        # Create the Q vector and Omega matrix
        Q = views

        # Calculate the posterior returns using the Black-Litterman model with the staticidzorek_method
        if st.session_state.get('insert_analyst_views', False):
            bl = BlackLittermanModel(
            view_confidences=confidences,
            cov_matrix=Risk_Matrix,
            pi=BlPrior,
            omega ="idzorek",
            Q=Q,
            P=P,
            tau=tau,
            risk_aversion=risk_aversion
            )
            Blposterior = bl.bl_returns()
            
        # Plot the implied prior returns as percentages within an expandable box
        # with st.expander("Show Expected Posterior Returns"):
            with plt.style.context('dark_background'):
                fig, ax = plt.subplots()
                
                # Increase the saturation of the colors
                # colors = plt.cm.magma(range(len(BlPrior)))
                # saturated_colors = [plt.cm.magma(i / len(colors) * 0.8 + 0.2) for i in range(len(colors))]
                
                # Create a DataFrame to hold the data for plotting
                plot_data = pd.DataFrame({
                    'Prior': BlPrior,
                    'Posterior': Blposterior if st.session_state.get('insert_analyst_views', False) else np.nan,
                    'Analyst Views': st.session_state['views'] if st.session_state.get('insert_analyst_views', False) else np.nan
                }, index=Asset_List)
                
                # Plot the data
                plot_data.plot(kind='bar', ax=ax, color=colors, edgecolor='none', alpha=0.7)
                
                # Customize the plot
                ax.set_xlabel('Tickers', color='white')
                ax.set_ylabel('Expected Return (%)', color='white')  # Update label to show percentage
                ax.set_title('Expected Implied Returns (%)', color='white')
                
                # Set tick parameters with white color
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                
                # Format the y-axis to show percentage sign
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
                
                # Set the figure background to transparent
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)
                
                # Add legend with transparent background
                legend = ax.legend()
                legend.get_frame().set_alpha(0)

                # Display the graph directly
                st.pyplot(fig)

# Plot the implied prior returns as percentages within an expandable box if analyst views are not inserted
if not st.session_state.get('insert_analyst_views', False):
    # with st.expander("Show Expected Implied Returns Plot"):
        with plt.style.context('dark_background'):
            fig, ax = plt.subplots()
            
            # Increase the saturation of the colors
            colors = custom_cmap(np.linspace(0, 1, len(BlPrior)))

            
            # Create a DataFrame to hold the data for plotting
            plot_data = pd.DataFrame({
                'Prior': BlPrior,
            }, index=Asset_List)
            
            # Plot the data
            plot_data.plot(kind='bar', ax=ax, color=colors[2], edgecolor='none', alpha=0.9)
            
            # Customize the plot
            ax.set_xlabel('Tickers', color='white')
            ax.set_ylabel('Expected Return (%)', color='white')  # Update label to show percentage
            ax.set_title('Expected Implied Returns (%)', color='white')
            
            # Set tick parameters with white color
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            
            # Format the y-axis to show percentage sign
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
            
            # Set the figure background to transparent
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            
            # Remove legend
            ax.get_legend().remove()

            # Display the graph directly
            st.pyplot(fig)

# Determine the returns and risk model matrix based on whether analyst views are inputted
if st.session_state.get('insert_analyst_views', False):
    returns = Blposterior
    risk_model_matrix = bl.bl_cov()
else:
    returns = BlPrior
    risk_model_matrix = Risk_Matrix

# Propagate the returns and risk model matrix to the next page
st.session_state['returns'] = returns
st.session_state['risk_model_matrix'] = risk_model_matrix

# Display the returns and risk model matrix
# Plot the risk model matrix
with plt.style.context('dark_background'):
    fig, ax = plt.subplots()
    
    sns.heatmap(risk_model_matrix, annot=True, fmt=".2f", cmap=custom_cmap, ax=ax)
    
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