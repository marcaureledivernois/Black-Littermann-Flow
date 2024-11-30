import streamlit as st

# Create a sidebar for navigation
st.sidebar.markdown("<h2 style='text-align: center;'>Navigation</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Manual of Use", "Next Updates", "Resolved and Known Bugs"])

# Define the content for each page
if page == "Manual of Use":
    st.title("Manual of Use")
    st.write("""
    ## Introduction

    Hello, World! :sunglasses:

    To provide a bit more context, this interactive app is designed to guide users through every step of a portfolio optimization using the Black-Litterman model. 

    The app offers complete customization of parameters, objective functions, and in-depth analysis, allowing for flexible and robust portfolio construction.""")
    st.divider()

    st.write("""
    ## Before you start         
    Make sure to follow the pages in the sidebar in order. This will ensure that you have a complete understanding of the Black-Litterman model and how to use the app effectively.

    ## Getting Started
    - Step 1: On page _1) Asset screener_, select the assets, dates, frequency, benchmark, and risk model.
    - Step 2: Follow the pages in order.
    - Step 3: Adjust the _advanced settings_ to alter performance.
    - Step 4: At the end of the process, you will have a complete portfolio optimization and performance metrics.""")

    st.divider()
    st.write("""
    ## Features
    - Easy-to-Use Interface: Effective and User-friendly Asset Screener.
    - Easy-to-Use Process: The app uses default values for the parameters, ensuring quick insights.     
    - Parameter Customization: The sidebar allows for the adjustment of parameters of the optimization.
    - Objective Function Flexibility: Choose the objective function that fits your strategy: 
        - e.g. Max Sharpe Ratio, MV Portfolio
    - Interactive Visualizations: View the Efficient Frontier, Portfolio Weights, and Risk Metrics.
    """)


elif page == "Next Updates":
    st.title("Next Updates")
    st.write("""
    ## Upcoming/Possible Features
    - Feature 1: Target return and volatility objectives.
    - Feature 2: Implementation of relative views.
    - Feature 3: Out-of-sample Backtest with performance metrics.
    - Feature 4: Fama-French 5-Factor Analysis.
    """)

elif page == "Resolved and Known Bugs":
    st.title("Resolved bugs")
    st.write("""
    - Problematic companies with very low/no data were filtered.
    - A warning message appears when a/various company(ies) do not cover all data range.
    - A bar chart ensures the visualization of negative weights when short selling is allowed.  
    - A linearly segmented colormap ensures color cohesion along the app. 
    """)

    st.divider()

    st.title("Known Bugs")
    st.write("""
    ## Current Issues
    - Bug 1: When a change is made in a previous page one must go trought every page in order to propagate the changes.
    - Bug 2: If the page is reloaded and the user is in an advanced page, an error is shown.
    - Bug 3: Somewhere in the formating the values are divided by 10 or the model is currently misleading.

    ## Reporting Bugs
    If you encounter any issues, please report them to [michele.moreschini@unil.ch].
    """)