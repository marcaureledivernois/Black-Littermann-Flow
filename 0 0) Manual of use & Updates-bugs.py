import streamlit as st

# Create a sidebar for navigation
st.sidebar.markdown("<h2 style='text-align: center;'>Navigation</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Manual of Use", "Next Updates", "Known Bugs"])

# Define the content for each page
if page == "Manual of Use":
    st.title("Manual of Use")
    st.write("""
    ## Introduction
    
    To provide a bit more context, this interactive app is designed to guide users through every step of portfolio optimization using the Black-Litterman model. The app offers complete customization of parameters, objective functions, and factors, allowing for flexible and robust portfolio construction. Moreover, the app allows for backtesting and in-depth analysis.

    ## Before you start         
    Before you start, make sure to follow the pages in the sidebar in order. This will ensure that you have a complete understanding of the Black-Litterman model and how to use the app effectively.
                     
    ## Getting Started
    1. Step 1: On the page Asset selection & Risk Model, select the assets.
    2. Step 2: Follow the pages in order.
    3. Step 3: At the end of the process, you will have a complete portfolio optimization, backtest and analysis.
    4. Step 4: Adjust the advanced settings to improve performance.
             
    ## Features
    - Easy-to-Use Interface: The app guides you through every step
    - Easy-to-Use Process: The app guides uses default values for the parameters, ensuring quick insights.     
    - Full Parameter Customization: In the sidebar you can adjust every factor involved in the optimization process.
    - Objective Function Flexibility: Choose the objective function that fits your strategy (e.g., max Sharpe ratio).
    - Interactive Visualizations: View the efficient frontier, portfolio weights, and risk metrics in real-time.
    """)


elif page == "Next Updates":
    st.title("Next Updates")
    st.write("""
    ## Upcoming Features
    - Feature 1: Target return and Target volatility objectives.
    - Feature 2: Portfolio perfromance metrics.
    - Feature 3: Out of sample Backtest with performance metrics.
    - Feature 4: Fama french 5 Analysis.
    """)

elif page == "Known Bugs":
    st.title("Known Bugs")
    st.write("""
    ## Current Issues
    - Bug 1: When a change is made in a previous page one must go trought every page in order to propagate the changes.
    - Bug 2: If the page is reloaded and the user is in advanced page error is shown.
    - Bug 3: The features of target var and ret are in the code but not working.
    - Bug 4: Somewhere in the formating the values are divided by 10 or the model is currently broken.
             
    ## Reporting Bugs
    If you encounter any issues, please report them to [michele.moreschini@unil.ch].
    """)
