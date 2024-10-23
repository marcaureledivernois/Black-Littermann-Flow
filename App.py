import streamlit as st

pg = st.navigation([st.Page("Manual of use & Updates-bugs.py"),
                    st.Page("Asset selection & Risk Model.py"), 
                    st.Page("Black-Littermann views.py"), 
                    st.Page("Efficient Frontier.py"), 
                    st.Page("Portfolio.py")])
pg.run()