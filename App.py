import streamlit as st

pg = st.navigation([st.Page("0 0) Manual of use & Updates-bugs.py"),
                    st.Page("1 1) Asset screener.py"), 
                    st.Page("2 2) Data Preparation.py"), 
                    st.Page("3 3) Model & Views.py"), 
                    st.Page("4 4) Efficient Frontier.py"),
                    st.Page("5 5) Portfolio.py"),
                    ])
pg.run()