
import home,eda,modeling,summary
import streamlit as st
PAGES = {
    'Home': home,
    'EDA': eda,
    'Modeling': modeling,
    'Summary': summary
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()