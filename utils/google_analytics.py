import streamlit as st

def set_google_analytics():
    return st.markdown(
        """
            <!-- Google tag (gtag.js) -->
            <script async src="https://www.googletagmanager.com/gtag/js?id=G-Z0M6M8ENE2"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag(){dataLayer.push(arguments);}
                gtag('js', new Date());

                gtag('config', 'G-Z0M6M8ENE2');
            </script>
        """, unsafe_allow_html=True
    )