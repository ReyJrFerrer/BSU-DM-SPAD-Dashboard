import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import warnings
from streamlit_option_menu import option_menu

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="SPAD Analysis Dashboard",
    page_icon="assets/image.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E8B57;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2E8B57;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 7px solid #4CAF50;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

class SPADDashboard:
    def __init__(self):
        self.df = None
        self.numeric_cols = []
        self.categorical_cols = []

    def load_data(self, uploaded_file=None):
        """Load data from uploaded file"""
        if uploaded_file is not None:
            try:
                self.df = pd.read_csv(uploaded_file)
                self.df.columns = self.df.columns.str.strip()
                if 'DATE' in self.df.columns:
                    self.df['DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce')
                self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
                self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
                return True
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return False
        return False

    def show_help_modal(self):
        with st.expander("View required CSV format"):
            st.markdown("""
            ### Required CSV Format

            The uploaded CSV file should have the following columns:

            - **SPAD**: `(numeric)` The SPAD value.
            - **CROP**: `(text)` The type of crop (e.g., 'CABBAGE', 'POTATO').
            - **A/L**: `(text)` The location where the crop was grown.
            - **L/F**: `(text)` Whether the sample is from a leaf or a fruit.
            - **NAMEOFIMAGE**: `(text)` The name of the image file associated with the sample.
            """)

def main():
    """Main dashboard function"""
    st.markdown('<div class="main-header">SPAD Analysis Dashboard</div>', unsafe_allow_html=True)

    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["SPAD EDA", "Image EDA", "About"],
            icons=['bar-chart-line', 'images', 'info-circle'],
            menu_icon="cast",
            default_index=0,
        )

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload your SPAD dataset CSV file"
    )

    dashboard = SPADDashboard()

    if uploaded_file is None:
        dashboard.show_help_modal()
        st.info("Please upload a CSV file to begin analysis.")
        return

    if dashboard.load_data(uploaded_file):
        st.sidebar.success("Data loaded successfully!")

        if selected == "SPAD EDA":
            st.markdown("## SPAD EDA")
            bento_col1, bento_col2 = st.columns((2, 1))

            with bento_col1:
                st.markdown('<div class="section-header">Data Inspection and Cleaning</div>', unsafe_allow_html=True)
                if 'CROP' in dashboard.df.columns:
                    for crop_type in dashboard.df['CROP'].unique():
                        st.markdown(f"##### {crop_type}")
                        st.dataframe(dashboard.df[dashboard.df['CROP'] == crop_type]['SPAD'].describe())
                else:
                    st.dataframe(dashboard.df['SPAD'].describe())

                st.markdown("#### Missing Values")
                st.dataframe(dashboard.df.isnull().sum().to_frame('Missing Values'))

            with bento_col2:
                st.markdown('<div class="section-header">Categorical Features</div>', unsafe_allow_html=True)
                for col in ['A/L', 'L/F']:
                    if col in dashboard.df.columns:
                        st.bar_chart(dashboard.df[col].value_counts())

            st.markdown('<div class="section-header">Univariate Analysis</div>', unsafe_allow_html=True)
            if 'CROP' in dashboard.df.columns:
                fig_hist = px.histogram(
                    dashboard.df, x='SPAD', color='CROP', nbins=20,
                    title='SPAD Distribution by CROP',
                    labels={'SPAD': 'SPAD Value', 'count': 'Frequency'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                fig_hist = px.histogram(
                    dashboard.df, x='SPAD', nbins=20,
                    title='SPAD Distribution',
                    labels={'SPAD': 'SPAD Value', 'count': 'Frequency'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown('<div class="section-header">Bivariate Analysis</div>', unsafe_allow_html=True)
            if 'CROP' in dashboard.df.columns:
                fig_box = px.box(
                    dashboard.df, x='CROP', y='SPAD',
                    title='SPAD Distribution by CROP'
                )
                st.plotly_chart(fig_box, use_container_width=True)

            for col in ['A/L', 'L/F']:
                if col in dashboard.df.columns:
                    st.markdown(f"#### SPAD vs. {col}")
                    fig_box = px.box(
                        dashboard.df, x=col, y='SPAD',
                        title=f'SPAD Distribution by {col}'
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

        

            st.markdown('<div class="section-header">Multivariate Analysis</div>', unsafe_allow_html=True)
            if 'CROP' in dashboard.df.columns and 'A/L' in dashboard.df.columns:
                st.markdown("#### Interaction between CROP and A/L on SPAD")
                fig_box = px.box(
                    dashboard.df, x='CROP', y='SPAD', color='A/L',
                    title='SPAD by CROP and A/L'
                )
                st.plotly_chart(fig_box, use_container_width=True)

        elif selected == "Image EDA":
            st.markdown("## Image EDA")
            st.markdown("This section is under construction.")

        elif selected == "About":
            st.markdown("## About")
            st.markdown("This section is under construction.")

    else:
        st.error("Failed to load data. Please upload a valid CSV file.")

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>SPAD Analysis Dashboard | Built with Streamlit</p>
            <p>For cabbage and potato leaf analysis</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()