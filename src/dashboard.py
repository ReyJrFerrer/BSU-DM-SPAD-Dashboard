import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
import glob
from streamlit_option_menu import option_menu
from PIL import Image

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
        font-size: 3rem;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2E8B57;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #2E8B57;
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
    .stMarkdown h5 {
        font-size: 1.25rem;
        font-weight: bold;
    }
    .stImageCaption {
        font-size: 1.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

class SPADDashboard:
    def __init__(self):
        self.df = None
        self.image_df = None
        self.numeric_cols = []
        self.categorical_cols = []

    def load_data(self):
        """Load data directly from the data folder."""
        try:
            data_path = 'data/data.csv'
            self.df = pd.read_csv(data_path)
            self.df.columns = self.df.columns.str.strip()
            if 'DATE' in self.df.columns:
                self.df['DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce')
            self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            
            # Load image paths
            image_paths = glob.glob('data/*.jpg') + glob.glob('data/*.png')
            if image_paths:
                image_data = []
                for img_path in image_paths:
                    filename = img_path.split('/')[-1]
                    image_data.append({'NAMEOFIMAGE': filename, 'path': img_path})
                self.image_df = pd.DataFrame(image_data)
                # Merge with main df
                self.df = pd.merge(self.df, self.image_df, on="NAMEOFIMAGE", how="left")

            return True
        except FileNotFoundError:
            st.error(f"Error: The file `data.csv` was not found in the `data/` directory.")
            return False
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False

    def render_eda(self):
        st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
        
        bento_col1, bento_col2 = st.columns((2, 1))

        with bento_col1:
            st.markdown("##### Categorical Feature Distribution")
          
            if 'A/L' in self.df.columns:
                fig = px.histogram(self.df, x='A/L', title='A/L (Atok/La Trinidad) Distribution')
                fig.update_layout(font_size=16, title_font_size=20, hoverlabel=dict(font_size=18))
                st.plotly_chart(fig, use_container_width=True)
            if 'L/F' in self.df.columns:
                fig = px.histogram(self.df, x='L/F', title='L/F (Lab/Field) Distribution')
                fig.update_layout(font_size=16, title_font_size=20, hoverlabel=dict(font_size=18))
                st.plotly_chart(fig, use_container_width=True)

        with bento_col2:
            st.markdown("##### Descriptive Statistics")
            if 'CROP' in self.df.columns:
                for crop_type in self.df['CROP'].unique():
                    st.markdown(f"**{crop_type} SPAD Stats**")
                    st.dataframe(self.df[self.df['CROP'] == crop_type]['SPAD'].describe())
            else:
                st.dataframe(self.df['SPAD'].describe())

        st.markdown('<div class="section-header">Univariate Analysis</div>', unsafe_allow_html=True)
        
        if 'CROP' in self.df.columns:
            for crop_type in self.df['CROP'].unique():
                st.markdown(f"### {crop_type} SPAD Distribution")
                crop_df = self.df[self.df['CROP'] == crop_type]
                
                hist_col1, hist_col2 = st.columns(2)
                
                with hist_col1:
                    lf_option = st.radio(f"Filter by L/F (Lab/Field) for {crop_type}", ['All'] + list(crop_df['L/F'].unique()), key=f'lf_{crop_type}')
                    filtered_df_lf = crop_df if lf_option == 'All' else crop_df[crop_df['L/F'] == lf_option]
                    fig_hist_lf = px.histogram(
                        filtered_df_lf, x='SPAD', nbins=30,
                        title=f'SPAD Distribution by L/F ({lf_option})',
                        labels={'SPAD': 'SPAD Value', 'count': 'Frequency'}
                    )
                    fig_hist_lf.update_layout(font_size=16, title_font_size=20, yaxis_title_font_size=18, xaxis_title_font_size=18, hoverlabel=dict(font_size=18))
                    st.plotly_chart(fig_hist_lf, use_container_width=True)

                with hist_col2:
                    al_option = st.radio(f"Filter by A/L (Atok/La Trinidad) for {crop_type}", ['All'] + list(crop_df['A/L'].unique()), key=f'al_{crop_type}')
                    filtered_df_al = crop_df if al_option == 'All' else crop_df[crop_df['A/L'] == al_option]
                    fig_hist_al = px.histogram(
                        filtered_df_al, x='SPAD', nbins=30,
                        title=f'SPAD Distribution by A/L ({al_option})',
                        labels={'SPAD': 'SPAD Value', 'count': 'Frequency'}
                    )
                    fig_hist_al.update_layout(font_size=16, title_font_size=20, yaxis_title_font_size=18, xaxis_title_font_size=18, hoverlabel=dict(font_size=18))
                    st.plotly_chart(fig_hist_al, use_container_width=True)

        st.markdown('<div class="section-header">Bivariate & Multivariate Analysis</div>', unsafe_allow_html=True)
        
        analysis_col1, analysis_col2 = st.columns(2)
        with analysis_col1:
            if 'CROP' in self.df.columns:
                fig_box = px.box(self.df, x='CROP', y='SPAD', title='SPAD by CROP')
                fig_box.update_layout(font_size=16, title_font_size=20, hoverlabel=dict(font_size=18))
                st.plotly_chart(fig_box, use_container_width=True)

            if 'A/L' in self.df.columns:
                fig_box = px.box(self.df, x='A/L', y='SPAD', title='SPAD by A/L (Atok/ La Trinidad)')
                fig_box.update_layout(font_size=16, title_font_size=20, hoverlabel=dict(font_size=18))
                st.plotly_chart(fig_box, use_container_width=True)

        with analysis_col2:
            if 'L/F' in self.df.columns:
                fig_box = px.box(self.df, x='L/F', y='SPAD', title='SPAD by L/F (Lab/Field)')
                fig_box.update_layout(font_size=16, title_font_size=20, hoverlabel=dict(font_size=18))
                st.plotly_chart(fig_box, use_container_width=True)

            if 'CROP' in self.df.columns and 'A/L' in self.df.columns:
                fig_box = px.box(self.df, x='CROP', y='SPAD', color='A/L', title='SPAD by CROP and A/L (Atok/La Trinidad)')
                fig_box.update_layout(font_size=16, title_font_size=20, hoverlabel=dict(font_size=18))
                st.plotly_chart(fig_box, use_container_width=True)

    def render_gallery(self):
        st.markdown('<div class="section-header">Image Gallery</div>', unsafe_allow_html=True)

        if self.df is not None and 'path' in self.df.columns and not self.df['path'].isnull().all():
            gallery_df = self.df.dropna(subset=['path'])

            # Sidebar controls for filtering and sorting
            st.sidebar.header("Gallery Controls")
            
            # Sorting
            sort_order = st.sidebar.radio("Sort by SPAD Value", ["Ascending", "Descending"])
            sort_ascending = True if sort_order == "Ascending" else False

            # Filtering
            filter_crop = st.sidebar.selectbox("Filter by Crop", ["All"] + list(gallery_df['CROP'].unique()))
            filter_al = st.sidebar.selectbox("Filter by A/L (Atok/La Trinidad)", ["All"] + list(gallery_df['A/L'].unique()))
            filter_lf = st.sidebar.selectbox("Filter by L/F (Lab/Field)", ["All"] + list(gallery_df['L/F'].unique()))

            # Apply filters
            if filter_crop != "All":
                gallery_df = gallery_df[gallery_df['CROP'] == filter_crop]
            if filter_al != "All":
                gallery_df = gallery_df[gallery_df['A/L'] == filter_al]
            if filter_lf != "All":
                gallery_df = gallery_df[gallery_df['L/F'] == filter_lf]

            # Apply sorting
            gallery_df = gallery_df.sort_values(by="SPAD", ascending=sort_ascending)

            st.markdown(f"#### Displaying {len(gallery_df)} images")

            for i in range(0, len(gallery_df), 4):
                cols = st.columns(4)
                for j in range(4):
                    if i + j < len(gallery_df):
                        row = gallery_df.iloc[i+j]
                        try:
                            image = Image.open(row['path'])
                            caption = f"SPAD: {row['SPAD']:.2f}"
                            if 'CROP' in row:
                                caption += f" | Crop: {row['CROP']}"
                            cols[j].image(image, caption=caption, use_container_width=True)
                        except FileNotFoundError:
                            cols[j].warning(f"Image not found: {row['NAMEOFIMAGE']}")
                        except Exception as e:
                            cols[j].error(f"Error loading {row['NAMEOFIMAGE']}")
        else:
            st.warning("No images found or 'NAMEOFIMAGE' column not properly linked.")

def main():
    """Main dashboard function"""
    st.markdown('<div class="main-header">SPAD Analysis Dashboard</div>', unsafe_allow_html=True)

    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["SPAD EDA", "Image Gallery", "About"],
            icons=['bar-chart-line', 'images', 'info-circle'],
            menu_icon="cast",
            default_index=0,
        )

    dashboard = SPADDashboard()
    
    if dashboard.load_data():
        st.sidebar.success("Data loaded successfully!")

        if selected == "SPAD EDA":
            dashboard.render_eda()
        
        elif selected == "Image Gallery":
            dashboard.render_gallery()

        elif selected == "About":
            st.markdown('<div class="section-header">About</div>', unsafe_allow_html=True)
            st.info("This section is under construction.")
    else:
        st.info("Please ensure `data.csv` is in the `src/data` directory and contains the required columns.")

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
