"""
Custom CSS styles for the Streamlit application.
Professional light mode with white, black, and grey colors.
"""
import streamlit as st


def inject_custom_css():
    """Inject custom CSS into the Streamlit app."""
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Global styles */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Main container */
        .main {
            background-color: #ffffff !important;
        }

        /* Streamlit containers */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Main title styling */
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            color: #000000;
            text-align: center;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
        }

        /* Subtitle styling */
        .subtitle {
            font-size: 1.2rem;
            color: #666666;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 400;
        }

        /* Model info cards */
        .model-info-card {
            background: #f5f5f5;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            text-align: center;
            transition: all 0.2s ease;
        }

        .model-info-card:hover {
            background: #eeeeee;
            border-color: #cccccc;
        }

        .model-info-card strong {
            color: #000000;
            font-weight: 700;
            font-size: 0.85rem;
        }

        /* Detection box */
        .detection-box {
            background: #f5f5f5;
            border: 1px solid #e0e0e0;
            border-radius: 16px;
            padding: 2.5rem;
            margin: 2rem 0;
            color: #000000;
        }

        .detection-box h2 {
            color: #000000;
            font-weight: 800;
            font-size: 2rem;
            margin-bottom: 1.5rem;
        }

        .detection-box ol {
            font-size: 1rem;
            line-height: 2;
            color: #333333;
        }

        .detection-box strong {
            color: #000000;
            font-weight: 600;
        }

        /* Stats metrics */
        [data-testid="stMetricValue"] {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            color: #000000 !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.875rem !important;
            font-weight: 600 !important;
            color: #666666 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }

        /* Button styling */
        .stButton > button {
            background: #000000 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.25rem !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            transition: all 0.2s ease !important;
            letter-spacing: 0.3px !important;
        }

        .stButton > button:hover {
            background: #1a1a1a !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        }

        .stButton > button:active {
            transform: translateY(0);
        }

        .stButton > button:disabled {
            background: #e0e0e0 !important;
            color: #999999 !important;
            cursor: not-allowed !important;
        }

        /* Slider styling */
        .stSlider [data-baseweb="slider"] {
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .stSlider [role="slider"] {
            background-color: #000000 !important;
            border: 2px solid #ffffff !important;
            width: 18px !important;
            height: 18px !important;
        }

        .stSlider [data-baseweb="slider"] > div:first-child {
            background: linear-gradient(90deg, #cccccc 0%, #000000 100%) !important;
        }

        /* Selectbox styling */
        .stSelectbox label, .stNumberInput label {
            color: #333333 !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            margin-bottom: 0.5rem !important;
        }

        .stSelectbox [data-baseweb="select"] > div,
        .stNumberInput input {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 8px !important;
        }

        .stSelectbox [data-baseweb="select"]:hover > div,
        .stNumberInput input:hover {
            border-color: #cccccc !important;
        }

        .stSelectbox [data-baseweb="select"]:focus-within > div,
        .stNumberInput input:focus {
            border-color: #000000 !important;
            box-shadow: 0 0 0 1px #000000 !important;
        }

        /* Dropdown menu */
        [data-baseweb="popover"] {
            background-color: #ffffff !important;
            border: 1px solid #e0e0e0 !important;
        }

        [data-baseweb="menu"] li {
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        [data-baseweb="menu"] li:hover {
            background-color: #f5f5f5 !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #fafafa !important;
            border-right: 1px solid #e0e0e0 !important;
        }

        [data-testid="stSidebar"] .stMarkdown {
            color: #333333 !important;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #000000 !important;
            font-weight: 700 !important;
        }

        /* Divider styling */
        .stDivider, hr {
            border: none !important;
            height: 1px !important;
            background: #e0e0e0 !important;
            margin: 1.5rem 0 !important;
        }

        /* Success/Info messages */
        .stSuccess, .stInfo {
            background-color: #f5f5f5 !important;
            border-left: 3px solid #000000 !important;
            color: #333333 !important;
            padding: 0.75rem 1rem !important;
            border-radius: 4px !important;
        }

        .stError, .stWarning {
            background-color: #f9f9f9 !important;
            border-left: 3px solid #666666 !important;
            color: #333333 !important;
            padding: 0.75rem 1rem !important;
            border-radius: 4px !important;
        }

        /* Video/Image container */
        .stImage > img {
            border-radius: 12px !important;
            border: 1px solid #e0e0e0 !important;
        }

        /* Zone setup styling */
        .zone-card {
            background: #f5f5f5;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        }

        .zone-card h3 {
            color: #000000;
            margin-bottom: 1rem;
            font-size: 1.2rem;
            font-weight: 700;
        }

        .zone-card p {
            color: #333333;
            line-height: 1.8;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #f5f5f5;
        }

        ::-webkit-scrollbar-thumb {
            background: #cccccc;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #999999;
        }

        /* Text input styling */
        .stTextInput input {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 8px !important;
        }

        .stTextInput input:focus {
            border-color: #000000 !important;
            box-shadow: 0 0 0 1px #000000 !important;
        }

        /* Markdown text color */
        .stMarkdown, .stMarkdown p {
            color: #333333 !important;
        }

        .stMarkdown strong {
            color: #000000 !important;
        }

        /* Loading spinner */
        .stSpinner > div {
            border-top-color: #000000 !important;
        }

        /* Code blocks */
        code {
            background-color: #f5f5f5 !important;
            color: #000000 !important;
            padding: 0.2rem 0.4rem !important;
            border-radius: 4px !important;
        }

        /* Expander */
        .streamlit-expanderHeader {
            background-color: #f5f5f5 !important;
            color: #000000 !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 8px !important;
        }

        .streamlit-expanderHeader:hover {
            background-color: #eeeeee !important;
            border-color: #cccccc !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #f5f5f5 !important;
            color: #666666 !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: #000000 !important;
            color: #ffffff !important;
            border-color: #000000 !important;
        }
        </style>
    """, unsafe_allow_html=True)


def show_fps_indicator(fps: float) -> str:
    """
    Return CSS class for FPS indicator based on value.

    Args:
        fps: Current FPS value

    Returns:
        CSS class name
    """
    if fps >= 10:
        return "fps-good"
    elif fps >= 5:
        return "fps-medium"
    else:
        return "fps-low"
