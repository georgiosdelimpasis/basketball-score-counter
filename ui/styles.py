"""
Custom CSS styles for the Streamlit application.
Provides polished, professional appearance.
"""
import streamlit as st


def inject_custom_css():
    """Inject custom CSS into the Streamlit app."""
    st.markdown("""
        <style>
        /* Main title styling */
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }

        /* Subtitle styling */
        .subtitle {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }

        /* Stats container */
        .stats-container {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }

        /* Stat item */
        .stat-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #ddd;
        }

        .stat-item:last-child {
            border-bottom: none;
        }

        .stat-label {
            font-weight: 600;
            color: #333;
        }

        .stat-value {
            color: #1f77b4;
            font-weight: 700;
        }

        /* FPS indicator */
        .fps-good {
            color: #00c853;
            font-weight: 700;
        }

        .fps-medium {
            color: #ffa726;
            font-weight: 700;
        }

        .fps-low {
            color: #f44336;
            font-weight: 700;
        }

        /* Detection box */
        .detection-box {
            border: 2px solid #1f77b4;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background-color: white;
        }

        /* Model info card */
        .model-info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }

        .model-info h3 {
            margin: 0;
            font-size: 1.1rem;
        }

        .model-info p {
            margin: 5px 0;
            font-size: 0.9rem;
        }

        /* Warning message */
        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12px;
            margin: 10px 0;
            border-radius: 4px;
        }

        /* Success message */
        .success-box {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 12px;
            margin: 10px 0;
            border-radius: 4px;
        }

        /* Error message */
        .error-box {
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 12px;
            margin: 10px 0;
            border-radius: 4px;
        }

        /* Button styling */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        /* Slider styling */
        .stSlider {
            padding: 10px 0;
        }

        /* Selectbox styling */
        .stSelectbox {
            margin-bottom: 10px;
        }

        /* Sidebar styling */
        .css-1d391kg {
            padding: 2rem 1rem;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Webcam container */
        .webcam-container {
            border: 3px solid #1f77b4;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }

        /* Class badge */
        .class-badge {
            display: inline-block;
            background-color: #1f77b4;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 4px;
        }

        /* Info tooltip */
        .info-tooltip {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            font-size: 0.9rem;
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
