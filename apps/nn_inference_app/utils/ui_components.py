import streamlit as st

def apply_custom_css():
    """Injects custom CSS for a modern, professional AI dashboard aesthetic."""
    st.markdown("""
    <style>
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Cards */
    .stCard {
        background-color: var(--secondary-background-color);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 24px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Primary Buttons */
    .stButton>button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.2s ease-in-out !important;
        width: 100%;
    }
    .stButton>button:hover {
        filter: brightness(1.1);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        transform: translateY(-1px);
    }
    
    /* Input Fields */
    .stSelectbox>div>div, .stNumberInput>div>div, .stTextInput>div>div {
        border-radius: 8px !important;
        border: 1px solid rgba(128, 128, 128, 0.4) !important;
    }
    .stSelectbox>div>div:focus-within, .stNumberInput>div>div:focus-within, .stTextInput>div>div:focus-within {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 1px var(--primary-color) !important;
    }

    /* Headers */
    h1, h2, h3 {
        font-weight: 600;
        color: var(--text-color);
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25em 0.75em;
        font-size: 0.85em;
        font-weight: 600;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 9999px;
        background-color: rgba(147, 51, 234, 0.15);
        color: #a855f7;
        margin-bottom: 1rem;
        border: 1px solid rgba(147, 51, 234, 0.3);
    }

    /* Sidebar Footer */
    .sidebar-footer {
        margin-top: auto;
        padding-top: 2rem;
        text-align: left;
        font-size: 0.85rem;
        color: var(--text-color);
        opacity: 0.7;
    }
    
    /* Result Cards */
    .result-card-success {
        background-color: rgba(34, 197, 94, 0.1);
        border-left: 4px solid #22c55e;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    .result-card-success h3 {
        color: #22c55e !important;
    }
    .result-card-danger {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    .result-card-danger h3 {
        color: #ef4444 !important;
    }
    .result-card-info {
        background-color: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    .result-card-info h3 {
        color: #3b82f6 !important;
    }
    .result-card-danger p, .result-card-success p, .result-card-info p {
        color: var(--text-color);
    }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Renders the standard navigation sidebar and footer."""
    st.sidebar.markdown("### NTI-ML NN App")
    st.sidebar.markdown("Explore Deep Learning & Neural Network models trained on the Loan dataset.")
    st.sidebar.markdown("---")
    
    # Model Status
    st.sidebar.markdown("**Network Status**")
    st.sidebar.success("🔮 Keras Models Loaded")
    
    # Footer
    st.sidebar.markdown("""
    <div class="sidebar-footer">
        <b>Developer:</b><br>Ahmed Sameh Mohamed Zaky<br>
        <i>NTI Deep Learning Project</i>
    </div>
    """, unsafe_allow_html=True)

def render_header(title: str, description: str, badge_text: str = "Deep Learning"):
    """Renders a modern hero section."""
    st.markdown(f'<span class="badge">🧠 {badge_text}</span>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f"<p style='font-size: 1.1rem; color: #475569;'>{description}</p>", unsafe_allow_html=True)
    st.markdown("---")

def render_about_section():
    """Renders the About section at the bottom of pages."""
    st.markdown("---")
    st.header("About the Neural Network Hub")
    st.markdown("""
    This specialized dashboard focuses on the **Multilayer Perceptron (MLP)** models implemented using **TensorFlow and Keras**. 
    It showcases model evaluations, feature permutation importances, and interactive inference engines for both classification and regression targets.
    
    **Architectures Compared:**
    - Baseline (Simple Dense Networks)
    - Regularized Models (L2 Penalty)
    - Dropout Regularized Networks
    - Batch-Normalized Models
    - Final Hybrid Deep architectures (combining Dropout & Batchnorm)
    
    *Developed by Ahmed Sameh Mohamed Zaky.*
    """)
