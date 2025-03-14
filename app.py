import streamlit as st
from components.document_summary import document_summarization_ui
from components.contract_analysis import contract_analysis_ui
from components.voice_qa import voice_qa_ui
import streamlit as st

# Configure page for caching
st.set_page_config(
    page_title="Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/legal-assistant',
        'Report a bug': 'https://github.com/yourusername/legal-assistant/issues',
        'About': "Legal Assistant - An AI-powered tool for legal professionals"
    }
)

# Initialize session state for caching
if 'cache' not in st.session_state:
    st.session_state.cache = {}

# Add custom CSS
def load_css():
    with open("static/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    load_css()
except:
    st.warning("Custom styles not loaded. Application will function normally.")

# Application title and description
st.title("🧾 AI-Enhanced Legal Assistant")
st.markdown("""
    This application provides AI-powered tools for legal professionals, including:
    - Document summarization for legal documents, treaties, and agreements
    - Document analysis for contracts, agreements, and international treaties
    - Voice-commanded question answering for legal research
    
    All processing is done locally on your machine for data privacy.
""")

# Session state initialization
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Document Summarization"

# Sidebar navigation
st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio(
    "Choose a tool:",
    ["Document Summarization", "Document Analysis", "Voice Q&A"],
    key="sidebar_selection"
)

st.session_state.active_tab = selected_tab

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.info("""
    ### About
    This application uses:
    - Ollama for AI model inference
    - Whisper for speech recognition
    - LangChain for document processing
    
    All processing is done locally for privacy.
""")

# Main content based on selected tab
if st.session_state.active_tab == "Document Summarization":
    document_summarization_ui()
elif st.session_state.active_tab == "Document Analysis":  # Changed from "Contract Analysis"
    contract_analysis_ui()  # Function name can stay the same
elif st.session_state.active_tab == "Voice Q&A":
    voice_qa_ui()

# Footer
st.markdown("---")
st.markdown("© 2025 Legal Assistant - Powered by AI")