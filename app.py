import streamlit as st
from components.document_summary import document_summarization_ui
from components.contract_analysis import contract_analysis_ui
from components.voice_qa import voice_qa_ui

# Page configuration
st.set_page_config(
    page_title="Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    - Document summarization
    - Contract review and analysis
    - Voice-commanded question answering
    
    All processing is done locally on your machine for data privacy.
""")

# Session state initialization
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Document Summarization"

# Sidebar navigation
st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio(
    "Choose a tool:",
    ["Document Summarization", "Contract Analysis", "Voice Q&A"],
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
elif st.session_state.active_tab == "Contract Analysis":
    contract_analysis_ui()
elif st.session_state.active_tab == "Voice Q&A":
    voice_qa_ui()

# Footer
st.markdown("---")
st.markdown("© 2025 Legal Assistant - Powered by AI")