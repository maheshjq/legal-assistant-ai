"""
Document summarization component for Streamlit application.
"""

import streamlit as st
import os
import tempfile
from io import BytesIO
from utils.document_processor import extract_text_from_file
from utils.llm_processor import summarize_document, check_ollama_status, get_available_models
from functools import partial

def document_summarization_ui():
    """
    Streamlit UI for document summarization component.
    """
    st.header("📄 Document Summarization")
    st.markdown("""
        Upload a legal document (PDF, DOCX, or TXT) to generate an AI-powered summary.
        The summary will highlight key points, legal implications, and important details.
    """)
    
    # Check if Ollama is running
    if not check_ollama_status():
        st.error("""
            ⚠️ Ollama is not running. Please start Ollama using the command:
            ```
            ollama serve
            ```
        """)
        return
    
    # Get available models
    models = get_available_models()
    if not models:
        models = ["llama2"]  # Default fallback
    
    # Model selection
    model_name = st.selectbox(
        "Select AI Model",
        options=models,
        index=0,
        help="Choose the Ollama model to use for summarization"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a document", 
        type=["pdf", "docx", "txt"],
        help="Upload a legal document for summarization"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "File type": uploaded_file.type
        }
        
        st.write("### File Details")
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")
        
        # Extract text from file
        st.write("### Processing")
        with st.spinner("Extracting text from document..."):
            try:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower().replace('.', '')
                file_content = uploaded_file.getvalue()
                
                # Create a BytesIO object for in-memory processing
                file_bytes = BytesIO(file_content)
                text = extract_text_from_file(file_bytes, file_extension)
                
                # Display text extraction success
                st.success(f"Successfully extracted {len(text)} characters from document")
                
                # Show a preview of the extracted text
                with st.expander("Preview Extracted Text"):
                    st.text_area("Document Text", text[:1000] + ("..." if len(text) > 1000 else ""), height=200)
            except Exception as e:
                st.error(f"Error extracting text: {str(e)}")
                return
        
        # Summarization options
        st.write("### Summarization Options")
        summary_length = st.select_slider(
            "Summary Length",
            options=["Concise", "Balanced", "Detailed"],
            value="Balanced",
            help="Choose how detailed the summary should be"
        )
        
        # Map summary length to model parameters (could be used for prompt engineering)
        summary_params = {
            "Concise": {"max_length": 1000},
            "Balanced": {"max_length": 2000},
            "Detailed": {"max_length": 3000}
        }
        
        # Generate summary
        if st.button("Generate Summary", key="generate_summary"):
            with st.spinner("Generating summary... This may take a while depending on document length and model..."):
                try:
                    # Adjust prompt based on summary length (this would be implemented in llm_processor.py)
                    summary = summarize_document(text, model_name=model_name)
                    
                    # Display summary
                    st.write("### Document Summary")
                    st.markdown(summary)
                    
                    # Option to download summary
                    summary_bytes = summary.encode()
                    st.download_button(
                        label="Download Summary",
                        data=summary_bytes,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_summary.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
    
    # Tips section
    with st.expander("Tips for Better Summaries"):
        st.markdown("""
            - **Clean Documents**: For best results, use documents with clear, machine-readable text
            - **Document Length**: The summarizer works best with documents under 50 pages
            - **Legal Terminology**: Documents with standard legal terminology work best
            - **Try Different Models**: Different Ollama models may give different results
        """)
    
    # Sample document option
    with st.expander("Don't have a document? Try a sample"):
        if st.button("Load Sample Document", key="load_sample"):
            # Load sample document text (could be stored in a separate file)
            sample_text = """
            CONFIDENTIALITY AGREEMENT
            
            This Confidentiality Agreement (the "Agreement") is entered into as of [Date] by and between [Party A], located at [Address] ("Disclosing Party"), and [Party B], located at [Address] ("Receiving Party").
            
            1. CONFIDENTIAL INFORMATION
            For purposes of this Agreement, "Confidential Information" shall mean any and all non-public information, including, without limitation, technical, developmental, marketing, sales, operating, performance, cost, know-how, business plans, business methods, and process information, disclosed to the Receiving Party.
            
            2. OBLIGATION TO MAINTAIN CONFIDENTIALITY
            The Receiving Party agrees to maintain the confidentiality of the Confidential Information and to not disclose any such information to third parties without prior written consent from the Disclosing Party.
            
            3. TERM
            The obligations under this Agreement shall remain in effect for a period of [Time Period] from the date of disclosure of the Confidential Information.
            
            IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the date first above written.
            """
            
            # Simulate a file upload
            st.session_state.sample_loaded = True
            st.session_state.sample_text = sample_text
            
            # Display text and summary generation button
            st.text_area("Sample Document", sample_text, height=200)
            
            if st.button("Generate Summary for Sample", key="generate_sample_summary"):
                with st.spinner("Generating summary..."):
                    try:
                        summary = summarize_document(sample_text, model_name=model_name)
                        st.write("### Sample Document Summary")
                        st.markdown(summary)
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")