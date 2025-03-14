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

from utils.background_processor import task_manager, summarize_document_with_progress
import time
import uuid

def document_summarization_ui():
    """
    Streamlit UI for document summarization component with background processing.
    """
    st.header("📄 Document Summarization")
    st.markdown("""
        Upload a legal document (PDF, DOCX, or TXT) to generate an AI-powered summary.
        The summary will highlight key points, legal implications, and important details.
    """)
    
    # Initialize session state
    if 'summary_task_id' not in st.session_state:
        st.session_state.summary_task_id = None
    
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
    col1, col2 = st.columns([2, 1])
    with col1:
        model_name = st.selectbox(
            "Select AI Model",
            options=models,
            index=0,
            help="Choose the Ollama model to use for summarization"
        )
    
    with col2:
        cache_option = st.checkbox("Use cache (faster)", value=True, 
                                  help="Cache results for faster processing of previously seen documents")
    
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
        
        # Check if a task is already running
        if st.session_state.summary_task_id:
            task_status = task_manager.get_task_status(st.session_state.summary_task_id)
            
            if task_status['status'] == 'running':
                # Show progress bar for running task
                st.info(f"Generating summary... {task_status['message']}")
                st.progress(task_status['progress'])
                
                # Add a button to check status
                if st.button("Check Progress"):
                    st.rerun()
            
            elif task_status['status'] == 'completed':
                # Show completed summary
                st.success("Summary generation completed!")
                summary = task_status['result']
                
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
                
                # Button to start a new summary
                if st.button("Generate New Summary"):
                    st.session_state.summary_task_id = None
                    st.rerun()
            
            elif task_status['status'] == 'error':
                # Show error message
                st.error(f"Error generating summary: {task_status['error']}")
                
                # Button to try again
                if st.button("Try Again"):
                    st.session_state.summary_task_id = None
                    st.rerun()
        
        else:
            # Generate summary button
            if st.button("Generate Summary", key="generate_summary"):
                # Cache key for document summarization
                cache_key = f"summary_{hash_text(text)}_{model_name}_{summary_length}"
                
                if cache_option and cache_key in st.session_state:
                    # Use cached result
                    summary = st.session_state[cache_key]
                    
                    st.write("### Document Summary (Cached)")
                    st.markdown(summary)
                    
                    # Option to download summary
                    summary_bytes = summary.encode()
                    st.download_button(
                        label="Download Summary",
                        data=summary_bytes,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_summary.txt",
                        mime="text/plain"
                    )
                else:
                    # Start background task
                    task_id = f"summary_{uuid.uuid4().hex}"
                    st.session_state.summary_task_id = task_id
                    
                    task_manager.start_task(
                        task_id=task_id,
                        func=summarize_document_with_progress,
                        kwargs={
                            'text': text,
                            'model_name': model_name,
                            'summary_length': summary_length
                        }
                    )
                    
                    # Show initial progress
                    st.info("Started summary generation in the background...")
                    st.progress(0)
                    
                    # Rerun to update UI
                    time.sleep(0.5)
                    st.rerun()
    
    # Tips section
    with st.expander("Tips for Better Summaries"):
        st.markdown("""
            - **Clean Documents**: For best results, use documents with clear, machine-readable text
            - **Document Length**: The summarizer works best with documents under 50 pages
            - **Legal Terminology**: Documents with standard legal terminology work best
            - **Try Different Models**: Different Ollama models may give different results
            - **Summary Length**: "Concise" is fastest, "Detailed" is most comprehensive
            - **Use Cache**: Enable caching for faster results with previously seen documents
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