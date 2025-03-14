"""
Document summarization component for Streamlit application.
"""

import streamlit as st
import os
import tempfile
from io import BytesIO
import hashlib
import time
import uuid
import threading
from datetime import datetime, timedelta
from utils.document_processor import extract_text_from_file
from utils.llm_processor import summarize_document, check_ollama_status, get_available_models
from functools import partial


import hashlib

def hash_text(text: str) -> str:
    """Generate a hash for text to use as a cache key"""
    return hashlib.md5(text.encode()).hexdigest()

def summarize_with_timeout(text, model_name, summary_length, status_callback=None, max_wait_time=180):
    """
    Run summarization with a timeout and status updates
    
    Args:
        text: Document text to summarize
        model_name: Name of the model to use
        summary_length: Length of summary
        status_callback: Callback to update UI
        max_wait_time: Maximum wait time in seconds
        
    Returns:
        Generated summary or error message
    """
    result = {"summary": "", "error": None, "completed": False}
    
    def summarize_task():
        try:
            if status_callback:
                status_callback(0.1, "Starting summarization...")
            
            # Split text into chunks if it's too long
            if len(text) > 4000:
                if status_callback:
                    status_callback(0.2, "Document is large. Breaking into chunks...")
                
            summary = summarize_document(text, model_name=model_name, summary_length=summary_length)
            result["summary"] = summary
            result["completed"] = True
            
            if status_callback:
                status_callback(1.0, "Summary complete!")
                
        except Exception as e:
            result["error"] = str(e)
            if status_callback:
                status_callback(1.0, f"Error: {str(e)}")
    
    # Start the task in a separate thread
    thread = threading.Thread(target=summarize_task)
    thread.daemon = True  # Allow the thread to be terminated when the main thread exits
    thread.start()
    
    # Wait for the thread to complete or timeout
    start_time = datetime.now()
    while thread.is_alive():
        # Check if we've exceeded the timeout
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed > max_wait_time:
            return {
                "summary": "",
                "error": f"Summarization timed out after {max_wait_time} seconds. The model may be taking too long to process this document.",
                "completed": False,
                "timeout": True
            }
        
        # Update progress based on elapsed time
        if status_callback and elapsed % 5 == 0:  # Update every 5 seconds
            progress = min(0.9, elapsed / max_wait_time)
            status_callback(progress, f"Processing... ({int(elapsed)}s elapsed)")
        
        time.sleep(1)
    
    return result

def document_summarization_ui():
    """
    Streamlit UI for document summarization component.
    """
    st.header("📄 Document Summarization")
    st.markdown("""
        Upload a legal document (PDF, DOCX, or TXT) to generate an AI-powered summary.
        The summary will highlight key points, legal implications, and important details.
    """)
    
    # Initialize session state variables
    if 'summary_in_progress' not in st.session_state:
        st.session_state.summary_in_progress = False
    
    if 'summary_result' not in st.session_state:
        st.session_state.summary_result = None
        
    if 'summary_error' not in st.session_state:
        st.session_state.summary_error = None
    
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
        
        # Cache key for document summarization
        cache_key = f"summary_{hash_text(text)}_{model_name}_{summary_length}"
        
        # Check if we have a cached result
        if cache_option and cache_key in st.session_state:
            # Display cached summary
            st.write("### Document Summary (Cached)")
            st.markdown(st.session_state[cache_key])
            
            # Option to download summary
            summary_bytes = st.session_state[cache_key].encode()
            st.download_button(
                label="Download Summary",
                data=summary_bytes,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_summary.txt",
                mime="text/plain"
            )
            
            # Option to regenerate
            if st.button("Regenerate Summary"):
                if cache_key in st.session_state:
                    del st.session_state[cache_key]
                st.rerun()
        
        # Check if summarization is already in progress
        elif st.session_state.summary_in_progress:
            # Show status
            st.info("Generating summary... Please wait. This may take several minutes for large documents.")
            st.warning("**Note:** The progress indicator doesn't update in real-time. The page will refresh when the summary is ready.")
            
            # Add explanatory text about the process
            st.markdown("""
            **What's happening?**
            - The document is being analyzed by the AI model
            - For large documents, this may take several minutes
            - The page will automatically update when the summary is complete
            - You can safely navigate to other tabs and come back
            """)
            
            # Add a refresh button
            if st.button("Check if summary is ready"):
                st.rerun()
            
            # Show a cancel button
            if st.button("Cancel summarization"):
                st.session_state.summary_in_progress = False
                st.session_state.summary_result = None
                st.session_state.summary_error = "Summarization cancelled by user"
                st.rerun()
        
        # Check if we have a result from a previous summarization
        elif st.session_state.summary_result is not None:
            # Display the summary
            st.write("### Document Summary")
            st.markdown(st.session_state.summary_result)
            
            # Save to cache
            if cache_option:
                st.session_state[cache_key] = st.session_state.summary_result
            
            # Option to download summary
            summary_bytes = st.session_state.summary_result.encode()
            st.download_button(
                label="Download Summary",
                data=summary_bytes,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_summary.txt",
                mime="text/plain"
            )
            
            # Clear button
            if st.button("Generate New Summary"):
                st.session_state.summary_result = None
                st.rerun()
        
        # Check if we have an error
        elif st.session_state.summary_error is not None:
            st.error(f"Error generating summary: {st.session_state.summary_error}")
            
            if st.button("Try Again"):
                st.session_state.summary_error = None
                st.rerun()
        
        # If nothing is in progress and no results yet, show the generate button
        else:
            if st.button("Generate Summary", key="generate_summary"):
                # Start the summarization process
                st.session_state.summary_in_progress = True
                st.session_state.summary_result = None
                st.session_state.summary_error = None
                
                # Create a synchronous function to generate the summary
                def generate_summary():
                    try:
                        # Generate the summary
                        summary = summarize_document(text, model_name=model_name, summary_length=summary_length)
                        
                        # Update session state with the result
                        # This is done only once at the end to avoid thread issues
                        st.session_state.summary_result = summary
                        st.session_state.summary_in_progress = False
                    except Exception as e:
                        # Update session state with the error
                        st.session_state.summary_error = str(e)
                        st.session_state.summary_in_progress = False
                
                # Start the summarization in a background thread
                thread = threading.Thread(target=generate_summary)
                thread.daemon = True
                thread.start()
                
                # Rerun to show the in-progress state
                st.rerun()
    
    # Tips section
    with st.expander("Tips for Better Summaries"):
        st.markdown("""
            - **Clean Documents**: For best results, use documents with clear, machine-readable text
            - **Document Length**: The summarizer works best with documents under 50 pages
            - **Small Models**: Try "deepseek-r1:1.5b" or other small models for faster results
            - **Legal Terminology**: Documents with standard legal terminology work best
            - **Summary Length**: "Concise" is fastest, "Detailed" is most comprehensive
            - **Use Cache**: Enable caching for faster results with previously seen documents
            - **Be Patient**: Summarization can take several minutes, especially for large documents
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