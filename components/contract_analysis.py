"""
Contract analysis component for Streamlit application.
"""

import streamlit as st
import os
import tempfile
import json
from io import BytesIO
from utils.document_processor import extract_text_from_file
from utils.llm_processor import analyze_contract, check_ollama_status, get_available_models
import pandas as pd

def contract_analysis_ui():
    """
    Streamlit UI for contract analysis component.
    """
    st.header("📝 Contract Analysis")
    st.markdown("""
        Upload a contract document (PDF, DOCX, or TXT) to generate an AI-powered analysis.
        The analysis will extract key information such as parties, obligations, terms, and risks.
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
        help="Choose the Ollama model to use for contract analysis",
        key="contract_model_select"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a contract", 
        type=["pdf", "docx", "txt"],
        help="Upload a contract document for analysis",
        key="contract_file_upload"
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
        with st.spinner("Extracting text from contract..."):
            try:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower().replace('.', '')
                file_content = uploaded_file.getvalue()
                
                # Create a BytesIO object for in-memory processing
                file_bytes = BytesIO(file_content)
                text = extract_text_from_file(file_bytes, file_extension)
                
                # Display text extraction success
                st.success(f"Successfully extracted {len(text)} characters from contract")
                
                # Show a preview of the extracted text
                with st.expander("Preview Extracted Text"):
                    st.text_area("Contract Text", text[:1000] + ("..." if len(text) > 1000 else ""), height=200, key="contract_text_preview")
            except Exception as e:
                st.error(f"Error extracting text: {str(e)}")
                return
        
        # Analysis options
        st.write("### Analysis Options")
        analysis_focus = st.multiselect(
            "Analysis Focus",
            options=["Parties", "Key Terms", "Obligations", "Termination Conditions", "Risks", "Recommendations"],
            default=["Parties", "Key Terms", "Obligations", "Risks"],
            help="Choose what aspects of the contract to focus on in the analysis"
        )
        
        # Generate analysis
        if st.button("Analyze Contract", key="analyze_contract"):
            with st.spinner("Analyzing contract... This may take a while depending on document length and model..."):
                try:
                    # Run the contract analysis
                    analysis_result = analyze_contract(text, model_name=model_name)
                    
                    # The analysis_result is expected to be a dictionary with analysis text
                    analysis_text = analysis_result.get('analysis', '')
                    
                    # Display analysis
                    st.write("### Contract Analysis Results")
                    
                    # Check if we have structured data
                    if "structured_data" in analysis_result:
                        data = analysis_result["structured_data"]
                        
                        # Create tabs for different sections
                        tabs = st.tabs(["Overview", "Parties & Terms", "Obligations", "Risks & Recommendations"])
                        
                        with tabs[0]:
                            st.write("### Contract Overview")
                            if "contract_type" in data:
                                st.write(f"**Contract Type:** {data['contract_type']}")
                            
                            # Display key dates if available
                            if "key_dates" in data and data["key_dates"]:
                                st.write("**Key Dates:**")
                                for date in data["key_dates"]:
                                    st.write(f"- {date}")
                            else:
                                st.write("**Key Dates:** None specified")
                        
                        with tabs[1]:
                            # Display parties
                            if "parties" in data and data["parties"]:
                                st.write("### Parties Involved")
                                for party in data["parties"]:
                                    st.write(f"- {party}")
                            
                            # Display key terms
                            if "key_terms" in data and data["key_terms"]:
                                st.write("### Key Terms and Conditions")
                                for term in data["key_terms"]:
                                    st.write(f"- {term}")
                            
                            # Display termination conditions
                            if "termination_conditions" in data and data["termination_conditions"]:
                                st.write("### Termination Conditions")
                                for condition in data["termination_conditions"]:
                                    st.write(f"- {condition}")
                        
                        with tabs[2]:
                            # Display obligations
                            if "obligations" in data and data["obligations"]:
                                st.write("### Obligations")
                                for obligation in data["obligations"]:
                                    st.write(f"- {obligation}")
                        
                        with tabs[3]:
                            # Display risks
                            if "risks" in data and data["risks"]:
                                st.write("### Potential Risks")
                                for risk in data["risks"]:
                                    st.write(f"- {risk}")
                            
                            # Display recommendations
                            if "recommendations" in data and data["recommendations"]:
                                st.write("### Recommendations")
                                for rec in data["recommendations"]:
                                    st.write(f"- {rec}")
                    else:
                        # If no structured data, display raw text
                        st.markdown(analysis_text)
                    
                    # Option to download analysis
                    analysis_bytes = json.dumps(analysis_result.get("structured_data", {}), indent=2).encode() if "structured_data" in analysis_result else analysis_text.encode()
                    st.download_button(
                        label="Download Analysis",
                        data=analysis_bytes,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_analysis.json" if "structured_data" in analysis_result else f"{os.path.splitext(uploaded_file.name)[0]}_analysis.txt",
                        mime="application/json" if "structured_data" in analysis_result else "text/plain",
                        key="download_analysis"
                    )
                except Exception as e:
                    st.error(f"Error analyzing contract: {str(e)}")
    
    # Tips section
    with st.expander("Tips for Better Contract Analysis"):
        st.markdown("""
            - **Use Clean Documents**: For best results, use clean, machine-readable contracts
            - **Contract Structure**: Well-structured contracts with clear sections work best
            - **Legal Language**: Standard legal terminology will be better recognized
            - **Contract Types**: The system works well with NDAs, service agreements, employment contracts, etc.
        """)
    
    # Sample contract option
    with st.expander("Don't have a contract? Try a sample"):
        if st.button("Load Sample Contract", key="load_sample_contract"):
            # Load sample contract text
            sample_contract = """
            EMPLOYMENT AGREEMENT
            
            This Employment Agreement (the "Agreement") is made and entered into as of January 1, 2025, by and between ABC Corp., a Delaware corporation (the "Company"), and John Doe (the "Employee").
            
            1. POSITION AND DUTIES
            The Company employs the Employee as Chief Technology Officer, and the Employee accepts such employment. The Employee shall perform the duties and responsibilities of a Chief Technology Officer to the best of Employee's abilities.
            
            2. TERM
            The term of this Agreement shall commence on January 1, 2025, and shall continue for a period of three (3) years thereafter, unless earlier terminated as provided herein.
            
            3. COMPENSATION
            Base Salary. The Company shall pay to the Employee a base salary of $200,000 per year, payable in accordance with the Company's regular payroll practices.
            Bonus. The Employee shall be eligible for an annual bonus of up to 30% of base salary based on performance metrics established by the Board of Directors.
            Stock Options. The Employee shall receive options to purchase 50,000 shares of the Company's common stock, vesting over a period of four years.
            
            4. TERMINATION
            The Company may terminate this Agreement for cause, including but not limited to: (a) material breach of this Agreement; (b) Employee's conviction of a felony; or (c) Employee's willful misconduct that has a material adverse effect on the Company.
            
            5. NON-DISCLOSURE AND NON-COMPETITION
            The Employee agrees to not disclose confidential information during or after employment and to not compete with the Company for a period of one year following termination.
            
            IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.
            """
            
            # Simulate a file upload
            st.session_state.sample_contract_loaded = True
            st.session_state.sample_contract_text = sample_contract
            
            # Display text and analysis generation button
            st.text_area("Sample Contract", sample_contract, height=200, key="sample_contract_text_area")
            
            if st.button("Analyze Sample Contract", key="analyze_sample_contract"):
                with st.spinner("Analyzing sample contract..."):
                    try:
                        analysis_result = analyze_contract(sample_contract, model_name=model_name)
                        analysis_text = analysis_result.get('analysis', '')
                        
                        st.write("### Sample Contract Analysis Results")
                        st.markdown(analysis_text)
                    except Exception as e:
                        st.error(f"Error analyzing sample contract: {str(e)}")