"""
Contract analysis component for Streamlit application.
"""

import streamlit as st
import os
import tempfile
import json
import re
from io import BytesIO
from utils.document_processor import extract_text_from_file
from utils.llm_processor import analyze_contract, check_ollama_status, get_available_models
import pandas as pd

def preprocess_legal_document(text: str) -> str:
    """
    Preprocess legal document text to improve analysis quality
    
    Args:
        text: Raw text extracted from a document
        
    Returns:
        str: Preprocessed text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers (common pattern in legal docs)
    text = re.sub(r'\s+\d+\s+', ' ', text)
    
    # Remove common header/footer patterns
    text = re.sub(r'Page \d+ of \d+', '', text)
    
    # Add proper spacing after full stops
    text = re.sub(r'\.(?=[A-Z])', '. ', text)
    
    # Try to identify section breaks
    text = re.sub(r'([A-Z][A-Z\s]+:)', r'\n\n\1', text)
    
    return text

def contract_analysis_ui():
    """
    Streamlit UI for contract analysis component.
    """
    st.header("📝 Document Analysis")
    st.markdown("""
        Upload a legal document (PDF, DOCX, or TXT) to generate an AI-powered analysis.
        This tool works with various legal documents including:
        - Contracts and agreements
        - International treaties
        - Legal handbooks and guides
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
        help="Choose the Ollama model to use for document analysis",
        key="contract_model_select"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a document", 
        type=["pdf", "docx", "txt"],
        help="Upload a legal document for analysis",
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
        with st.spinner("Extracting text from document..."):
            try:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower().replace('.', '')
                file_content = uploaded_file.getvalue()
                
                # Create a BytesIO object for in-memory processing
                file_bytes = BytesIO(file_content)
                text = extract_text_from_file(file_bytes, file_extension)
                
                # Apply preprocessing to improve analysis quality
                text = preprocess_legal_document(text)
                
                # Display text extraction success
                st.success(f"Successfully extracted {len(text)} characters from document")
                
                # Show a preview of the extracted text
                with st.expander("Preview Extracted Text"):
                    st.text_area("Document Text", text[:1000] + ("..." if len(text) > 1000 else ""), height=200, key="contract_text_preview")
            except Exception as e:
                st.error(f"Error extracting text: {str(e)}")
                return
        
        # Analysis options
        st.write("### Analysis Options")
        
        document_type = st.selectbox(
            "Document Type",
            options=["Automatic Detection", "Contract/Agreement", "Treaty/International Agreement", "Legal Handbook/Guide"],
            index=0,
            help="Select the type of document for more accurate analysis"
        )
        
        analysis_focus = st.multiselect(
            "Analysis Focus",
            options=["Parties", "Key Terms", "Obligations", "Provisions", "Benefits", "Restrictions"],
            default=["Parties", "Key Terms", "Obligations", "Provisions"],
            help="Choose what aspects of the document to focus on in the analysis"
        )
        
        # Generate analysis
        if st.button("Analyze Document", key="analyze_contract"):
            with st.spinner("Analyzing document... This may take a while depending on document length and model..."):
                try:
                    # Run the document analysis
                    analysis_result = analyze_contract(text, model_name=model_name)
                    
                    # The analysis_result is expected to be a dictionary with analysis text
                    analysis_text = analysis_result.get('analysis', '')
                    
                    # Display analysis
                    st.write("### Document Analysis Results")
                    
                    # Check if we have structured data
                    if "structured_data" in analysis_result:
                        data = analysis_result["structured_data"]
                        
                        # Create tabs for different sections
                        tabs = st.tabs(["Overview", "Key Information", "Provisions", "Obligations & Restrictions"])
                        
                        with tabs[0]:
                            st.write("### Document Overview")
                            if "document_type" in data:
                                st.write(f"**Document Type:** {data['document_type']}")
                            
                            if "title" in data:
                                st.write(f"**Title:** {data['title']}")
                            
                            # Display effective date if available
                            if "effective_date" in data and data["effective_date"]:
                                st.write(f"**Effective Date:** {data['effective_date']}")
                        
                        with tabs[1]:
                            # Display parties
                            if "parties" in data and data["parties"]:
                                st.write("### Parties Involved")
                                for party in data["parties"]:
                                    st.write(f"- {party}")
                            
                            # Display key sections
                            if "key_sections" in data and data["key_sections"]:
                                st.write("### Key Sections")
                                for section in data["key_sections"]:
                                    st.write(f"- {section}")
                            
                            # Display key terms if available
                            if "key_terms" in data and data["key_terms"]:
                                st.write("### Key Terms")
                                for term in data["key_terms"]:
                                    st.write(f"- {term}")
                        
                        with tabs[2]:
                            # Display key provisions
                            if "key_provisions" in data and data["key_provisions"]:
                                st.write("### Key Provisions")
                                for provision in data["key_provisions"]:
                                    st.write(f"- {provision}")
                            
                            # Display termination conditions
                            if "termination_conditions" in data and data["termination_conditions"]:
                                st.write("### Termination Conditions")
                                for condition in data["termination_conditions"]:
                                    st.write(f"- {condition}")
                        
                        with tabs[3]:
                            # Display obligations
                            if "obligations" in data and data["obligations"]:
                                st.write("### Obligations")
                                for obligation in data["obligations"]:
                                    st.write(f"- {obligation}")
                            
                            # Display benefits
                            if "benefits" in data and data["benefits"]:
                                st.write("### Benefits")
                                for benefit in data["benefits"]:
                                    st.write(f"- {benefit}")
                            
                            # Display restrictions
                            if "restrictions" in data and data["restrictions"]:
                                st.write("### Restrictions")
                                for restriction in data["restrictions"]:
                                    st.write(f"- {restriction}")
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
                    st.error(f"Error analyzing document: {str(e)}")
    
    # Tips section
    with st.expander("Tips for Better Document Analysis"):
        st.markdown("""
            - **Use Clean Documents**: For best results, use clean, machine-readable documents
            - **Document Structure**: Well-structured documents with clear sections work best
            - **Document Type**: Select the correct document type for more accurate analysis
            - **Legal Language**: Standard legal terminology will be better recognized
            - **Try Different Models**: Different models may perform better on different types of documents
        """)
    
    # Sample document option
    with st.expander("Don't have a document? Try a sample"):
        sample_type = st.selectbox(
            "Sample Document Type",
            options=["Contract", "International Agreement"],
            index=0,
            key="sample_document_type"
        )
        
        if st.button("Load Sample Document", key="load_sample_document"):
            # Load sample document based on selected type
            if sample_type == "Contract":
                # Sample contract text
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
                st.session_state.sample_contract_loaded = True
                st.session_state.sample_contract_text = sample_contract
                
                # Display text and analysis generation button
                st.text_area("Sample Contract", sample_contract, height=200, key="sample_contract_text_area")
            else:
                # Sample international agreement text
                sample_agreement = """
                TRADE COOPERATION AGREEMENT
                
                This Trade Cooperation Agreement (hereinafter referred to as "the Agreement") is made and entered into on March 15, 2025, by and between:
                
                The Government of Country A, represented by its Ministry of International Trade and Commerce, and
                
                The Government of Country B, represented by its Department of Foreign Trade Relations,
                
                (hereinafter collectively referred to as "the Parties")
                
                ARTICLE I: PURPOSE AND OBJECTIVES
                
                1.1 The Parties hereby establish a framework for trade cooperation with the objective of enhancing bilateral trade relations.
                
                1.2 The Parties shall work towards the elimination of unnecessary barriers to trade, facilitation of cross-border business activities, and promotion of economic growth in both countries.
                
                ARTICLE II: TARIFF REDUCTION
                
                2.1 The Parties agree to reduce tariffs on selected goods as specified in Annex A of this Agreement.
                
                2.2 The tariff reduction schedule shall be implemented over a period of five (5) years from the effective date of this Agreement.
                
                ARTICLE III: RULES OF ORIGIN
                
                3.1 To qualify for preferential treatment under this Agreement, goods must meet the Rules of Origin criteria specified in Annex B.
                
                3.2 The Parties shall establish a joint committee to oversee the implementation of Rules of Origin.
                
                ARTICLE IV: TERM AND TERMINATION
                
                4.1 This Agreement shall enter into force thirty (30) days after both Parties have notified each other, through diplomatic channels, of the completion of their respective internal legal procedures necessary for the implementation of this Agreement.
                
                4.2 This Agreement shall remain in effect for an initial period of ten (10) years and shall be automatically renewed for successive five-year periods unless either Party provides written notice of termination at least six (6) months prior to the expiration of the current term.
                
                ARTICLE V: DISPUTE RESOLUTION
                
                5.1 Any dispute arising from the interpretation or implementation of this Agreement shall be resolved through friendly consultations between the Parties.
                
                5.2 If the dispute cannot be resolved through consultations within ninety (90) days, either Party may request the establishment of an arbitration panel.
                
                IN WITNESS WHEREOF, the undersigned, being duly authorized by their respective Governments, have signed this Agreement.
                
                DONE in duplicate at [City], on this 15th day of March, 2025, in the English language, both texts being equally authentic.
                
                For the Government of Country A:           For the Government of Country B:
                ___________________________           ___________________________
                Minister of International Trade         Secretary of Foreign Trade Relations
                """
                st.session_state.sample_agreement_loaded = True
                st.session_state.sample_agreement_text = sample_agreement
                
                # Display text and analysis generation button
                st.text_area("Sample International Agreement", sample_agreement, height=200, key="sample_agreement_text_area")
            
            if st.button("Analyze Sample Document", key="analyze_sample_document"):
                with st.spinner("Analyzing sample document..."):
                    try:
                        sample_text = st.session_state.get('sample_contract_text', st.session_state.get('sample_agreement_text', ''))
                        analysis_result = analyze_contract(sample_text, model_name=model_name)
                        
                        st.write("### Sample Document Analysis Results")
                        
                        if "structured_data" in analysis_result:
                            data = analysis_result["structured_data"]
                            
                            # Display first few elements of the analysis in a more condensed format
                            st.json(json.dumps(data, indent=2))
                        else:
                            st.markdown(analysis_result.get('analysis', ''))
                    except Exception as e:
                        st.error(f"Error analyzing sample document: {str(e)}")