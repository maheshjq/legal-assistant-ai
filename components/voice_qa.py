"""
Voice-commanded question answering component for Streamlit application.
"""

import streamlit as st
import os
import tempfile
import time
from io import BytesIO
from utils.document_processor import extract_text_from_file
from utils.audio_processor import record_and_transcribe, record_audio, save_audio_to_temp_file, transcribe_audio_file
from utils.llm_processor import answer_question, check_ollama_status, get_available_models
import numpy as np

def voice_qa_ui():
    """
    Streamlit UI for voice-commanded question answering component.
    """
    st.header("🎤 Voice Q&A")
    st.markdown("""
        Upload a document and ask questions about it using your voice.
        The AI will transcribe your voice and answer questions based on the document content.
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
        help="Choose the Ollama model to use for answering questions",
        key="qa_model_select"
    )
    
    # Initialize session state for document text
    if "document_text" not in st.session_state:
        st.session_state.document_text = ""
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a document", 
        type=["pdf", "docx", "txt"],
        help="Upload a document to ask questions about",
        key="qa_file_upload"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "File type": uploaded_file.type
        }
        
        st.write("### Document Details")
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")
        
        # Extract text from file
        with st.spinner("Extracting text from document..."):
            try:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower().replace('.', '')
                file_content = uploaded_file.getvalue()
                
                # Create a BytesIO object for in-memory processing
                file_bytes = BytesIO(file_content)
                text = extract_text_from_file(file_bytes, file_extension)
                
                # Store document text in session state
                st.session_state.document_text = text
                
                # Display text extraction success
                st.success(f"Successfully extracted {len(text)} characters from document")
                
                # Show a preview of the extracted text
                with st.expander("Preview Document Text"):
                    st.text_area("Document Text", text[:1000] + ("..." if len(text) > 1000 else ""), height=200, key="qa_text_preview")
            except Exception as e:
                st.error(f"Error extracting text: {str(e)}")
                return
    
    # Document content check
    if not st.session_state.document_text:
        st.warning("Please upload a document first or use the sample document.")
        
        # Sample document option
        with st.expander("Use a sample document instead"):
            if st.button("Load Sample Document", key="load_qa_sample"):
                # Load sample document text
                sample_text = """
                LEGAL MEMORANDUM
                
                TO: Senior Partners
                FROM: Legal Research Team
                DATE: March 10, 2025
                RE: Copyright Infringement in Digital Media
                
                QUESTION PRESENTED
                
                Under current U.S. copyright law, to what extent are content creators liable for copyright infringement when they use portions of copyrighted material in their digital content under the fair use doctrine?
                
                SHORT ANSWER
                
                Content creators may use portions of copyrighted material without permission if their use qualifies as "fair use." The determination of fair use depends on four factors: (1) the purpose and character of the use, (2) the nature of the copyrighted work, (3) the amount and substantiality of the portion used, and (4) the effect on the potential market for the copyrighted work. Transformative uses that add new meaning, expression, or message are more likely to be considered fair use, while commercial uses that simply substitute for the original work are less likely to qualify.
                
                FACTS
                
                Digital content creation has exploded in recent years, with millions of creators publishing videos, music remixes, commentary, and other content online. Many creators incorporate clips, images, or portions of copyrighted works into their content, raising questions about copyright infringement and fair use. Our client, a digital media company that hosts user-generated content, seeks guidance on the current state of fair use law to develop content guidelines for creators.
                
                DISCUSSION
                
                I. Fair Use Doctrine Overview
                
                The fair use doctrine, codified in 17 U.S.C. § 107, provides an affirmative defense to copyright infringement. It permits the unlicensed use of copyright-protected works in certain circumstances, particularly for purposes such as criticism, comment, news reporting, teaching, scholarship, or research.
                
                Courts evaluate fair use claims on a case-by-case basis, considering four factors:
                
                1. Purpose and character of the use
                2. Nature of the copyrighted work
                3. Amount and substantiality of the portion used
                4. Effect on the potential market for the copyrighted work
                
                II. Transformative Use
                
                In Campbell v. Acuff-Rose Music, Inc., 510 U.S. 569 (1994), the Supreme Court emphasized the importance of "transformative use" in fair use analysis. A use is transformative if it adds something new, with a further purpose or different character, altering the first with new expression, meaning, or message.
                
                Digital content that meaningfully transforms copyrighted material is more likely to be protected. For example, in Bill Graham Archives v. Dorling Kindersley Ltd., 448 F.3d 605 (2d Cir. 2006), the court found that using concert posters in a biographical book was transformative because the posters were used for historical purposes rather than for their original promotional purposes.
                
                III. Amount and Substantiality
                
                Courts consider both the quantity and quality of the copyrighted material used. In Harper & Row Publishers, Inc. v. Nation Enterprises, 471 U.S. 539 (1985), the Supreme Court found that using just 300 words from a memoir constituted copyright infringement because those words represented the "heart" of the work.
                
                For digital content creators, using brief clips or portions may generally be safer than using extensive portions of copyrighted works, but even small portions can be problematic if they constitute the "heart" of the original work.
                
                IV. Recent Digital Media Cases
                
                In recent years, courts have addressed fair use in the digital context. In Google LLC v. Oracle America, Inc., 141 S. Ct. 1183 (2021), the Supreme Court found that Google's use of Oracle's Java API code was fair use, emphasizing the transformative nature of Google's implementation and the public benefit of allowing programmers to use familiar code.
                
                For content creators, cases like Lenz v. Universal Music Corp., 815 F.3d 1145 (9th Cir. 2016), are instructive. The court held that copyright holders must consider fair use before issuing takedown notices for online content, recognizing that many uses of copyrighted material in user-generated content may be protected.
                
                CONCLUSION
                
                Content creators may incorporate portions of copyrighted materials into their digital content if such use qualifies as fair use under the four-factor test. Transformative uses that add new meaning or context are more likely to be protected, while uses that simply republish copyrighted content without adding new expression are more vulnerable to infringement claims.
                
                Our client should advise content creators to consider the following guidelines:
                
                1. Use only the amount of copyrighted material necessary to achieve the intended purpose
                2. Add substantial original commentary, criticism, or other transformative elements
                3. Avoid uses that would compete with or substitute for the original work in the marketplace
                4. Consider the nature of the copyrighted work, with factual works receiving less protection than highly creative ones
                
                Even with these guidelines, fair use determinations remain highly fact-specific, and there is inherent uncertainty in predicting how courts would rule in any particular case.
                """
                
                # Store document text in session state
                st.session_state.document_text = sample_text
                
                # Show a preview of the sample document
                with st.expander("Preview Sample Document"):
                    st.text_area("Sample Document", sample_text[:1000] + "...", height=200, key="sample_qa_text_preview")
                
                st.success("Sample document loaded! You can now ask questions about it.")
    
    # Voice recording and question answering
    if st.session_state.document_text:
        st.write("### Ask a Question")
        st.markdown("""
            You can either:
            1. Record a voice question using the microphone
            2. Type your question in the text box below
        """)
        
        # Voice recording
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎤 Record Voice Question", key="record_voice"):
                with st.spinner("Recording... Speak now"):
                    # Record audio
                    try:
                        audio_data, sample_rate = record_audio(duration=5)  # 5 seconds recording
                        st.session_state.audio_data = audio_data
                        st.session_state.sample_rate = sample_rate
                        
                        # Save audio to temp file
                        temp_file = save_audio_to_temp_file(audio_data, sample_rate)
                        
                        # Transcribe audio
                        with st.spinner("Transcribing..."):
                            result = transcribe_audio_file(temp_file)
                            transcribed_text = result.get("text", "")
                            
                            if transcribed_text:
                                st.success("Transcription successful!")
                                st.session_state.transcribed_text = transcribed_text
                            else:
                                st.error("Failed to transcribe audio. Please try again or type your question.")
                        
                        # Delete temp file
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                    except Exception as e:
                        st.error(f"Error recording audio: {str(e)}")
        
        # Display transcribed text if available
        if "transcribed_text" in st.session_state:
            st.write("### Transcribed Question")
            st.write(st.session_state.transcribed_text)
            
            # Update text input with transcribed text
            if "question_text" not in st.session_state:
                st.session_state.question_text = st.session_state.transcribed_text
        
        # Text input as alternative to voice
        question_text = st.text_input(
            "Or type your question here",
            value=st.session_state.get("question_text", ""),
            key="qa_text_input"
        )
        
        # Answer button
        if st.button("Get Answer", key="get_answer"):
            question = question_text or st.session_state.get("transcribed_text", "")
            
            if not question:
                st.warning("Please provide a question either by voice or text.")
            else:
                with st.spinner("Finding answer..."):
                    try:
                        # Get answer from LLM
                        answer = answer_question(question, st.session_state.document_text, model_name=model_name)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({"question": question, "answer": answer})
                        
                        # Clear question input and transcribed text for next question
                        st.session_state.question_text = ""
                        if "transcribed_text" in st.session_state:
                            del st.session_state.transcribed_text
                        
                        # Force a rerun to update UI
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error getting answer: {str(e)}")
        
        # Display chat history
        if st.session_state.chat_history:
            st.write("### Q&A History")
            for i, qa in enumerate(st.session_state.chat_history):
                st.markdown(f"**Question {i+1}:** {qa['question']}")
                st.markdown(f"**Answer {i+1}:** {qa['answer']}")
                st.markdown("---")
    
    # Tips section
    with st.expander("Tips for Better Voice Questions"):
        st.markdown("""
            - **Speak Clearly**: Enunciate your words clearly for better transcription
            - **Specific Questions**: Ask specific questions rather than general ones
            - **Mention Keywords**: Include key terms from the document in your question
            - **Quiet Environment**: Record in a quiet environment for better audio quality
            - **Adjust Recording Time**: If your questions are being cut off, try typing them instead
        """)