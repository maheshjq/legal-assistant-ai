"""
LLM processing utilities for Ollama and LangChain integration.
"""

import requests
from langchain.llms import Ollama
from langchain.chains import ConversationChain, AnalyzeDocumentChain, LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader
# from langchain_community.document_loaders.base import Document
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any, Optional, Union, Callable

def check_ollama_status(url: str = "http://localhost:11434") -> bool:
    """
    Check if Ollama is running
    
    Args:
        url: Ollama API URL
        
    Returns:
        bool: True if Ollama is running, False otherwise
    """
    try:
        response = requests.get(f"{url}/api/tags")
        return response.status_code == 200
    except:
        return False

def get_available_models(url: str = "http://localhost:11434") -> List[str]:
    """
    Get list of available models from Ollama
    
    Args:
        url: Ollama API URL
        
    Returns:
        List[str]: List of available model names
    """
    try:
        response = requests.get(f"{url}/api/tags")
        data = response.json()
        
        # Extract model names
        return [model["name"] for model in data.get("models", [])]
    except:
        return []

def create_ollama_llm(model_name: str = "llama2") -> Ollama:
    """
    Create Ollama LLM instance
    
    Args:
        model_name: Name of the Ollama model to use
        
    Returns:
        Ollama: LangChain Ollama LLM instance
    """
    return Ollama(model=model_name)

def create_document_from_text(text: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
    """
    Create a LangChain Document from text
    
    Args:
        text: The text content
        metadata: Optional metadata for the document
        
    Returns:
        Document: LangChain Document instance
    """
    if metadata is None:
        metadata = {}
    
    return Document(page_content=text, metadata=metadata)

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into smaller chunks
    
    Args:
        documents: List of LangChain Document objects
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List[Document]: List of split documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return text_splitter.split_documents(documents)

def create_vector_index(documents: List[Document], model_name: str = "llama2") -> FAISS:
    """
    Create a vector index from documents
    
    Args:
        documents: List of LangChain Document objects
        model_name: Name of the Ollama model to use for embeddings
        
    Returns:
        FAISS: Vector index for document retrieval
    """
    embeddings = OllamaEmbeddings(model=model_name)
    return FAISS.from_documents(documents, embeddings)

def summarize_document(text: str, model_name: str = "llama2") -> str:
    """
    Summarize a document using LangChain and Ollama
    
    Args:
        text: Document text to summarize
        model_name: Name of the Ollama model to use
        
    Returns:
        str: Generated summary
    """
    llm = create_ollama_llm(model_name)
    
    # Create a summarization chain
    prompt_template = """
    You are a legal document summarizer. Summarize the following document in a comprehensive way,
    highlighting the key points, legal implications, and important details.
    
    Document:
    {text}
    
    Summary:
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    summarize_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Split the document if it's too long
    if len(text) > 4000:
        # Create multiple summaries and then combine them
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
        texts = text_splitter.split_text(text)
        
        summaries = []
        for chunk in texts:
            summaries.append(summarize_chain.run(text=chunk))
        
        # Combine the summaries
        combined_summary = "\n\n".join(summaries)
        
        # Create a final summary of the combined summaries
        final_summary = summarize_chain.run(text=combined_summary)
        return final_summary
    else:
        return summarize_chain.run(text=text)

# utils/llm_processor.py - update the analyze_contract function

import json
import re

def analyze_contract(text: str, model_name: str = "llama2", **kwargs) -> Dict[str, Any]:
    """
    Analyze a contract document
    
    Args:
        text: Contract text to analyze
        model_name: Name of the Ollama model to use
        **kwargs: Additional parameters for analysis customization
        
    Returns:
        Dict: Analysis results including parties, key terms, obligations, etc.
    """
    llm = create_ollama_llm(model_name)
    
    # Determine document type from kwargs or try to infer it
    document_type = kwargs.get('document_type', 'Automatic Detection')
    focus_areas = kwargs.get('focus_areas', ['Parties', 'Key Terms', 'Obligations', 'Provisions'])
    
    # Create a more targeted prompt based on document type
    if document_type == "Treaty/International Agreement":
        prompt_template = """
        You are a legal analyst specializing in international agreements and treaties. Analyze the following document and extract key information.
        
        Document:
        {text}
        
        Extract all relevant information about this international agreement or treaty.
        
        Provide a JSON object with the following structure:
        {{
            "document_type": "Type of document (e.g., Trade Agreement, Treaty, Convention, Memorandum of Understanding)",
            "title": "Full title of the document",
            "parties": ["Country/Entity 1", "Country/Entity 2", ...],
            "effective_date": "When the agreement came into force",
            "key_sections": ["Section 1: description", "Section 2: description", ...],
            "key_provisions": ["Provision 1", "Provision 2", ...],
            "obligations": ["Obligation 1", "Obligation 2", ...],
            "termination_conditions": ["Condition 1", "Condition 2", ...],
            "benefits": ["Benefit 1", "Benefit 2", ...],
            "restrictions": ["Restriction 1", "Restriction 2", ...]
        }}
        
        Extract REAL information from the document. DO NOT use placeholder text like "Date 1: description". 
        If you cannot find specific information, leave that field as an empty array or null.
        """
    elif document_type == "Legal Handbook/Guide":
        prompt_template = """
        You are a legal analyst specializing in legal handbooks and guides. Analyze the following document and extract key information.
        
        Document:
        {text}
        
        Extract all relevant information about this legal handbook or guide.
        
        Provide a JSON object with the following structure:
        {{
            "document_type": "Type of document (e.g., Handbook, Guide, Manual)",
            "title": "Full title of the document",
            "parties": ["Organization/Author", "Target Audience", ...],
            "publication_date": "When the document was published",
            "key_sections": ["Section 1: description", "Section 2: description", ...],
            "key_topics": ["Topic 1", "Topic 2", ...],
            "key_provisions": ["Provision 1", "Provision 2", ...],
            "important_definitions": ["Definition 1", "Definition 2", ...],
            "notable_points": ["Point 1", "Point 2", ...]
        }}
        
        Extract REAL information from the document. DO NOT use placeholder text like "Topic 1". 
        If you cannot find specific information, leave that field as an empty array or null.
        """
    else:  # Default for contracts or automatic detection
        prompt_template = """
        You are a legal document analyst. Analyze the following document and extract key information.
        
        Document:
        {text}
        
        First, determine what type of legal document this is (contract, agreement, treaty, handbook, etc.).
        Then extract all relevant information based on the document type.
        
        Provide a JSON object with the following structure:
        {{
            "document_type": "Type of document (e.g., Contract, Agreement, Treaty, Handbook)",
            "title": "Full title of the document",
            "parties": ["Party 1", "Party 2", ...],
            "effective_date": "When the document came into force (if applicable)",
            "key_sections": ["Section 1: description", "Section 2: description", ...],
            "key_terms": ["Term 1", "Term 2", ...],
            "key_provisions": ["Provision 1", "Provision 2", ...],
            "obligations": ["Obligation 1", "Obligation 2", ...],
            "termination_conditions": ["Condition 1", "Condition 2", ...],
            "benefits": ["Benefit 1", "Benefit 2", ...],
            "restrictions": ["Restriction 1", "Restriction 2", ...]
        }}
        
        Extract REAL information from the document. DO NOT use placeholder text like "Party 1". 
        If you cannot find specific information, leave that field as an empty array or null.
        """
    
    prompt = PromptTemplate.from_template(prompt_template)
    analysis_chain = LLMChain(llm=llm, prompt=prompt)
    
    # For long documents, split into chunks and analyze separately
    if len(text) > 6000:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=500)
        chunks = text_splitter.split_text(text)
        
        # Get first chunk for basic document info and last chunk for conclusions
        first_chunk = chunks[0]
        last_chunk = chunks[-1]
        
        # Get a middle chunk to ensure comprehensive coverage
        middle_chunk = chunks[len(chunks)//2] if len(chunks) > 2 else ""
        
        # Combine key chunks with document summary
        summary = f"""
        FIRST PART:
        {first_chunk}
        
        MIDDLE PART:
        {middle_chunk}
        
        LAST PART:
        {last_chunk}
        
        This is a partial extraction from a longer document. Focus on identifying the document type,
        parties involved, key provisions, and overall purpose.
        """
        
        result = analysis_chain.run(text=summary)
    else:
        result = analysis_chain.run(text=text)
    
    # Parse the JSON response
    try:
        # Find JSON content in the response (if there's any text before/after JSON)
        json_pattern = r'(?s)\{.*\}'
        json_match = re.search(json_pattern, result)
        
        if json_match:
            json_text = json_match.group(0)
            try:
                structured_data = json.loads(json_text)
                return {"analysis": result, "structured_data": structured_data}
            except json.JSONDecodeError:
                # Try to clean the JSON string by removing any markdown code block markers
                cleaned_json = json_text.replace("```json", "").replace("```", "").strip()
                structured_data = json.loads(cleaned_json)
                return {"analysis": result, "structured_data": structured_data}
        else:
            # If no JSON pattern found, return just the text
            return {"analysis": result}
            
    except Exception as e:
        print(f"JSON parsing error: {e}")
        # If we couldn't parse JSON, return just the text
        return {"analysis": result}

def answer_question(question: str, context: str, model_name: str = "llama2") -> str:
    """
    Answer a question based on the provided context
    
    Args:
        question: The question to answer
        context: The context text to use for answering
        model_name: Name of the Ollama model to use
        
    Returns:
        str: Answer to the question
    """
    llm = create_ollama_llm(model_name)
    
    # Create a QA chain
    prompt_template = """
    You are a legal assistant. Use the following context to answer the question.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    qa_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run the QA chain
    return qa_chain.run(context=context, question=question)

def create_conversation_chain(model_name: str = "llama2") -> ConversationChain:
    """
    Create a conversation chain with memory
    
    Args:
        model_name: Name of the Ollama model to use
        
    Returns:
        ConversationChain: LangChain conversation chain
    """
    llm = create_ollama_llm(model_name)
    memory = ConversationBufferMemory()
    
    return ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

def run_direct_query(prompt: str, model_name: str = "llama2") -> str:
    """
    Run a direct query to Ollama
    
    Args:
        prompt: The prompt to send to Ollama
        model_name: Name of the Ollama model to use
        
    Returns:
        str: Response from Ollama
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt}
        )
        return response.json().get("response", "")
    except Exception as e:
        return f"Error querying Ollama: {str(e)}"