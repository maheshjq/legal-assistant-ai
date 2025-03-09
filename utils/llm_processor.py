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
from langchain_community.document_loaders.base import Document
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

def analyze_contract(text: str, model_name: str = "llama2") -> Dict[str, Any]:
    """
    Analyze a contract document
    
    Args:
        text: Contract text to analyze
        model_name: Name of the Ollama model to use
        
    Returns:
        Dict: Analysis results including parties, key terms, obligations, etc.
    """
    llm = create_ollama_llm(model_name)
    
    # Create a contract analysis chain
    prompt_template = """
    You are a legal contract analyst. Analyze the following contract and extract key information.
    
    Contract:
    {text}
    
    Please provide a structured analysis with the following information:
    1. Parties involved
    2. Contract type
    3. Key dates (effective date, termination date, etc.)
    4. Key terms and conditions
    5. Obligations of each party
    6. Termination conditions
    7. Potential legal issues or risks
    8. Recommendations
    
    Format your response as a structured JSON object.
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    analysis_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run the analysis
    result = analysis_chain.run(text=text)
    
    # Note: In a real implementation, we would parse the JSON result
    # Here we're just returning the text as the LLM might not always 
    # provide valid JSON format
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