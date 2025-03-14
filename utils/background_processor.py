# New file: utils/background_processor.py

import threading
import time
from typing import Dict, Any, Callable, Optional
import streamlit as st

class BackgroundTaskManager:
    """Manager for background tasks with progress tracking"""
    
    def __init__(self):
        self.tasks = {}
        
    def start_task(self, 
                   task_id: str, 
                   func: Callable, 
                   args: tuple = (), 
                   kwargs: Dict[str, Any] = {}) -> str:
        """
        Start a background task
        
        Args:
            task_id: Unique identifier for the task
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            str: Task ID
        """
        # Initialize task state
        if 'bg_tasks' not in st.session_state:
            st.session_state.bg_tasks = {}
            
        task_state = {
            'status': 'running',
            'progress': 0,
            'message': 'Starting task...',
            'result': None,
            'error': None
        }
        
        st.session_state.bg_tasks[task_id] = task_state
        
        # Define wrapper to update task state
        def task_wrapper():
            try:
                # Update progress callback
                def update_progress(progress: float, message: str = ''):
                    task_state = st.session_state.bg_tasks.get(task_id)
                    if task_state:
                        task_state['progress'] = progress
                        if message:
                            task_state['message'] = message
                
                # Add progress callback to kwargs
                kwargs['progress_callback'] = update_progress
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Update task state
                task_state = st.session_state.bg_tasks.get(task_id)
                if task_state:
                    task_state['status'] = 'completed'
                    task_state['progress'] = 100
                    task_state['result'] = result
                    task_state['message'] = 'Task completed successfully'
            
            except Exception as e:
                # Update task state with error
                task_state = st.session_state.bg_tasks.get(task_id)
                if task_state:
                    task_state['status'] = 'error'
                    task_state['error'] = str(e)
                    task_state['message'] = f'Error: {str(e)}'
        
        # Start task in background thread
        thread = threading.Thread(target=task_wrapper)
        thread.start()
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of a task
        
        Args:
            task_id: Task ID
            
        Returns:
            Dict: Task status
        """
        if 'bg_tasks' not in st.session_state or task_id not in st.session_state.bg_tasks:
            return {
                'status': 'not_found',
                'progress': 0,
                'message': 'Task not found',
                'result': None,
                'error': None
            }
        
        return st.session_state.bg_tasks[task_id]
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        Get result of a completed task
        
        Args:
            task_id: Task ID
            
        Returns:
            Any: Task result or None if task is not completed
        """
        status = self.get_task_status(task_id)
        if status['status'] == 'completed':
            return status['result']
        
        return None

# Create a global task manager
task_manager = BackgroundTaskManager()

# Helper function to add progress tracking to the summarize_document function
def summarize_document_with_progress(
    text: str, 
    model_name: str = "llama2", 
    summary_length: str = "Balanced",
    progress_callback: Callable[[float, str], None] = None
) -> str:
    """
    Summarize a document with progress tracking
    
    Args:
        text: Document text to summarize
        model_name: Name of the Ollama model to use
        summary_length: Desired summary length ("Concise", "Balanced", or "Detailed")
        progress_callback: Callback function for progress updates
        
    Returns:
        str: Generated summary
    """
    from utils.llm_processor import summarize_document
    
    # Update progress
    if progress_callback:
        progress_callback(0.1, "Initializing summarization...")
    
    # For long documents, update progress during chunking
    if len(text) > 4000:
        # Simulate progress for now
        # In a real implementation, you would modify summarize_document to
        # accept a progress_callback parameter
        if progress_callback:
            progress_callback(0.3, "Processing document in chunks...")
        
        # Call the actual summarization with progress tracking
        summary = summarize_document(text, model_name=model_name, summary_length=summary_length)
        
        if progress_callback:
            progress_callback(0.9, "Finalizing summary...")
        
        return summary
    else:
        if progress_callback:
            progress_callback(0.5, "Generating summary...")
        
        summary = summarize_document(text, model_name=model_name, summary_length=summary_length)
        
        if progress_callback:
            progress_callback(1.0, "Summary completed")
        
        return summary