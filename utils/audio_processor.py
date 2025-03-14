"""
Audio processing utilities for recording and transcribing speech.
"""

import tempfile
import os
import subprocess
import sys
import time
import sounddevice as sd
import numpy as np
import whisper
from scipy.io.wavfile import write as write_wav
import wavio
from typing import Tuple, Optional, Dict, Any, Callable

# Global whisper model instance (load once)
_whisper_model = None

def get_whisper_model(model_name: str = "base") -> Any:
    """
    Get or load the Whisper model
    
    Args:
        model_name: Size of the Whisper model to load ("tiny", "base", "small", "medium", "large")
        
    Returns:
        The loaded Whisper model
    """
    global _whisper_model
    
    if _whisper_model is None:
        _whisper_model = whisper.load_model(model_name)
    
    return _whisper_model

def record_audio(duration: int = 10, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Record audio from the microphone
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Audio sample rate in Hz
        
    Returns:
        Tuple containing the audio data and sample rate
    """
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    return audio_data, sample_rate

def save_audio_to_temp_file(audio_data: np.ndarray, sample_rate: int) -> str:
    """
    Save audio data to a temporary WAV file
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Audio sample rate in Hz
        
    Returns:
        Path to the temporary WAV file
    """
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"recording_{int(time.time())}.wav")
    
    # Normalize audio data
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Save as WAV
    wavio.write(temp_file, audio_data, sample_rate, sampwidth=2)
    
    return temp_file

def transcribe_audio_file(file_path: str, language: str = "en") -> Dict[str, Any]:
    """
    Transcribe audio file using Whisper
    
    Args:
        file_path: Path to the audio file
        language: Language code (e.g., "en" for English)
        
    Returns:
        Dict containing the transcription results
    """
    model = get_whisper_model()
    result = model.transcribe(file_path, language=language)
    return result

def transcribe_audio_data(audio_data: np.ndarray, sample_rate: int, language: str = "en") -> Dict[str, Any]:
    """
    Transcribe audio data using Whisper
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Audio sample rate in Hz
        language: Language code (e.g., "en" for English)
        
    Returns:
        Dict containing the transcription results
    """
    temp_file = save_audio_to_temp_file(audio_data, sample_rate)
    result = transcribe_audio_file(temp_file, language)
    
    # Clean up temp file
    try:
        os.remove(temp_file)
    except Exception as e:
        print(f"Warning: Failed to remove temporary file {temp_file}: {e}")
    
    return result

def record_and_transcribe(duration: int = 10, language: str = "en") -> Dict[str, Any]:
    """
    Record audio from microphone and transcribe it
    
    Args:
        duration: Recording duration in seconds
        language: Language code (e.g., "en" for English)
        
    Returns:
        Dict containing the transcription results
    """
    audio_data, sample_rate = record_audio(duration=duration)
    return transcribe_audio_data(audio_data, sample_rate, language=language)

def get_text_from_transcription(transcription_result: Dict[str, Any]) -> str:
    """
    Extract text from transcription result
    
    Args:
        transcription_result: Transcription result from Whisper
        
    Returns:
        Transcribed text
    """
    return transcription_result.get("text", "")

# Functional composition for common operations
def record_and_get_text(duration: int = 10, language: str = "en") -> str:
    """
    Record audio and return only the transcribed text
    
    Args:
        duration: Recording duration in seconds
        language: Language code (e.g., "en" for English)
        
    Returns:
        Transcribed text
    """
    result = record_and_transcribe(duration=duration, language=language)
    return get_text_from_transcription(result)


def check_ffmpeg_installed() -> bool:
    """
    Check if ffmpeg is installed on the system
    
    Returns:
        bool: True if ffmpeg is installed, False otherwise
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=False
        )
        return True
    except FileNotFoundError:
        return False

# Add this check to the record_and_transcribe function
def record_and_transcribe(duration: int = 10, language: str = "en") -> Dict[str, Any]:
    """
    Record audio from microphone and transcribe it
    
    Args:
        duration: Recording duration in seconds
        language: Language code (e.g., "en" for English)
        
    Returns:
        Dict: Containing the transcription results or error message
    """
    # Check for ffmpeg
    if not check_ffmpeg_installed():
        return {
            "error": True,
            "text": "ffmpeg is not installed. Please install ffmpeg and try again.",
            "install_instructions": {
                "macos": "brew install ffmpeg",
                "ubuntu_debian": "sudo apt update && sudo apt install ffmpeg",
                "windows": "choco install ffmpeg or download from https://ffmpeg.org/download.html"
            }
        }
    
    audio_data, sample_rate = record_audio(duration=duration)
    return transcribe_audio_data(audio_data, sample_rate, language=language)