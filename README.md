# AI-Enhanced Legal Assistant

A comprehensive legal assistance tool that uses AI to provide document summarization, contract analysis, and voice-commanded question answering capabilities. This application runs locally on your machine to ensure data privacy and security.

## Features

- **Document Summarization**: Extract key information from legal documents
- **Contract Analysis**: Identify parties, terms, obligations, and risks in contracts
- **Voice Q&A**: Ask questions about documents using voice commands

## Architecture

This project uses:

- **Streamlit**: For the user interface
- **LangChain**: For document processing and LLM integration
- **Ollama**: For local AI model inference
- **Whisper**: For local speech transcription
- **PyPDF2/pdfplumber**: For document parsing

All components run locally on your machine, ensuring data privacy and security.

## Prerequisites

- Python 3.9+ installed
- [Ollama](https://ollama.ai/) installed and running locally
- A MacBook Silicon (M1/M2/M3/M4) chip recommended for optimal performance
- Microphone for voice commands

## Installation

1. Clone this repository:
   ```bash
   git clone [repository-url]
   cd legal-assistant
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install and start Ollama:
   ```bash
   # Follow instructions at https://ollama.ai/ to install Ollama
   ollama serve  # Start the Ollama server
   ```

5. Pull a language model (in a separate terminal):
   ```bash
   ollama pull llama2  # or your preferred model
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. The application will open in your web browser (typically at http://localhost:8501)

3. Select a tool from the sidebar:
   - **Document Summarization**: Upload a document and generate a summary
   - **Contract Analysis**: Upload a contract and analyze its key components
   - **Voice Q&A**: Upload a document and ask questions by voice or text

## Folder Structure

```
legal_assistant/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
├── .gitignore                    # Git ignore file
├── utils/
│   ├── __init__.py
│   ├── document_processor.py     # PDF and document processing utilities
│   ├── audio_processor.py        # Audio recording and transcription utilities
│   └── llm_processor.py          # Ollama and LangChain integration
├── models/
│   └── __init__.py               # Model configuration
├── components/
│   ├── __init__.py
│   ├── document_summary.py       # Document summarization component
│   ├── contract_analysis.py      # Contract analysis component
│   └── voice_qa.py               # Voice Q&A component
└── static/
    └── styles.css                # Custom CSS styles
```

## Development

### Adding New Features

To add a new feature:

1. Create a new component in the `components/` directory
2. Add any necessary utility functions in the `utils/` directory
3. Update `app.py` to include the new component in the sidebar navigation

### Using Different Models

You can use any model supported by Ollama. To use a different model:

1. Pull the model using Ollama:
   ```bash
   ollama pull [model-name]
   ```

2. Select the model in the application's model selector dropdown

## Troubleshooting

- **Ollama Connection Issues**: Ensure Ollama is running with `ollama serve`
- **Audio Recording Issues**: Check microphone permissions and settings
- **Document Parsing Errors**: Try a different document format or check if the PDF is scanned/OCR'd

## License

[MIT License](LICENSE)

## Acknowledgements

- [Ollama](https://ollama.ai/) for providing local LLM inference
- [Whisper](https://github.com/openai/whisper) for speech recognition
- [LangChain](https://langchain.com/) for LLM integration
- [Streamlit](https://streamlit.io/) for the user interface