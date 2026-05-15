# RAG Chat Application

A comprehensive Retrieval-Augmented Generation (RAG) chat application featuring a graphical interface, document processing, local LLM inference, and voice-to-text capabilities.

## Features
- **Document RAG**: Load and query PDF documents using FAISS and BM25.
- **Local Inference**: Uses `llama-cpp-python` for private, local LLM execution.
- **Voice Transcription**: Integrated `faster-whisper` for speech-to-text input.
- **Web Search**: Fallback to DuckDuckGo search for real-time information.
- **Modern GUI**: Built with PySide6 featuring animations and a chat-like experience.

## Prerequisites
- Python 3.8 or higher
- C++ Compiler (required for building `llama-cpp-python`)
- (Optional) CUDA-capable GPU for faster inference

## Installation

1. **Clone the repository** (or save the files to a folder).

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   *Note: If you have an NVIDIA GPU, you may want to install the CUDA-enabled version of llama-cpp-python:*
   ```bash
   # Example for Windows/Linux with CUDA
   pip uninstall llama-cpp-python -y
   set CMAKE_ARGS="-DGGML_CUDA=on"
   pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
   ```

## Usage

1. **Run the application:**
   ```bash
   python rag.py
   ```

2. **Using the App:**
   - **Load Documents**: Use the interface to select PDF files for indexing.
   - **Chat**: Type your questions in the input box.
   - **Voice**: Use the microphone feature to transcribe your speech.

## Dependencies
- **PySide6**: GUI Framework
- **PyMuPDF (fitz)**: PDF parsing
- **FAISS**: Vector similarity search
- **Sentence-Transformers**: Text embeddings
- **Llama-cpp-python**: Local LLM runner
- **Faster-Whisper**: Speech-to-text
- **DuckDuckGo-Search**: Web search integration
