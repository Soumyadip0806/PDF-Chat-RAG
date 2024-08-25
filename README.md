# Chat-With-PDF
This is a RAG Application for Chat with PDF
- live : https://pdf-chat-rag-sg.streamlit.app/#chat-with-your-pdf

# Chat with Your PDF

This project allows you to interactively chat with a PDF document using a custom Streamlit app. The application processes PDF files, extracts text, and provides a conversational interface where you can ask questions and receive answers based on the content of the PDF.

## Features

- **PDF Upload**: Upload one or more PDF files.
- **Text Extraction**: Automatically extract text from the uploaded PDF files.
- **Conversation History**: Maintain a history of the conversation, so you can see all the previous questions and answers.
- **Interactive Chat Interface**: Ask questions and get responses as if you're chatting with your document.
- **Customizable with CSS and JavaScript**: The app includes custom CSS and JavaScript to enhance the user interface, including the ability to submit queries using the Enter key.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Soumyadip0806/PDF-Chat-RAG.git
    cd chat-with-pdf
    ```

2. **Create a virtual environment and activate it:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    - Create a `.env` file in the root directory and add your API keys. For example:
    
      ```bash
      GOOGLE_API_KEY=your-google-api-key
      OPENAI_API_KEY=your-openai-api-key
      ```

5. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

## Usage

1. Upload your PDF files in the sidebar.
2. Click on "Process" to extract text and store embeddings.
3. Start typing your questions in the chat interface.
4. View the responses in the chat window and see the conversation history.

## Project Structure

- `app.py`: The main Streamlit app file.
- `requirements.txt`: List of Python dependencies.
- `static/`: Contains custom CSS and JavaScript files.
- `documents/`: Directory where the uploaded PDF files are stored.
- `utils.py`: Utility functions for PDF processing, text extraction, etc.


### PDF Processing

The project uses functions like `get_pdf_text`, `get_text_chunk`, and `get_vectorstore` to process PDF files and generate embeddings. You can customize these functions in `utils.py` to fit your specific needs.

## Troubleshooting

- **StreamlitAPIException**: If you encounter issues with session state or UI components, ensure that all required keys are properly initialized in the `initialize_session_state` function.
- **Missing API Keys**: Make sure your `.env` file contains valid API keys.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to suggest improvements or report bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


