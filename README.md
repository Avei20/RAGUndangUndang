# PDF Knowledge Assistant

This application provides a conversational AI assistant that can answer questions about Indonesian legal documents. The application uses Streamlit for the user interface, LangChain for orchestrating the RAG workflow, and ChromaDB for vector storage.

## Features

- Load and process PDF documents from a specified directory
- Store document embeddings in a persistent ChromaDB database
- Answer questions in Bahasa Indonesia based on the document content
- Track processed documents to avoid reprocessing

## Setup and Usage

### Prerequisites

- Docker and Docker Compose for running ChromaDB
- Python 3.11 or higher

### Environment Variables

Create a `.env` file in the `src` directory with the following variables:

```
GOOGLE_API_KEY=your_google_api_key_here
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

### PDF Documents

Add your PDF documents to the `knowledge` directory. The application will process these documents and create embeddings.

### Running the Application

1. Start ChromaDB using Docker Compose:

```bash
docker-compose up -d
```

2. Run the Streamlit application:

```bash
cd src
streamlit run streamlit.py
```

### Performance Optimization

- The application uses document hashing to avoid reprocessing documents that have already been embedded
- ChromaDB persistence ensures that embeddings are stored and reused between application restarts
- Batch processing is implemented for improved performance during document ingestion

## System Prompt

The assistant is configured to respond in Bahasa Indonesia, as the legal documents are written in Bahasa. The system prompt instructs the model to:

1. Act as an AI assistant specializing in Indonesian law
2. Always respond in Bahasa Indonesia
3. Be polite and professional
4. Admit when it doesn't have enough information to answer a question
