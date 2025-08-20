import os
import streamlit as st
import logging
import sys
import time
import concurrent.futures
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
# Import LangChain's Embeddings interface for integration
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb import Client, Settings, HttpClient, PersistentClient
from langchain_core.documents import Document
import hashlib
import json
import importlib
from datetime import datetime
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Configure logging
# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

DEFAULT_PDF_DIRECTORY = "./knowledge"
# Load env variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
CHROMA_COLLECTION_NAME = "pdf_documents"
INDEX_METADATA_FILE = "./index_metadata.json"
CHROMA_PERSIST_DIR = "./chroma_db"
# Enhanced system prompt for legal document context
SYSTEM_PROMPT = """Anda adalah asisten hukum Indonesia.
Jawablah pertanyaan berdasarkan dokumen hukum yang telah disediakan.
Selalu berikan jawaban dalam Bahasa Indonesia dengan gaya formal.
Jika dokumen tidak memberikan informasi yang cukup, katakan dengan jujur bahwa informasi tersebut tidak tersedia."""

st.set_page_config(
    page_title="PDF Knowledge Assitant", layout="wide"
)

# Log environment variables (masked)
if GOOGLE_API_KEY:
    masked_key = GOOGLE_API_KEY[:4] + "..." + GOOGLE_API_KEY[-4:]
    logger.info(f"API Key loaded: {masked_key}")
else:
    logger.warning("No API key found in environment variables")
logger.info(f"ChromaDB host: {CHROMA_HOST}")
logger.info(f"ChromaDB port: {CHROMA_PORT}")

def generate_document_hash(document: Document) -> str:
    """Generate a unique hash for a document based on its content and metadata"""
    content = document.page_content
    metadata_str = json.dumps(document.metadata, sort_keys=True)
    combined = f"{content}{metadata_str}".encode('utf-8')
    return hashlib.md5(combined).hexdigest()

def get_indexed_documents():
    """Get list of document hashes that have already been indexed"""
    if os.path.exists(INDEX_METADATA_FILE):
        try:
            with open(INDEX_METADATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_indexed_documents(indexed_docs):
    """Save the indexed documents metadata"""
    with open(INDEX_METADATA_FILE, 'w') as f:
        json.dump(indexed_docs, f)

def init_session_state():
    # Initialize basic session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

    # Track API key status
    if "api_key_valid" not in st.session_state:
        st.session_state.api_key_valid = False

    # Track embedding model status
    if "embedding_model_ready" not in st.session_state:
        st.session_state.embedding_model_ready = False

    # Track last processing timestamp for rate limiting
    if "last_query_time" not in st.session_state:
        st.session_state.last_query_time = 0

    # Initialize the knowledge base on app startup
    try:
        init_knowledge_base()
    except Exception as e:
        logger.error(f"Error initializing knowledge base: {str(e)}")
        st.error(f"❌ Error initializing knowledge base: {str(e)}")
        st.info("Please check your configuration and try again.")

@st.cache_resource
def create_conversational_chain(
    pdf_directory: str,
):
    # Get API key from environment
    google_api_key = GOOGLE_API_KEY

    if not google_api_key:
        st.error("❌ No Google API key found. Please add it to your .env file.")
        return None

    # Set startup time for performance tracking
    import time
    start_processing_time = time.time()

    # First check if we have already created a vector database
    chroma_exists = False
    try:
        # Try to connect to the persistent ChromaDB
        try:
            port = int(CHROMA_PORT)
            client = HttpClient(host=CHROMA_HOST, port=port)
            collections = client.list_collections()
            for collection in collections:
                if collection.name == CHROMA_COLLECTION_NAME:
                    chroma_exists = True
                    doc_count = collection.count()
                    st.write(f"✅ Found existing collection in ChromaDB with {doc_count} documents")
                    logger.info(f"Found collection {CHROMA_COLLECTION_NAME} with {doc_count} documents")
                    break
        except ValueError as ve:
            logger.error(f"Invalid ChromaDB port: {CHROMA_PORT}. Error: {str(ve)}")
            st.error(f"❌ Invalid ChromaDB port: {CHROMA_PORT}. Using default port 8000.")
            client = HttpClient(host=CHROMA_HOST, port=8000)
            collections = client.list_collections()
            for collection in collections:
                if collection.name == CHROMA_COLLECTION_NAME:
                    chroma_exists = True
                    doc_count = collection.count()
                    st.write(f"✅ Found existing collection in ChromaDB with {doc_count} documents")
                    logger.info(f"Found collection {CHROMA_COLLECTION_NAME} with {doc_count} documents")
                    break
    except Exception as e:
        logger.error(f"Error connecting to ChromaDB: {str(e)}")
        st.error(f"❌ Error connecting to ChromaDB: {str(e)}")

    # Document loading progress
    with st.status("Loading PDF documents...") as status:
        # Get list of PDF files
        try:
            # Check if directory exists, create if not
            if not os.path.exists(pdf_directory):
                os.makedirs(pdf_directory, exist_ok=True)
                st.info(f"📁 Created directory {pdf_directory}")

            # Get PDF files
            pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        except Exception as e:
            st.error(f"❌ Error accessing PDF directory: {str(e)}")
            os.makedirs(pdf_directory, exist_ok=True)
            pdf_files = []

        if not pdf_files:
            st.write("⚠️ No PDF files found in directory")
            documents = []
        else:
            # Create a progress bar for document loading
            doc_progress = st.progress(0)
            st.write("📂 Reading PDF files...")

            # Load all documents at once from the directory
            try:
                loader = PyPDFDirectoryLoader(pdf_directory)
                documents = loader.load()
                doc_progress.progress(1.0)
            except Exception as e:
                st.error(f"❌ Error loading PDF documents: {str(e)}")
                documents = []

            # Get previously indexed document hashes
            indexed_docs = get_indexed_documents()

            # Show files being processed
            for i, pdf_file in enumerate(pdf_files):
                st.write(f"📄 Processed: {pdf_file}")

            st.write(f"✅ Loaded {len(documents)} document chunks from {len(pdf_files)} files")
        status.update(label="Splitting documents into chunks...", state="running")

        # Text splitting (if there are documents to split)
        if documents:
            # Create a progress bar for document splitting
            split_progress = st.progress(0)
            st.write("✂️ Splitting documents into chunks...")

            # Use more aggressive chunk settings for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""],
                length_function=len,
            )

            # Process documents in batches for better UI feedback
            batch_size = min(10, max(1, len(documents)//5))
            all_chunks = []

            for i in range(0, len(documents), batch_size):
                batch = documents[i:min(i+batch_size, len(documents))]
                batch_chunks = text_splitter.split_documents(batch)
                all_chunks.extend(batch_chunks)

                # Update progress
                progress = min(1.0, (i + batch_size) / len(documents))
                split_progress.progress(progress)

                # Update status message periodically
                if i % (batch_size * 2) == 0 or i + batch_size >= len(documents):
                    st.write(f"🔄 Processed {min(i+batch_size, len(documents))}/{len(documents)} documents, generated {len(all_chunks)} chunks so far")

            chunks = all_chunks
            st.write(f"✅ Split into {len(chunks)} chunks")
        else:
            chunks = []
            st.write("⚠️ No documents to split")
        status.update(label="Generating embeddings...", state="running")

        # Show progress for embedding generation
        embedding_progress = st.progress(0)
        st.write("🔢 Generating embeddings...")

        # Try to use FastEmbed for embeddings with batch size for efficiency
        use_fastembed = True
        try:
            logger.info("Initializing FastEmbed with BAAI/bge-small-en-v1.5 model")
            embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", batch_size=32)
            # Test the model with a simple embedding to ensure it works
            test_embed = list(embedding_model.embed(["Test embedding"]))
            if len(test_embed) > 0:
                logger.info(f"FastEmbed test successful, embedding dimension: {len(test_embed[0])}")
                st.write("✅ Using FastEmbed for embeddings")
            else:
                raise ValueError("FastEmbed test returned empty embeddings")
        except Exception as e:
            logger.error(f"Error initializing FastEmbed: {str(e)}")
            st.error(f"❌ Error initializing FastEmbed: {str(e)}")
            st.write("⚠️ Falling back to Google Generative AI Embeddings")
            use_fastembed = False

        # Custom embedding class to track progress
        class ProgressEmbeddings(Embeddings):
            def __init__(self, model, progress_bar):
                self.model = model
                self.progress_bar = progress_bar
                self.total_embedded = 0
                self.total_to_embed = 0
                # Default dimension for zero embeddings in case of errors
                self.embedding_dim = 384  # Default for BAAI/bge-small-en-v1.5

            def embed_documents(self, texts):
                """Embed search documents."""
                self.total_to_embed = len(texts)

                if self.total_to_embed == 0:
                    return []

                embeddings = []
                batch_size = 32

                # Measure embedding time
                start_embed_time = time.time()

                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    try:
                        batch_embeddings = list(self.model.embed(batch))
                        if batch_embeddings and len(batch_embeddings) > 0:
                            # Set the embedding dimension for potential error cases
                            self.embedding_dim = len(batch_embeddings[0])
                        embeddings.extend(batch_embeddings)
                    except Exception as e:
                        st.error(f"❌ Error embedding batch {i//batch_size}: {str(e)}")
                        # Create zero embeddings for failed batches to maintain document count
                        for _ in range(len(batch)):
                            embeddings.append([0.0] * self.embedding_dim)

                    # Update progress
                    self.total_embedded += len(batch)
                    self.progress_bar.progress(min(1.0, self.total_embedded / self.total_to_embed))

                    # Show status updates periodically
                    if self.total_embedded % 50 == 0 or self.total_embedded == self.total_to_embed:
                        elapsed = time.time() - start_embed_time
                        st.write(f"🔄 Embedded {self.total_embedded}/{self.total_to_embed} chunks ({elapsed:.1f}s, {self.total_embedded/max(1, elapsed):.1f} chunks/s)")

                return embeddings

            def embed_query(self, text):
                """Embed a query."""
                try:
                    result = list(self.model.embed([text]))
                    return result[0] if result else [0.0] * self.embedding_dim
                except Exception as e:
                    st.error(f"❌ Error embedding query: {str(e)}")
                    return [0.0] * self.embedding_dim

        # Create embeddings based on what's available
        # if use_fastembed:
        embeddings = ProgressEmbeddings(embedding_model, embedding_progress)
        logger.info("Using custom ProgressEmbeddings with FastEmbed model")
        st.session_state.embedding_model_ready = True
        # else:
        #     # Fall back to Google's embeddings if FastEmbed isn't available
        #     try:
        #         st.write("🔄 Initializing Google Generative AI Embeddings...")
        #         logger.info("Attempting to initialize GoogleGenerativeAIEmbeddings")
        #         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        #         # Test the embedding model
        #         test_result = embeddings.embed_query("Test embedding")
        #         if test_result:
        #             logger.info(f"Google embeddings test successful, dimension: {len(test_result)}")
        #             st.write("✅ Using Google Generative AI Embeddings")
        #             st.session_state.embedding_model_ready = True
        #         else:
        #             raise ValueError("Google embeddings test returned empty result")
        #     except Exception as e:
        #         logger.error(f"Error initializing Google Embeddings: {str(e)}")
        #         st.error(f"❌ Error initializing Google Embeddings: {str(e)}")
        #         st.error("Cannot continue without embeddings. Please check your API keys and try again.")
        #         st.session_state.embedding_model_ready = False
        #         return None

        # Connect to persistent ChromaDB
        status.update(label="Connecting to ChromaDB...", state="running")
        st.write("🔌 Connecting to ChromaDB...")

        # Calculate document hashes more efficiently with parallel processing

        def process_chunk(chunk):
            doc_hash = generate_document_hash(chunk)
            return (chunk, doc_hash)

        # Filter out already indexed documents using parallel processing
        new_chunks = []
        skipped_chunks = 0

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Process chunks in parallel
            chunk_results = list(executor.map(lambda chunk: process_chunk(chunk), chunks))

            for chunk, doc_hash in chunk_results:
                if doc_hash not in indexed_docs:
                    new_chunks.append(chunk)
                else:
                    skipped_chunks += 1

        if skipped_chunks > 0:
            st.write(f"⏩ Skipped {skipped_chunks} previously indexed chunks")

        st.write(f"🔄 Processing {len(new_chunks)} new document chunks")

        # Create or get existing collection
        status.update(label="Creating vector store...", state="running")
        st.write("🗄️ Creating/updating vector database...")

        if len(new_chunks) > 0:
            # Create persistent ChromaDB client
            try:
                client = HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
            except ValueError:
                st.error(f"❌ Invalid ChromaDB port: {CHROMA_PORT}. Using default port 8000.")
                client = HttpClient(host=CHROMA_HOST, port=8000)

            # Use batch processing for better performance
            batch_size = 100
            total_chunks = len(new_chunks)

            if total_chunks > 0:
                vector_store = Chroma(
                    collection_name=CHROMA_COLLECTION_NAME,
                    embedding_function=embeddings,
                    client=client
                )

                # Add documents in batches for better performance
                for i in range(0, total_chunks, batch_size):
                    batch_end = min(i + batch_size, total_chunks)
                    current_batch = new_chunks[i:batch_end]
                    st.write(f"🔄 Adding batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1} ({len(current_batch)} chunks)")
                    vector_store.add_documents(documents=current_batch)
                    # Update progress after each batch
                    embedding_progress.progress(min(1.0, (i + batch_size) / total_chunks))

                # Update the index metadata
                for chunk in new_chunks:
                    doc_hash = generate_document_hash(chunk)
                    indexed_docs[doc_hash] = {
                        "filename": chunk.metadata.get("source", "unknown"),
                        "indexed_at": datetime.now().isoformat()
                    }

                # Save updated index metadata
                save_indexed_documents(indexed_docs)

            st.write("✅ Vector database updated successfully")
        else:
            # Just connect to the existing collection
            try:
                client = HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
            except ValueError:
                st.error(f"❌ Invalid ChromaDB port: {CHROMA_PORT}. Using default port 8000.")
                client = HttpClient(host=CHROMA_HOST, port=8000)
            vector_store = Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=embeddings,
                client=client
            )
            st.write("✅ Using existing vector database")
        status.update(label="Document processing complete!", state="complete")

    if vector_store is not None:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # We're using RetrievalQA without memory
        pass

        # Check API key before initializing the model
        if not google_api_key:
            st.error("❌ Google API key is not set. Please check your .env file.")
            return None

        # Check package versions
        try:
            # Check versions
            import google.generativeai as genai
            import langchain
            import langchain_google_genai

            # Display version info
            version_info = f"Google Generative AI version: {genai.__version__}, "
            version_info += f"LangChain version: {langchain.__version__}"
            st.info(version_info)
            logger.info(version_info)
        except Exception as e:
            error_msg = f"Could not check versions: {str(e)}"
            st.warning(error_msg)
            logger.error(error_msg)

        try:
            # Set API key in environment
            os.environ["GOOGLE_API_KEY"] = google_api_key
            logger.info("Set GOOGLE_API_KEY in environment variables")

            # Create a simple LLM instance with minimal parameters
            logger.info("Initializing GoogleGenerativeAI with gemini-pro model")
            llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
            # Test the LLM with a simple query to verify it works
            test_result = llm.invoke("Test")
            logger.info(f"Successfully created and tested GoogleGenerativeAI LLM, result type: {type(test_result)}")
            st.success("✅ Successfully created Google Generative AI model")
            st.session_state.api_key_valid = True

        except Exception as e:
            error_msg = f"Error initializing LLM: {str(e)}"
            st.error(f"❌ {error_msg}")
            logger.error(error_msg)

            if "safety_settings" in str(e):
                st.info("Try using a simpler model initialization without custom safety settings")
            elif "invalid_api_key" in str(e).lower() or "api key" in str(e).lower():
                st.error("Invalid API key. Check that you have entered the correct API key in your .env file.")
            elif "not found" in str(e).lower() or "supported" in str(e).lower():
                st.error("Model not found error. This could be due to:")
                st.info("1. The model name is incorrect or not available in your region")
                st.info("2. Your API key doesn't have access to this model")
                st.info("3. There might be network connectivity issues")

                # Try a completely different approach as last resort
                try:
                    st.warning("Attempting direct initialization without LangChain...")
                    import google.generativeai as genai
                    genai.configure(api_key=google_api_key)
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content("Test")
                    st.success("✅ Direct Gemini API test successful! Issue is likely with LangChain integration.")
                    logger.info("Direct Gemini API test successful")
                except Exception as genai_e:
                    st.error(f"❌ Direct Gemini API test also failed: {str(genai_e)}")
                    logger.error(f"Direct Gemini API test failed: {str(genai_e)}")
            return None

        try:
            # Most basic chain configuration without custom prompt
            logger.info("Creating ConversationalRetrievalChain with basic configuration")

            # Log the retriever configuration
            logger.info(f"Retriever search_kwargs: k=5")

            # Use RetrievalQA instead which works better with GoogleGenerativeAI
            # Create a custom prompt template for better Bahasa Indonesia responses using ChatPromptTemplate
            system_prompt = """Anda adalah asisten hukum Indonesia.
Gunakan informasi berikut untuk menjawab pertanyaan pengguna dalam Bahasa Indonesia.
Jika Anda tidak tahu jawabannya, katakan saja Anda tidak tahu. JANGAN mencoba membuat jawaban.
Selalu berikan jawaban dalam Bahasa Indonesia dengan gaya formal.

Konteks: {context}"""

            logger.info("Created custom prompt template for Bahasa Indonesia responses")

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])

            # Create the document chain and retrieval chain using modern LangChain approach
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            chain = create_retrieval_chain(retriever, question_answer_chain)

            logger.info("Successfully created ConversationalRetrievalChain")

            # Calculate and show total processing time
            processing_time = time.time() - start_processing_time
            st.success(f"✅ Conversation chain created successfully in {processing_time:.1f} seconds")
            logger.info(f"Conversation chain created successfully in {processing_time:.1f} seconds")

            # Show the configuration of the chain for debugging
            st.info(f"Using model: {llm.model}")
            logger.info(f"Chain configuration - Model: {llm.model}, Temperature: {llm.temperature}")

            return chain
        except Exception as e:
            error_msg = f"Error creating conversation chain: {str(e)}"
            st.error(f"❌ {error_msg}")
            logger.error(error_msg)
            return None
    else:
        st.error("❌ No documents to create a retriever from")
        return None

    return chain

def init_knowledge_base():
    try:
        # Use the already loaded API key
        api_key = GOOGLE_API_KEY

        if not api_key:
            st.error("❌ Gemini API Key not found in the .env file")
            st.info("Please create a .env file in the src directory with GOOGLE_API_KEY=your_api_key")
            with st.expander("How to get a Gemini API key"):
                st.markdown("""
                1. Go to [Google AI Studio](https://aistudio.google.com/)
                2. Create or sign in to your Google account
                3. Navigate to 'Get API key' in the left sidebar
                4. Create a new API key
                5. Copy the key and add it to your .env file as GOOGLE_API_KEY=your-key-here
                """)
            return

        # Create persistent directory for ChromaDB if using local mode
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        # Check if ChromaDB is running
        try:
            try:
                client = HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
            except ValueError:
                st.error(f"❌ Invalid ChromaDB port: {CHROMA_PORT}. Using default port 8000.")
                client = HttpClient(host=CHROMA_HOST, port=8000)
            client.heartbeat()
            st.success("✅ Connected to ChromaDB server")
        except Exception as e:
            st.error(f"❌ Failed to connect to ChromaDB server at {CHROMA_HOST}:{CHROMA_PORT}. Please start the Docker container first.")
            st.error(f"Error: {str(e)}")
            st.info("ℹ️ Run 'docker-compose up -d' in the project root directory to start ChromaDB.")
            return

        if not os.path.exists(DEFAULT_PDF_DIRECTORY):
            os.makedirs(DEFAULT_PDF_DIRECTORY, exist_ok=True)
            st.warning(f"⚠️ Created directory {DEFAULT_PDF_DIRECTORY}. Please add PDF files and restart.")
            return

        pdf_files = [f for f in os.listdir(DEFAULT_PDF_DIRECTORY) if f.lower().endswith('.pdf')]
        if not pdf_files:
            st.warning(f"⚠️ No PDF files found in {DEFAULT_PDF_DIRECTORY}. Please add PDF files and restart the app.")
            return

        try:
            # Calculate total size of PDFs
            total_size_bytes = sum(os.path.getsize(os.path.join(DEFAULT_PDF_DIRECTORY, f)) for f in pdf_files)
            total_size_mb = total_size_bytes / (1024 * 1024)

            st.info(f"📚 Found {len(pdf_files)} PDF files in {DEFAULT_PDF_DIRECTORY} (Total: {total_size_mb:.2f} MB)")

            # Show files in a more compact format
            col1, col2 = st.columns(2)
            for i, pdf_file in enumerate(pdf_files):
                file_size = os.path.getsize(os.path.join(DEFAULT_PDF_DIRECTORY, pdf_file)) / 1024  # KB
                if i % 2 == 0:
                    col1.write(f"📄 {pdf_file} ({file_size:.1f} KB)")
                else:
                    col2.write(f"📄 {pdf_file} ({file_size:.1f} KB)")

            with st.spinner("⏳ Initializing model and preparing knowledge base..."):
                # Ensure API key is set in environment
                os.environ["GOOGLE_API_KEY"] = api_key

                # Also test the embedding model availability
                try:
                    logger.info("Testing Google Generative AI Embeddings availability")
                    test_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    test_result = test_embedding.embed_query("Test embedding")
                    if test_result:
                        logger.info(f"Google Generative AI Embeddings test successful, dimension: {len(test_result)}")
                        st.info("✅ Google Embeddings API is working properly")
                        # Update session state
                        st.session_state.embedding_model_ready = True
                except Exception as e:
                    logger.warning(f"Failed to test Google Generative AI Embeddings: {str(e)}")
                    st.warning("⚠️ Could not verify Google Embeddings API. FastEmbed will be used if available.")
                    # We'll set embedding_model_ready to True later if FastEmbed works

                # Create start time to measure duration
                start_time = time.time()

                # Check if collection already exists in ChromaDB
                try:
                    try:
                        client = HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
                    except ValueError:
                        st.error(f"❌ Invalid ChromaDB port: {CHROMA_PORT}. Using default port 8000.")
                        client = HttpClient(host=CHROMA_HOST, port=8000)

                    collections = client.list_collections()
                    collection_exists = any(col.name == CHROMA_COLLECTION_NAME for col in collections)

                    if collection_exists:
                        count = next((col.count() for col in collections if col.name == CHROMA_COLLECTION_NAME), 0)
                        if count > 0:
                            st.info(f"📚 Using existing ChromaDB collection with {count} documents")
                except Exception as e:
                    st.error(f"❌ Error checking ChromaDB collections: {str(e)}")
                    collection_exists = False

                # Create conversation chain with progress indicators
                chain = create_conversational_chain(DEFAULT_PDF_DIRECTORY)

                # Only update if chain was successfully created
                if chain is not None:
                    st.session_state.conversation_chain = chain

                    # Calculate duration
                    duration = time.time() - start_time

                    st.success(f"✅ Knowledge base created successfully with {len(pdf_files)} PDF files in {duration:.1f} seconds!")

                    # Check if ChromaDB collection has documents
                    try:
                        try:
                            client = HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
                        except ValueError:
                            st.error(f"❌ Invalid ChromaDB port: {CHROMA_PORT}. Using default port 8000.")
                            client = HttpClient(host=CHROMA_HOST, port=8000)

                        collections = client.list_collections()
                        for collection in collections:
                            if collection.name == CHROMA_COLLECTION_NAME:
                                doc_count = collection.count()
                                st.info(f"📊 ChromaDB collection has {doc_count} documents")
                    except Exception as e:
                        st.warning(f"⚠️ Could not check document count: {str(e)}")
                else:
                    st.error("❌ Failed to create knowledge base. Please check the PDF files.")
        except Exception as e:
            st.error(f"❌ Error processing PDF files: {str(e)}")
    except FileNotFoundError as e :
        st.error(f"File Not found")
    except Exception as e :
        st.error(f"An error occured: {str(e)}")

# Initialize session state (which will also call init_knowledge_base)
init_session_state()

st.title("📚 PDF Knowledge Assistant")

# Add empty state check and upload button
pdf_files = []
if os.path.exists(DEFAULT_PDF_DIRECTORY):
    pdf_files = [f for f in os.listdir(DEFAULT_PDF_DIRECTORY) if f.lower().endswith('.pdf')]

if not pdf_files:
    st.warning("⚠️ No PDF files found in the knowledge directory.")
    st.info("Please add PDF files to the `./knowledge` directory and restart the app.")

    # Display instructions for adding documents
    with st.expander("How to add documents"):
        st.markdown("""
        1. Create a directory named `knowledge` in the project root if it doesn't exist
        2. Add your PDF files to this directory
        3. Restart the application

        The files should be automatically processed when the app restarts.
        """)

# Show ChromaDB connection status
with st.sidebar:
    st.header("ChromaDB Connection")
    try:
        try:
            client = HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
        except ValueError:
            st.error(f"❌ Invalid ChromaDB port: {CHROMA_PORT}. Using default port 8000.")
            client = HttpClient(host=CHROMA_HOST, port=8000)
        client.heartbeat()
        st.success(f"✅ Connected to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")
        collections = client.list_collections()
        if collections:
            st.info(f"📊 Available collections: {', '.join([c.name for c in collections])}")
        else:
            st.info("No collections found in ChromaDB")
    except Exception as e:
        st.error(f"❌ ChromaDB connection failed: {str(e)}")
        st.info("ℹ️ Run 'docker-compose up -d' to start ChromaDB")

# with st.sidebar:
#     st.header("Configuration")
#     pdf_directory = st.text_input("PDF Directory Path", value="./knowledge")
#     api_key = st.text_input("Google API Key", value="", type="password")

with st.expander("Knowledge Base Information"):
    st.write(f"Documents loaded from: {DEFAULT_PDF_DIRECTORY}")
    st.write("To add new documents, place PDF files in the ./knowledge directory and restart the app.")


for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Disable chat input if no documents or model isn't ready
chat_disabled = len(pdf_files) == 0 or st.session_state.conversation_chain is None
if chat_disabled:
    st.text_input("Ask a question about your documents...",
                  disabled=True,
                  placeholder="Add documents to enable questions")

    # Show more informative message
    if len(pdf_files) == 0:
        st.error("⛔ No documents available - please add PDFs to the knowledge directory")
    elif not st.session_state.api_key_valid:
        st.error("⛔ API key invalid or missing - please check your API key in the .env file")
    elif not st.session_state.embedding_model_ready:
        st.error("⛔ Embedding model not initialized - check error messages above")
    else:
        st.error("⛔ Knowledge base not initialized - check error messages above")

elif prompt := st.chat_input("Ask a question about your documents..."):
    if st.session_state.conversation_chain is None:
        st.error("Knowledge base not initialized. Please check error messages above.")
        st.info("Try adding PDF files to the 'knowledge' directory and restart the application.")
    else:
        # Check if we're being rate limited (minimum 1 second between queries)
        current_time = time.time()
        if current_time - st.session_state.last_query_time < 1.0:
            time.sleep(1.0) # Add a small delay to prevent rapid-fire queries

        # Update last query time
        st.session_state.last_query_time = time.time()

        # Add the user message to the history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display the user message
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Create a status indicator for the retrieval and answer generation process
            with st.status("Processing your question...") as status:
                status.update(label="🔍 Searching knowledge base...", state="running")

                # Track processing time
                start_time = time.time()

                try:
                    # Keep query simple to avoid any issues with special characters
                    bahasa_prompt = prompt
                    logger.info(f"User prompt: {prompt}")
                    logger.info(f"Formatted prompt: {bahasa_prompt}")

                    # Log processing start time for performance tracking
                    process_start_time = time.time()

                    # Check if conversation chain is still valid
                    if st.session_state.conversation_chain is None:
                        st.error("❌ Conversation chain not available.")
                        status.update(label="Error: Conversation chain not available", state="error")

                        # Add error message to chat history
                        error_message = "Maaf, model AI tidak tersedia. Silakan periksa log error."
                        st.markdown(f"⚠️ {error_message}")

                        # Add to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_message
                        })
                        st.stop()

                    # Get response from conversation chain
                    try:
                        logger.info("Sending query to retrieval chain")
                        start_query_time = time.time()
                        try:
                            # Use the new input format for create_retrieval_chain
                            response = st.session_state.conversation_chain.invoke({"input": bahasa_prompt})
                        except Exception as chain_error:
                            # Try alternate invocation format if the first fails
                            logger.warning(f"First invocation failed: {str(chain_error)}, trying alternative")
                            response = st.session_state.conversation_chain({"input": bahasa_prompt})
                        query_time = time.time() - start_query_time
                        logger.info(f"Received response in {query_time:.2f}s")
                        logger.info(f"Response type: {type(response)}")

                        # Log detailed response structure for debugging
                        if isinstance(response, dict):
                            logger.info(f"Response keys: {list(response.keys())}")

                        # Get source documents if available (new format uses 'context' key)
                        source_docs = []
                        if isinstance(response, dict) and 'context' in response:
                            source_docs = response['context']
                        elif hasattr(response, 'source_documents'):
                            source_docs = response.source_documents
                        elif isinstance(response, dict) and 'source_documents' in response:
                            source_docs = response['source_documents']

                        if source_docs:
                            with st.expander("📚 Sumber Referensi"):
                                for i, doc in enumerate(source_docs):
                                    st.markdown(f"**Dokumen {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                                    st.markdown(f"```\n{doc.page_content[:300]}...\n```")
                    except Exception as e:
                        if "API key not valid" in str(e) or "authentication" in str(e).lower():
                            st.error("❌ API key error: Please check your Google API key")
                            status.update(label="Error: API key issue", state="error")

                            # Add error message to chat history
                            error_message = "Maaf, API key tidak valid. Silakan periksa konfigurasi API key Anda."
                            st.markdown(f"⚠️ {error_message}")

                            # Add to session state
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_message
                            })
                            st.stop()
                        else:
                            raise e

                    # Calculate and show retrieval time
                    retrieval_time = time.time() - start_time
                    status.update(label=f"⚙️ Generating answer (retrieval took {retrieval_time:.2f}s)...", state="running")

                    # Get the answer from the create_retrieval_chain response with better handling
                    if isinstance(response, dict):
                        # Try multiple possible response formats (new format uses 'answer' key)
                        answer = response.get('answer', '')
                        if not answer:
                            answer = response.get('result', '')
                        if not answer and isinstance(response, dict):
                            # Search through all dictionary keys for any text content
                            for key, value in response.items():
                                if isinstance(value, str) and value and key not in ['context', 'input']:
                                    answer = value
                                    break
                        if not answer:
                            answer = str(response)
                    else:
                        answer = str(response)

                    # Display the answer with styling
                    st.markdown(f"{answer}")

                    # Show completion status with time
                    total_time = time.time() - start_time
                    status.update(label=f"✅ Response complete in {total_time:.2f} seconds", state="complete")

                    # Add response to session state messages with formatted sources
                    content = answer

                    # Add sources if available in a more structured format
                    if source_docs:
                        unique_sources = set()
                        for doc in source_docs:
                            source = doc.metadata.get('source', 'Unknown')
                            if source and source != 'Unknown':
                                filename = source.split('/')[-1] if '/' in source else source
                                unique_sources.add(filename)

                        if unique_sources:
                            content += "\n\n**Sumber Referensi:**\n" + "\n".join([f"- {s}" for s in unique_sources])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": content
                    })

                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
                    status.update(label="Failed to generate response", state="error")

                    # Add error message to chat history
                    error_message = f"Maaf, terjadi kesalahan dalam menghasilkan respons: {str(e)}. Silakan coba lagi."
                    st.markdown(f"⚠️ {error_message}")

                    # Add to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
