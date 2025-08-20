# Explanation Documentation
Basically the current architecture is like this

```
LLM -> Langchain -> Chromadb
```

Streamlit act as UI input for the User

## The algorithm Used
Basically the code working on this algorithm

1. Initialize VectorDB
  - Connect to ChromaDB
  - Check if the document is already index in the DB
  - If not then  read the knowledge directory
  - Chunking the document
  - Embed the chunk using FastEmbed
  - Insert into vector db
2. Initialize session with : empty message and empty vector db
3. Handle user input
   - For every user input LLM will query the knowledge to retrieve the information about user questions
   - If there is no information stored in the VectorDB, LLM will return nothing about the data's in the knowledge base.

## How to run the app

The application is setup using uv, so you can use uv to run or simply use docker to run everything

1. Copy the .env.example

2. Replace the `GOOGLE_API_KEY` with your own api key

3. Then run this command,

```
docker compose up --build -d
```

4. The website is served at `localhost:8501`
5. Wait the document is loaded into the VectorDB, it might takes a while.
6. Voila you can chat with the knowledge .

## Evaluation
### Functionality: Does the RAG system work correctly and provide relevant answers?
Yes, the code is running correcly. You can see the reference knowledg details when interacting with LLM by expanding the result.

### Code Structure: Is the code well-structured, easy to read, and does it follow best practices?
Nope, the current code implementation is scripting to fulfill the functionality first. But, the code quality can be well read by separating each functionality into one function.

### Technology Usage: To what extent did you successfully implement the mentioned technologies (Chainlit/Streamlit, LangChain/LangGraph, Async, etc.)?
Actually i want to use Google ADK or Agno to simply load the knowledge using some builtin function with Agentic mode. But, in this case i use Langchain and Streamlit as Frontend User Interface

### Development Practices: Did you use Git effectively, and does your code demonstrate an understanding of code quality (production-level Python)?
Yes the current code is published into public GitHub repository. But, i think the code style wont match the production-level. But the functionality is already production-ready.

### Documentation: We expect you to include clear and detailed documentation.
Here is the documentation file where the algoritm used and explained.

### Future Improvement: We expect you to be able to articulate potential future improvements for the application and how they could be scaled.
Using LLM is hard because there is a constant response from the LLM for single question. So, to ensure the negative case with the current implementation. I planned to use OpenTelemetry Instrumentation such as OpenLit and sent it to Jaeger and Visualize using Grafana Dashboard. I tried to use this in GDG Cloud Jakarta Cloud Next event last mont in tiket.com office to demonstrate the dashboard using OpenLit. The data provided by OpenLit is completed, you can see how much the VectorDB is accessed, how much token is spend and which provider is hit, etc. You can check the implementation in the OpenLit documentation [here](https://openlit.io/)
