# lmi_onboarding
Basic RAG pipeline, onboarding code for the LMI client project team

## libraries used
- OpenAI
- CHromaDB
- os (for directory management and environment variables)

## what the script does
After initializing an OpenAI and ChromaDB client, the script creates several functions. These accomplish the tasks of populating or retrieving from ChromaDB, which stores the text embeddings, using an OpenAI Embedding function, loading text files, querying ChromaDB for embeddings, and sending user input and retrieving output from ChatGPT.
The main script checks if ChromaDB has the embeddings, and if not, it creates them using an OpenAI Embedding function. Next, it gets the user query, and retrieves the appropriate text files from the embeddings from ChromaDB. After that, it creates a prompt including those text files as context and the user query and sends it to ChatGPT, and outputs the response. 

## to run
A virtual environment should also be created, and should have
```console
pip install openai chromadb
```
run before running the script. 

The terminal command
```console
export OPENAI_API_KEY="your_api_key"
```
must also be run before running the script, either directly or through a .env file, and replacing ```your_api_key``` with an API key from OpenAI.

