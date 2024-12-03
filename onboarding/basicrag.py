import os
import openai
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_data")

# OpenAI embedding function for ChromaDB
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ['OPENAI_API_KEY'],
                model_name="text-embedding-3-small"
            )


# Create or load a ChromaDB collection
collection_name = "niche_docs"
collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=openai_ef)

# Helper function to add texts to ChromaDB
def add_texts_to_chromadb(texts, collection):
    for i, text in enumerate(texts):
        collection.add(
            documents=[text],
            ids=[f"doc_{i}"],
            metadatas=[{"source": f"Document {i+1}"}]
        )

# Load text files
def load_text_files(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

# Query ChromaDB
def query_chromadb(query, collection, k=3):
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    return results["documents"][0]  # Top-k documents

# Create and print the prompt
def create_prompt(contexts, user_query):
    prompt = (
        "You are a helpful assistant with expertise in niche topics. Use the context below to answer the user's question.\n\n"
        "Context:\n\n"
        f"{' '.join(contexts)}\n\n"
        f"Question: {user_query}\n\n"
        "Answer:"
    )
    print("\n===== Prompt =====\n")
    print(prompt)
    print("\n==================\n")
    return prompt

# Send the prompt to GPT-4o-mini
def get_response(prompt, model="gpt-4o-mini"):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers queries regarding a given context."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Main script
if __name__ == "__main__":
    # Step 1: Load texts and add to ChromaDB
    text_directory = "txts"    # Replace with your text file directory

    if collection.count() == 0:  # Check if collection is empty
        print("Adding documents to ChromaDB...")
        texts = load_text_files(text_directory)
        add_texts_to_chromadb(texts, collection)
    else:
        print("ChromaDB collection already populated.")

    # Step 2: Query ChromaDB
    user_query = input("Enter your query: ")
    k = int(input("Enter the number of top results to retrieve (k): "))
    top_contexts = query_chromadb(user_query, collection, k=k)

    # Step 3: Create and print the prompt
    prompt = create_prompt(top_contexts, user_query)

    # Step 4: Get and print the response
    print("\n===== Response =====\n")
    print(get_response(prompt))
