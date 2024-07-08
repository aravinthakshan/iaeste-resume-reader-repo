import time
import google.generativeai as genai
import numpy as np
import pandas as pd
import faiss

# Configure the Gemini API key
genai.configure(api_key='AIzaSyA5XxNs2x2ySVODemRLr7s3mz9n0196p5Q')

def embed_fn(text):
    """Generate embeddings for a given text."""
    return genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document",
        title="requirements"
    )["embedding"]

def build_index(embeddings):
    """
    Build a Faiss index for efficient similarity search.
    """
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def calculate_similarity(query_text, large_text):
    """
    Calculate the cosine similarity between a query text and a larger text using Faiss.
    """
    start_time = time.time()

    # Embed the query and large text using the specified model
    query_embedding = embed_fn(query_text)
    large_text_embedding = embed_fn(large_text)

    # Convert embeddings to numpy arrays
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    large_text_embedding = np.array(large_text_embedding).astype('float32').reshape(1, -1)

    # Build Faiss index with the large text embedding
    index = build_index(large_text_embedding)

    # Perform the similarity search
    distances, indices = index.search(query_embedding, 1)

    similarity_score = 1 - distances[0][0] / 2  # Convert to cosine similarity

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds")

    return similarity_score




