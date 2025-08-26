# movie_search.py
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np

# -----------------------------
# Load dataset and create embeddings (global for testing)
# -----------------------------
CSV_PATH = "movies.csv"
df = pd.read_csv(CSV_PATH)

# -----------------------------
# Load the Sentence Transformer model
# -----------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# -----------------------------
# Convert the 'plot' of the movies into embeddings
# -----------------------------
embeddings = model.encode(df['plot'].tolist(), convert_to_numpy=True, show_progress_bar=False)

# Fit NearestNeighbors index for cosine similarity search
index = NearestNeighbors(n_neighbors=min(5, len(df)), metric='cosine')
index.fit(embeddings)

# -----------------------------
# Define search function
# -----------------------------
def search_movies(query, top_n=5):
    """
    Search top_n movies based on plot similarity to the query.
    Returns a DataFrame with columns: title, plot, similarity
    """
    # Convert query to embedding
    q_emb = model.encode([query], normalize_embeddings=True)

    # Compute nearest neighbors
    distances, idx = index.kneighbors(q_emb, n_neighbors=min(top_n, len(df)), return_distance=True)
    
    # Build results
    results = []
    for i, dist in zip(idx[0], distances[0]):
        row = df.iloc[i]
        results.append({
            "title": row["title"],
            "plot": row["plot"],
            "similarity": round(1.0 - dist, 4)  # cosine distance -> similarity
        })
    
    return pd.DataFrame(results)
