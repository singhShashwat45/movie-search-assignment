# Movie Semantic Search Assignment

This repository contains a solution for a semantic search assignment on movie plots. It allows you to search for movies based on plot similarity using **Sentence Transformers**.

---

## 🎬 Project Overview

This project implements a semantic search system for movie plots. Given a query like `"spy thriller in Paris"`, it returns the most relevant movies, ranked by their plot similarity to the query.

### Features

-   Uses the `all-MiniLM-L6-v2` SentenceTransformer model for generating embeddings.
-   Supports returning the top-N most similar search results.
-   Includes unit tests to verify core functionality.

---

## 🚀 Setup

Follow these steps to get the project running on your local machine.

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/movie-search-assignment.git](https://github.com/your-username/movie-search-assignment.git)
    cd movie-search-assignment
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    ```
    * **Windows**:
        ```bash
        venv\Scripts\activate
        ```
    * **macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the notebook**:
    The main workflow, including data loading, embedding generation, and search demonstration, is in the Jupyter notebook.
    ```bash
    jupyter notebook movie_semantic_search.ipynb
    ```

---

## 🧪 Testing

To verify that the search function works correctly, you can run the included unit tests.

```bash
# Run all unit tests from the root directory
python -m unittest discover -s tests -p "*.py" -v

# Or run the specific test file
python -m unittest tests.test_movie_search -v
