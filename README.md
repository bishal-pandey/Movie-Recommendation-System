# 🎬 Movie Recommendation System

A **content-based movie recommendation system** built using **TF-IDF, NLP, and Streamlit**.  
It recommends movies based on textual similarity of movie metadata such as title, overview, genres, cast, and director.

---

## Features
- Search movies by **title or keywords**
- **TF-IDF + Cosine Similarity** based recommendations
- NLP preprocessing with **lemmatization**
- Displays **movie posters and details**
- Interactive **Streamlit UI**

---

##Tech Stack
- Python
- Streamlit
- scikit-learn
- NLTK
- Pandas / NumPy

---

## How It Works
1. Combine movie metadata into a single text field  
2. Clean and lemmatize text  
3. Vectorize using **TF-IDF (1–2 ngrams)**  
4. Compute similarity using **cosine similarity**  
5. Recommend top similar movies  
---
