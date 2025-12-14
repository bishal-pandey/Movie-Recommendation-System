import pandas as pd
import numpy as np
import pickle
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('wordnet')    
from nltk.stem import WordNetLemmatizer

lemitizer = WordNetLemmatizer()

#load Data
df = pd.read_csv("Data\clean_movies_data.csv")


def runtime_category(runtime):
    """
    convert runtime into categories
    
    """
    if runtime <= 60:
        return "_Short_"
    elif 60 < runtime <= 150:
        return "_Medium_"
    else:
        return "_Long_"
    


def join_all(df):
    """
    Join all necessary columns for vectoriation
    
    """
    return " ".join([df['title'], df['original_language'], df['overview'], str(df['release_year']), 
               df['runtime_category'],df['genres'], df['top_cast'], df['director'], df['writers'], 
               df['production_companies'], df['spoken_languages']])



def clean_text(text):
    """
    NLP preprcessing
    """
    text = text.replace("|", " ")
    lower_text = str(text).lower()

    punctuation_remove = re.sub(f"[{re.escape(string.punctuation)}]","",lower_text)
    
    extra_space_remove = re.sub(r'\s+', ' ',punctuation_remove).strip()
    lemitize_text = [lemitizer.lemmatize(word) for word in extra_space_remove.split(" ")]
    return " ".join(lemitize_text)


#apply runtime categorization
df['runtime_category'] = df['runtime'].apply(runtime_category)

#combine necessary columns
df["all_combined"]= df.apply(join_all, axis=1)

#apply text cleaning
dataset = df["all_combined"].apply(clean_text)

#Perform TfIDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
x = vectorizer.fit_transform(dataset)

#Saving model
with open("model.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

#saving vectors
with open("movies_vectors.pkl", "wb") as f:
   pickle.dump(x, f)