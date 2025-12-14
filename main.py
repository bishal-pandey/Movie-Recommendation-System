from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import streamlit as st
import numpy as np
import string
import pandas as pd
import pickle
import re


 
st.markdown("""
    <style>
     .block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
    padding-left: 0;
    padding-right: 0;
    }       

    .title-style {
        color: #BBE44C;
        font-weight: 700;
        letter-spacing: 2px;
        }      

    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding:5px 12px;
        border-radius: 8px;
        border: none;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        height:40px;
    }
    .movie-detail{
        background: #fffff;
        padding: 7px 10px;
        border-radius: 15px;
        border: 1px solid #e6e6e6;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        font-family: 'Arial', sans-serif;    
        }
    .movie-detail p{
            font-size:16px;
        }
    .movie-detail span{
            font-size:18px;
            font-weight: 700;
            color: #8D604F;
        }
    h3{
           font-size:28px;
            font-weight: 500;
            font-family: "Courier New";
            color: #E27375; 
            margin-top: 167px;
            margin-left: 136px;
            }
    
    </style>
""", unsafe_allow_html=True)


images = pd.read_csv(r"D:\DATA\archive (5)\TMDB_Dataset\movie_posters.csv")
df = pd.read_csv("C:\\Users\\Dell\\Desktop\\MovieRecommend\\Data\\clean_movies_data.csv")

with open("model.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("movies_vectors.pkl", "rb") as f:
    movies_vectors = pickle.load(f)
# movies_vectors = np.load('movies_vectors.npy', allow_pickle=True)

lemmatizer = WordNetLemmatizer()


def clean_text(text):
    lower_text = str(text).lower()
    punctuation_remove = re.sub(f"[{re.escape(string.punctuation)}]", "", lower_text)
    extra_space_remove = re.sub(r"\s+", " ", punctuation_remove).strip()
    clean_text = ' '.join([lemmatizer.lemmatize(word) for word in extra_space_remove.split()])
    return clean_text

def recommand_movies(text):
    sim_threshold = True
    text = clean_text(text)
    y = vectorizer.transform([text])
    cosine_sim = list(enumerate(cosine_similarity(movies_vectors, y).flatten()))
    sort = sorted(cosine_sim, reverse=True, key = lambda x: x[1])
    if sort[0][1] < 0.20:
        return None
    index = [idx[0] for idx in sort[:11]]
    movie = df[['title','movie_id']].iloc[index]

    return movie

# st.title("Movie Recommendation System")
st.markdown("<h1 class = 'title-style'>Movie Recommendation System</h1>",unsafe_allow_html=True)
with st.container():
    col1, col2 = st.columns([0.7, 0.3], gap="large")

search = col1.text_input("input",placeholder="Enter movie title or keywords:",label_visibility="collapsed")
btn = col2.button('Search Movies')


content = st.empty()
if not btn:
    with content.container():
        st.header('Popular Movie')
        row1 = st.columns([0.6,0.4],gap="small")
        row2 = st.columns(4,gap="medium")
        row3 = st.columns(4,gap="medium")
        popular_movie = df.sort_values(by="popularity")[:8]
        for i,col in enumerate(row2+row3):
            movie_id = popular_movie["movie_id"].iloc[i]
            movie_name = popular_movie["title"].iloc[i]
            
            poster_path = images[images["movie_id"]==movie_id]["poster_url_original"].values[0]

            col.image(poster_path,width=144)
            col.write(movie_name)


else:
    if search:
        with content.container():
            if search.lower() in df['title'].str.lower().to_numpy():
                movie_df = df[df["title"].str.lower()==search.lower()]
                movie_df = movie_df.sort_values(by="popularity",ascending=False).reset_index()
                movie_name = movie_df.iloc[0]["title"]
                search_movie_id = movie_df.iloc[0]["movie_id"]
                actors = ", ".join(movie_df.iloc[0]["top_cast"].split("|")[:5])
                director = movie_df.iloc[0]["director"]
                runtime = movie_df.iloc[0]["runtime"]
                genre = ", ".join(movie_df.iloc[0]["genres"].split("|"))
                release_year = movie_df.iloc[0]["release_year"]
                spoken_languages = ", ".join(movie_df.iloc[0]["spoken_languages"].split("|"))
                overview = movie_df.iloc[0]["overview"]
                writers = movie_df.iloc[0]["writers"]
                
                row1 = st.columns([0.3,0.4])
                row2 = st.columns(4,gap="medium")
                row3 = st.columns(4,gap="medium")
                with row1[0]:
                    poster_path = images[images["movie_id"]==search_movie_id]["poster_url_original"].values[0]
                    row1[0].image(poster_path,width=230)
                    row1[0].write(movie_name)

                with row1[1]:

                    row1[1].markdown(f"""
                            <div class='movie-detail'>
                                     <p><span>Director: </span> {director}</p>
                                     <p><span>Genre: </span> {genre}</p>
                                     <p><span>Top Cast: </span> {actors}</p>
                                     <p><span>Release Year: </span> {release_year}</p>
                                     <p><span>Spoken Languages: </span> {spoken_languages.upper()}</p>
                                     <p><span>Runtime: </span> {runtime}min</p>
                            </div>
                                """, unsafe_allow_html=True)
                    # row1[1].write(f"**Director:** {director}")
                    # row1[1].write(f"**Genre:** {genre}")
                    # row1[1].write(f"**Top Cast:** {actors}")
                    # row1[1].write(f"**Release Year:** {release_year}")
                    # row1[1].write(f"**Spoken Languages:** {spoken_languages.upper()}")
                    # row1[1].write(f"**Runtime:** {runtime} minutes")

                search_movie_desc_combine = movie_name +" "+ overview +" "+ str(release_year) +" "+genre.replace(",","")+" " +actors.replace(",","") +" " + director.replace(",","")+" "+writers.replace("|"," ") +" " + spoken_languages.replace(",","")
                for i,col in zip(recommand_movies(search_movie_desc_combine).iloc[1:].values, row2+row3):
                    movie_name = i[0]
                    movie_id = i[1]
                   
                    poster_path = images[images["movie_id"]==movie_id]["poster_url_original"].values[0]
                    col.image(poster_path,width=144)
                    col.write(movie_name)
            
            else:
                row2 = st.columns(4,gap="medium")
                row3 = st.columns(4,gap="medium")
                rec_movie = recommand_movies(search)
                if rec_movie is None:
                    st.markdown("<h3>No Result Found</h3>",unsafe_allow_html=True)
                else:
                    for i,col in zip(rec_movie.values, row2+row3):
                        movie_name = i[0]
                        movie_id = i[1]
                        
                        poster_path = images[images["movie_id"]==movie_id]["poster_url_original"].values[0]

                        col.image(poster_path,width=144)
                        col.write(movie_name)
    else:
        with content.container():
            row1 = st.columns([0.6,0.4],gap="small")
            row2 = st.columns(4,gap="medium")
            row3 = st.columns(4,gap="medium")
            popular_movie = df.sort_values(by="popularity")[:20].sample(frac=1).reset_index(drop=True)
            for i,col in enumerate(row2+row3):
                movie_id = popular_movie["movie_id"].iloc[i]
                movie_name = popular_movie["title"].iloc[i]
                
                poster_path = images[images["movie_id"]==movie_id]["poster_url_original"].values[0]

                col.image(poster_path,width=144)
                col.write(movie_name)


