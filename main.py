from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

app = Flask(__name__)

WEIGHTS = {
    "genre": 5,
    "score": 3,
    "names": 2,
    "crew": 2,
    "overview": 2,
    "country": 2
}

# Pozostałe wagi domyślnie 0

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / union if union != 0 else 0

def kgrams(term, k):
    return set(term[i:i+k] for i in range(len(term) - k + 1))

def correct_query(query, movies, k=2, threshold=0.5):
    query = query.lower()
    query_kgrams = kgrams(query, k)

    for movie in movies:
        if query in movie.values():
            return query

    potential_corrections = []
    for movie in movies:
        for key, value in movie.items():
            if isinstance(value, str):
                values = [value]
            elif isinstance(value, list):
                values = value
            else:
                continue

            for item in values:
                item_kgrams = kgrams(item.lower(), k)
                similarity = jaccard_similarity(query_kgrams, item_kgrams)
                if similarity >= threshold:
                    potential_corrections.append((item, similarity))

    if potential_corrections:
        return max(potential_corrections, key=lambda x: x[1])[0]

    return query

def calculate_tf_idf(movies):
    documents = []
    for movie in movies:
        document = " ".join(
            f"{key} {' '.join(str(movie[key]).split()) * WEIGHTS[key]}" 
            for key in WEIGHTS if movie[key] and WEIGHTS[key] > 0
        )
        documents.append(document)

    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(documents)
    return tf_idf_matrix, vectorizer

def search_movies(query, tf_idf_matrix, vectorizer, movies):
    query_transformed = vectorizer.transform([query])
    cosine_similarities = np.dot(tf_idf_matrix, query_transformed.T).toarray().ravel()
    top_indices = np.argsort(cosine_similarities)[::-1][:50]
    top_movies = [(movies[i], cosine_similarities[i]) for i in top_indices]
    return top_movies

@app.route('/', methods=['GET', 'POST'])
def movie_search():
    if request.method == 'POST':
        user_query = request.form.get('user_query', '').strip()
        genre = request.form.getlist('genre')
        score = int(request.form.get('score', 0))

        # Load data from CSV file
        df = pd.read_csv('imdb_movies.csv')

        # Filter data based on user input
        filtered_movies = df.copy()
        if genre:
            filtered_movies = filtered_movies[filtered_movies['genre'].str.contains('|'.join(genre))]
        if score:
            filtered_movies = filtered_movies[filtered_movies['score'] >= score]

        # Convert DataFrame to list of dictionaries
        movies = filtered_movies.to_dict(orient='records')

        corrected_query = correct_query(user_query, movies)
        tf_idf_matrix, vectorizer = calculate_tf_idf(movies)
        top_movies = search_movies(corrected_query, tf_idf_matrix, vectorizer, movies)

        return render_template('quiz.html', results=top_movies)

    return render_template('quiz.html')

if __name__ == "__main__":
    app.run(debug=True)
