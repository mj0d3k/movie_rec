from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

app = Flask(__name__)

WEIGHTS = {
    "names": 5,
    "date_x": 2,
    "score": 3,
    "genre": 3,
    "overview": 2,
    "crew": 2,
    "orig_title": 2,
    "status": 1,
    "orig_lang": 1,
    "budget_x": 1,
    "revenue": 1,
    "country": 1
}

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
        threshold = float(request.form.get('threshold', 0.5))
        number_of_results = int(request.form.get('num_results', 50))

        # Load data from CSV file
        df = pd.read_csv('movies.csv')

        # Convert DataFrame to list of dictionaries
        movies = df.to_dict(orient='records')

        corrected_query = correct_query(user_query, movies)
        tf_idf_matrix, vectorizer = calculate_tf_idf(movies)
        top_movies = search_movies(corrected_query, tf_idf_matrix, vectorizer, movies)
        top_movies = top_movies[:number_of_results]

        return render_template('view1.html', results=top_movies)

    return render_template('view1.html')

if __name__ == "__main__":
    app.run(debug=True)
