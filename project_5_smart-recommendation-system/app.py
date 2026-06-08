# app.py

import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

data = pd.read_csv("dataset.csv")

tfidf = TfidfVectorizer(stop_words='english')

matrix = tfidf.fit_transform(data['description'])

similarity = cosine_similarity(matrix)

def recommend(course_name):

    if course_name not in data['title'].values:
        return []

    index = data[data['title'] == course_name].index[0]

    scores = list(enumerate(similarity[index]))

    sorted_scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )[1:4]

    recommendations = []

    for item in sorted_scores:
        recommendations.append(
            data.iloc[item[0]]['title']
        )

    return recommendations


@app.route('/')
def home():
    return render_template(
        'index.html',
        courses=data['title'].tolist()
    )

@app.route('/recommend', methods=['POST'])
def get_recommendation():

    course = request.form['course']

    result = recommend(course)

    return render_template(
        'result.html',
        recommendations=result,
        selected=course
    )

if __name__ == '__main__':
    app.run(debug=True)