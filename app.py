from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


with open('mt_predict.pkl', 'rb') as file:
    mt_predict = pickle.load(file)

corpus = api.load('text8')
model = Word2Vec(corpus)

def app():
    app = Flask(__name__)
    return app


vocab = list(model.wv.index_to_key)
def vectorize_title(title, model):
    words = title.split()  # Split title into words
    words = [word.lower() for word in words]
    words = [word for word in words if word in vocab]
    words = [word for word in words if word not in stop_words]
    word_vectors = [model.wv[word] for word in words]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)  # Return zero vector if no words are in the model
    return np.mean(word_vectors, axis=0)

def test_title(title):
    title_len = len(title.split())
    vectors = vectorize_title(title, model)
    frames = []
    for val in vectors:
        frames.append(pd.Series(val))
    frames.append(pd.Series(title_len))
    new_features = pd.concat(frames, axis=1)
    new_features = np.asarray(new_features)
    predicted_claps = mt_predict.predict(new_features)
    return predicted_claps

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/title',methods=['GET', 'POST'])
def title():
    if request.method == 'POST':
        
        title = request.json['title'] 
        clap_count = str(test_title(title))
        return jsonify(MediumClaps=clap_count)

if __name__ == '__main__':
    app.run(debug=True)