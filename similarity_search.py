from flask import Flask, render_template, url_for, redirect
from flask_cors import CORS
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length
import os
import pandas as pd
import pickle
import numpy as np
import faiss
from sklearn.metrics import ndcg_score, dcg_score, average_precision_score
from sentence_transformers import SentenceTransformer


r = pd.read_csv('papers.csv')

app = Flask(__name__)

SECRET_KEY = 'sim'
app.config['SECRET_KEY'] = SECRET_KEY


def vector_search(query):
    
  
  DOCUMENTS = list(r.abstract)  
    
  model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

  QUERY_STR = query

  query =  model.encode([QUERY_STR])

  embeddings = pickle.load(open('embeddings.pkl','rb'))

  index = faiss.IndexFlatL2(embeddings.shape[1]) 

  index.add(np.ascontiguousarray(embeddings))

  D, I = index.search(query, 5) 
 
  print(f'L2 distance: {D.flatten().tolist()}\n\nMAG paper IDs: {I.flatten().tolist()}')
    
  return I[0]


def get_documents(ids):
    
    DOCUMENTS = list(r.abstract)
    results = []
    for i in ids:
        results.append(DOCUMENTS[i])

    return results
 
results = []


class SearchForm(FlaskForm):
    query = StringField('', validators=[DataRequired()])
    submit = SubmitField('Search')


@app.route("/")
@app.route("/home", methods=['GET', 'POST'])
def home():
    form = SearchForm()
    if form.validate_on_submit():
        query = form.query.data
        global results
        #results = [query]
        ids = vector_search(query)
        results = get_documents(ids)
        return redirect(url_for('search_result'))
    return render_template('home.html', form=form)



@app.route("/search_result", methods=['GET', 'POST'])
def search_result():
    form = SearchForm()
    if form.validate_on_submit():
        query = form.query.data
        global results
        #results = [query]
        ids = vector_search(query)
        results = get_documents(ids)
        return redirect(url_for('search_result'))

    return render_template('search_result.html', form=form, results=results)



if __name__ == '__main__':
    app.run(debug=True)
