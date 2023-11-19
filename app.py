import spacy
nlp = spacy.load('pt_core_news_lg')
from flask import Flask, render_template
import pandas as pd
import numpy as np
import string
import json
from rank_bm25 import BM25Okapi
from flask_bootstrap import Bootstrap5
from flask import Flask
from flask import jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
app = Flask(__name__)
bootstrap = Bootstrap5(app)

# Ler os DataFrames uma vez e armazenar como variáveis globais
df_senadores = pd.read_csv('df_senadores.csv')
df_sugestoes = pd.read_csv('df_sugestoes_bm25_geral.csv')
df_interesses = pd.read_csv('interesse.csv')
df_detalhes_materias = pd.read_csv('df_detalhes_materias.csv')
df_autorias = pd.read_csv('df_autorias.csv')

def obter_dataframe_senadores():
    return df_senadores

def obter_dataframe_sugestoes():
    return df_sugestoes

def obter_dataframe_detalhe_materia():
    return df_detalhes_materias

def obter_dataframe_interesses(id):
    # Filtrar os 10 principais interesses para o parlamentar específico
    interesses_para_senador = df_interesses[df_interesses['CodigoParlamentar'] == id].sort_values(by='interesse', ascending=False).head(10)
    return interesses_para_senador

@app.route('/')
def pagina_inicial():
    senadores = obter_dataframe_senadores().to_dict(orient='records')
    return render_template('pagina_inicial.html', senadores=senadores)

if __name__ == '__main__':
    app.run(debug=True)
