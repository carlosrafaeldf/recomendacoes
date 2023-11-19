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

df_detalhes_materias["combinacao"] = df_detalhes_materias["combinacao"].astype(str)
corpus_parla = df_detalhes_materias["combinacao"]
corpus_parla = [doc.translate(str.maketrans('', '', string.punctuation)).replace('\n',' ').lower() for doc in corpus_parla]
tokenized_corpus_parla = [doc.split(" ") for doc in corpus_parla]
bm25_parla = BM25Okapi(tokenized_corpus_parla)

#Similaridade de coseno
lista_stop_words = stopwords.words('portuguese') + ['lei','altera','modifica']
cv = TfidfVectorizer(stop_words=lista_stop_words, use_idf=True)
count_matrix = cv.fit_transform(df_detalhes_materias["combinacao"])
cosine_sim_normal = cosine_similarity(count_matrix)

# Realizar merge entre df_interesses e df_detalhes_materias
df_interesses = pd.merge(df_interesses, df_detalhes_materias, on='CodigoMateria')

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

def obter_similares_sugeridas_prioritarias(codigo_materia, codigo_parlamentar):
    # Lógica para obter matérias similares sugeridas
    # Substitua a lógica abaixo pela implementação real
    
    bm25 = bm25_parla
    query = df_detalhes_materias[df_detalhes_materias['CodigoMateria'] == codigo_materia]['combinacao'].to_string(index=False)
    
    query = query.translate(str.maketrans('', '', string.punctuation)).replace('\n',' ').lower()
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    descending_indices = (-bm25_scores).argsort()
    dataframe_filtrado = df_detalhes_materias.filter(items=descending_indices, axis=0)
    dataframe_filtrado = dataframe_filtrado[dataframe_filtrado['Identificacao'].str.startswith('P') and dataframe_filtrado['prioridade'] == 'Sim']
    
    
    #Define a base que será usada para alvo    
    df_detalhes_materias_teste = pd.DataFrame(df_detalhes_materias)

    #Remove as autorias da base de teste    
    df_autorias_parla = df_autorias[df_autorias['CodigoParlamentar'] == codigo_parlamentar]    
    codigos_autorias_parla = df_autorias_parla['CodigoMateria'].unique()
    df_detalhes_materias_teste = df_detalhes_materias_teste[~df_detalhes_materias_teste['CodigoMateria'].isin(codigos_autorias_parla)]
    
    
    #Aplica as condições na base de teste
    condicao = ((df_detalhes_materias_teste['Identificacao'].str.startswith('P')) &
                (df_detalhes_materias_teste['Tramitando'] == 'Sim') &
                (df_detalhes_materias_teste['CodigoMateria'] != codigo_materia) &
                (~df_detalhes_materias_teste['ClasseHierarquica'].str.contains('Honorífico|Orçamento Anual',na=False)) & 
                (~df_detalhes_materias_teste['Natureza'].str.contains('Concessão',na=False)))
    df_detalhes_materias_teste = df_detalhes_materias_teste[condicao]
    
    achou = 0
    total_escapa = 0
    df_retorno = pd.DataFrame()

    for index, row in dataframe_filtrado.iterrows():
        if df_detalhes_materias_teste['CodigoMateria'].isin([row['CodigoMateria']]).any():
            achou += 1
            df_retorno = pd.concat([df_retorno, pd.DataFrame(row).T])

        total_escapa += 1

        if achou >= 5:
            break
        if total_escapa > 500:
            print('escapei achando', achou)
            break
    
    # Verificar se o resultado é uma Series (uma linha) e converter para DataFrame se necessário
    if isinstance(df_retorno, pd.Series):
        df_retorno = pd.DataFrame([df_retorno])

    return df_retorno

def get_index_materia(DataFrameMaterias,CodigoMateria):
    
    return DataFrameMaterias[DataFrameMaterias.CodigoMateria == int(CodigoMateria)].index.values[0]

def obter_similares_sugeridas_coseno(codigo_materia, codigo_parlamentar):
    DataFrameCosine = cosine_sim_normal
    DataFrameMaterias = df_detalhes_materias
    CodigoMateria = codigo_materia
    DataFrameMateriasAlvo = df_detalhes_materias
    
    indice = get_index_materia(DataFrameMaterias,CodigoMateria)    
    similares = list(enumerate(DataFrameCosine[indice]))    
    similares = sorted(similares, key=lambda x:x[1], reverse=True)        
    df_similares = pd.DataFrame(similares, columns=['Indice', 'ValorSimilaridade'])    
    df_similares = df_similares.set_index('Indice')    
    DataFrameMaterias_filtrado = DataFrameMaterias.loc[df_similares.index]    
    DataFrameMaterias_filtrado['ValorSimilaridade'] = df_similares['ValorSimilaridade']
    
    # DataFrameMaterias_filtrado agora contém as linhas filtradas com a coluna de similaridade

    #Filtra para conter apenas os valores contidos em DataFrameMateriasAlvo
    condicao = DataFrameMaterias_filtrado['CodigoMateria'].isin(DataFrameMateriasAlvo['CodigoMateria'])
    DataFrameMaterias_filtrado = DataFrameMaterias_filtrado[condicao]

    #Filtra para conter apenas as matérias que começam com P
    DataFrameMaterias_filtrado = DataFrameMaterias_filtrado[DataFrameMaterias_filtrado['Identificacao'].str.startswith('P')]
    DataFrameMaterias_filtrado = DataFrameMaterias_filtrado.replace({np.nan: None})
    
    materias = pd.DataFrame()
    achou = 0;    
    for indice, row in DataFrameMaterias_filtrado.iterrows():
        codigo_materia_pesquisada = row["CodigoMateria"]            
        #esta_tramitando = is_materia_tramitando(codigo_materia_pesquisada)
        esta_tramitando = row["Tramitando"]
        #print('Está:'+esta_tramitando)
        if (esta_tramitando == "Sim"):             
            #materias = materias.append(row)            
            #print(pd.DataFrame(row).T)
            materias = pd.concat([materias,pd.DataFrame(row).T])
            achou = achou + 1    

        if achou >= 5:            
            break  
    
    materias = materias[materias['CodigoMateria'] != CodigoMateria]
    return materias
    
    # Verificar se o resultado é uma Series (uma linha) e converter para DataFrame se necessário
    #if isinstance(df_retorno, pd.Series):
    #    df_retorno = pd.DataFrame([df_retorno])

    #return df_retorno

def obter_similares_sugeridas(codigo_materia, codigo_parlamentar):
    # Lógica para obter matérias similares sugeridas
    # Substitua a lógica abaixo pela implementação real
    
    bm25 = bm25_parla
    query = df_detalhes_materias[df_detalhes_materias['CodigoMateria'] == codigo_materia]['combinacao'].to_string(index=False)
    
    query = query.translate(str.maketrans('', '', string.punctuation)).replace('\n',' ').lower()
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    descending_indices = (-bm25_scores).argsort()
    dataframe_filtrado = df_detalhes_materias.filter(items=descending_indices, axis=0)
    dataframe_filtrado = dataframe_filtrado[(dataframe_filtrado['Identificacao'].str.startswith('P')) ]
    dataframe_filtrado = dataframe_filtrado.replace({np.nan: None})
    
    #Define a base que será usada para alvo    
    df_detalhes_materias_teste = pd.DataFrame(df_detalhes_materias)

    #Remove as autorias da base de teste    
    df_autorias_parla = df_autorias[df_autorias['CodigoParlamentar'] == codigo_parlamentar]    
    codigos_autorias_parla = df_autorias_parla['CodigoMateria'].unique()
    df_detalhes_materias_teste = df_detalhes_materias_teste[~df_detalhes_materias_teste['CodigoMateria'].isin(codigos_autorias_parla)]
    
    
    #Aplica as condições na base de teste
    condicao = ((df_detalhes_materias_teste['Identificacao'].str.startswith('P')) &
                (df_detalhes_materias_teste['Tramitando'] == 'Sim') &
                (df_detalhes_materias_teste['CodigoMateria'] != codigo_materia) &
                (~df_detalhes_materias_teste['ClasseHierarquica'].str.contains('Orçamento Anual',na=False)) & 
                (~df_detalhes_materias_teste['Natureza'].str.contains('Concessão',na=False)))
    df_detalhes_materias_teste = df_detalhes_materias_teste[condicao]
    
    achou = 0
    total_escapa = 0
    df_retorno = pd.DataFrame()

    for index, row in dataframe_filtrado.iterrows():
        if df_detalhes_materias_teste['CodigoMateria'].isin([row['CodigoMateria']]).any():
            achou += 1
            df_retorno = pd.concat([df_retorno, pd.DataFrame(row).T])

        total_escapa += 1

        if achou >= 5:
            break
        if total_escapa > 500:
            print('escapei achando', achou)
            break
    
    # Verificar se o resultado é uma Series (uma linha) e converter para DataFrame se necessário
    if isinstance(df_retorno, pd.Series):
        df_retorno = pd.DataFrame([df_retorno])

    return df_retorno

def obter_similares_pesquisa(codigo_parlamentar, query):    
    bm25 = bm25_parla
        
    query = query.translate(str.maketrans('', '', string.punctuation)).replace('\n',' ').lower()
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    descending_indices = (-bm25_scores).argsort()
    print("Indices: ",descending_indices)
    dataframe_filtrado = df_detalhes_materias.filter(items=descending_indices, axis=0)
    dataframe_filtrado = dataframe_filtrado[dataframe_filtrado['Identificacao'].str.startswith('P')]
    
    #Define a base que será usada para alvo    
    df_detalhes_materias_teste = pd.DataFrame(df_detalhes_materias)
        
    #Aplica as condições na base de teste
    condicao = ((df_detalhes_materias_teste['Identificacao'].str.startswith('P')) &
                (df_detalhes_materias_teste['Tramitando'] == 'Sim') &                
                (~df_detalhes_materias_teste['ClasseHierarquica'].str.contains('Orçamento Anual',na=False)) & 
                (~df_detalhes_materias_teste['Natureza'].str.contains('Concessão',na=False)))
    df_detalhes_materias_teste = df_detalhes_materias_teste[condicao]
    
    achou = 0
    total_escapa = 0
    df_retorno = pd.DataFrame()

    for index, row in dataframe_filtrado.iterrows():
        if df_detalhes_materias_teste['CodigoMateria'].isin([row['CodigoMateria']]).any():
            achou += 1
            df_retorno = pd.concat([df_retorno, pd.DataFrame(row).T])

        total_escapa += 1

        if achou >= 5:
            break
        if total_escapa > 500:
            print('escapei achando', achou)
            break
    
    # Verificar se o resultado é uma Series (uma linha) e converter para DataFrame se necessário
    if isinstance(df_retorno, pd.Series):
        df_retorno = pd.DataFrame([df_retorno])
    
    df_retorno = df_retorno.head(5)[['Identificacao','Ementa','CodigoMateria']]
    df_retorno = df_retorno.replace(np.nan, '')
    df_retorno = df_retorno.to_dict(orient='records')
    return jsonify({'resultados': df_retorno})
    

@app.route('/')
def pagina_inicial():
    senadores = obter_dataframe_senadores().to_dict(orient='records')
    return render_template('pagina_inicial.html', senadores=senadores)

if __name__ == '__main__':
    app.run(debug=True)
