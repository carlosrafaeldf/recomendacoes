import spacy
nlp = spacy.load('pt_core_news_lg')
from flask import Flask, render_template, request
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
    DataFrameMaterias_filtrado['ValorSimilaridade'] = pd.to_numeric(DataFrameMaterias_filtrado['ValorSimilaridade'], errors='coerce') * 100
    DataFrameMaterias_filtrado['ValorSimilaridade'] = DataFrameMaterias_filtrado['ValorSimilaridade'].round(2)
        

    # DataFrameMaterias_filtrado agora contém as linhas filtradas com a coluna de similaridade
    #Filtra para conter apenas os valores contidos em DataFrameMateriasAlvo
    condicao = DataFrameMaterias_filtrado['CodigoMateria'].isin(DataFrameMateriasAlvo['CodigoMateria'])
    DataFrameMaterias_filtrado = DataFrameMaterias_filtrado[condicao]

    #Filtra para conter apenas as matérias que começam com P e estão tramitando
    condicao = ((DataFrameMaterias_filtrado['Identificacao'].str.startswith('P')) &
                (DataFrameMaterias_filtrado['Tramitando'] == 'Sim') &
                (DataFrameMaterias_filtrado['CodigoMateria'] != codigo_materia) &
                (~DataFrameMaterias_filtrado['ClasseHierarquica'].str.contains('Orçamento Anual',na=False)) & 
                (~DataFrameMaterias_filtrado['Natureza'].str.contains('Concessão',na=False)))
    DataFrameMaterias_filtrado = DataFrameMaterias_filtrado[condicao]    
    DataFrameMaterias_filtrado = DataFrameMaterias_filtrado.replace({np.nan: None})
    
    return DataFrameMaterias_filtrado.head(6)
    

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

    limite_similaridade = 0.5

    # Filtra os documentos com escores BM25 acima do limite
    documentos_selecionados = [i for i, score in enumerate(bm25_scores) if score > limite_similaridade]


    descending_indices = (-bm25_scores).argsort()
    print("Indices: ",descending_indices)
    dataframe_filtrado = df_detalhes_materias.filter(items=documentos_selecionados, axis=0)
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

        if achou >= 12:
            break
        if total_escapa > 500:
            print('escapei achando', achou)
            break
    
    # Verificar se o resultado é uma Series (uma linha) e converter para DataFrame se necessário
    if isinstance(df_retorno, pd.Series):
        df_retorno = pd.DataFrame([df_retorno])
    
    if df_retorno.empty:
        return df_retorno
    else:
        df_retorno = df_retorno.head(12)[['Identificacao','Ementa','CodigoMateria','justificativa_prioridade']]    
        df_retorno = df_retorno.replace({np.nan: None})
    return df_retorno
    

@app.route('/')
def pagina_inicial():
    senadores = obter_dataframe_senadores().to_dict(orient='records')
    return render_template('pagina_inicial.html', senadores=senadores)

@app.route('/senador/<int:id>')
def pagina_detalhes(id):
    senador = obter_dataframe_senadores()[obter_dataframe_senadores()['CodigoParlamentar'] == id].to_dict(orient='records')[0]    
    #sugestoes_para_senador = obter_dataframe_sugestoes()[(obter_dataframe_sugestoes()['CodigoParlamentar'] == id)].head(12).to_dict(orient='records')
    df_retorno = obter_dataframe_sugestoes()[(obter_dataframe_sugestoes()['CodigoParlamentar'] == id)]
    df_retorno = df_retorno.replace({np.nan: None})
    df_retorno = df_retorno.head(12).to_dict(orient='records')
    ##sugestoes_para_senador = obter_dataframe_sugestoes()[(obter_dataframe_sugestoes()['CodigoParlamentar'] == id) & (obter_dataframe_sugestoes()['prioridade'] == 'Não')].head(12).to_dict(orient='records')
    #sugestoes_prioritarias = obter_dataframe_sugestoes()[(obter_dataframe_sugestoes()['CodigoParlamentar'] == id) & (obter_dataframe_sugestoes()['prioridade'] == 'Sim')].head(12).to_dict(orient='records')
    return render_template('pagina_detalhes.html', senador=senador, sugestoes=df_retorno)

@app.route('/senador/<int:id>/interesses')
def pagina_interesses(id):
    interesses_para_senador = obter_dataframe_interesses(id).to_dict(orient='records')
    senador = obter_dataframe_senadores()[obter_dataframe_senadores()['CodigoParlamentar'] == id].to_dict(orient='records')[0]
    return render_template('pagina_interesses.html',senador=senador, interesses=interesses_para_senador,similares_sugeridas=obter_similares_sugeridas_coseno)

@app.route('/senador/<int:id>/detalhes_materias/<int:codigo_materia>')
def pagina_detalhes_materias(id, codigo_materia):
    interesse_para_senador = obter_dataframe_interesses(id)[obter_dataframe_interesses(id)['CodigoMateria'] == codigo_materia]
    
    if not interesse_para_senador.empty:
        interesse_para_senador = interesse_para_senador.to_dict(orient='records')[0]
    else:
        interesse_para_senador = None
    
    senador = obter_dataframe_senadores()[obter_dataframe_senadores()['CodigoParlamentar'] == id].to_dict(orient='records')[0]
    detalhe_materia = obter_dataframe_detalhe_materia()[obter_dataframe_detalhe_materia()['CodigoMateria'] == codigo_materia].to_dict(orient='records')[0]
    return render_template('pagina_detalhes_materias.html',senador=senador, materia=detalhe_materia,similares_sugeridas=obter_similares_sugeridas_coseno,interesse=interesse_para_senador )

@app.route('/senador/<int:id>/pesquisar/', methods=['GET'])
def pesquisar_similares(id):    
    query = request.args.get('query', '')
    df_retorno = obter_similares_pesquisa(id,query)    
    df_retorno = df_retorno.replace({np.nan: None})
    df_retorno = df_retorno.to_dict(orient='records')    
    senador = obter_dataframe_senadores()[obter_dataframe_senadores()['CodigoParlamentar'] == id].to_dict(orient='records')[0]
    return render_template('pagina_resultado_pesquisa.html', resultados_pesquisa=df_retorno, senador=senador)
    
if __name__ == '__main__':
    app.run(debug=True)
