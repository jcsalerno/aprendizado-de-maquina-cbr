import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# --------------------------------------------------------------------------------
# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# --------------------------------------------------------------------------------

df = pd.read_csv("../data/Training.csv")
symptom_columns = [col for col in df.columns if col != "prognosis"]

# Converte os dados para um formato que o TfidfVectorizer entende:
# Cada caso se torna uma única string com os nomes dos sintomas separados por espaços.
corpus = []
for i, row in df.iterrows():
    sintomas = [s for s in symptom_columns if row[s] == 1]
    corpus.append(" ".join(sintomas))

# --------------------------------------------------------------------------------
# CÁLCULO DOS PESOS IDF (Inverse Document Frequency)
# --------------------------------------------------------------------------------

# Inicializa o vetorizador TF-IDF.
# `use_idf=True` garante que os pesos IDF sejam calculados.
# `smooth_idf=True` evita divisões por zero.
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)

# Calcula a matriz TF-IDF.
# As linhas são os casos, as colunas são os sintomas.
tfidf_matrix = vectorizer.fit_transform(corpus)

# Cria um dicionário que mapeia cada sintoma (feature) ao seu peso IDF.
# Este dicionário é a nossa "inteligência" sobre a importância de cada sintoma.
feature_names = vectorizer.get_feature_names_out()
idf_weights = dict(zip(feature_names, vectorizer.idf_))

# --------------------------------------------------------------------------------
# NOVA FUNÇÃO DE SIMILARIDADE PONDERADA (JACCARD COM TF-IDF)
# --------------------------------------------------------------------------------

def weighted_jaccard_similarity(set1, set2, weights):
    """
    Calcula a similaridade de Jaccard PONDERADA.
    A interseção e a união são calculadas somando os pesos IDF dos sintomas.
    """
    # Soma dos pesos IDF dos sintomas em comum (interseção).
    intersection_weight = sum(weights.get(symptom, 0) for symptom in set1.intersection(set2))
    
    # Soma dos pesos IDF de todos os sintomas únicos em ambos os conjuntos (união).
    union_weight = sum(weights.get(symptom, 0) for symptom in set1.union(set2))
    
    if union_weight == 0:
        return 0.0
        
    return intersection_weight / union_weight

# --------------------------------------------------------------------------------
# ESTRUTURA DO TESTE LEAVE-ONE-OUT (USANDO A NOVA SIMILARIDADE)
# --------------------------------------------------------------------------------

all_cases = []
for _, row in df.iterrows():
    diagnostico = row["prognosis"]
    all_cases.append({
        "description": {
            "sintomas": {s for s in symptom_columns if row[s] == 1},
            "diagnostico": diagnostico
        }
    })

total_de_casos = len(all_cases)
acertos = 0

print(f"🚀 Iniciando o teste Leave-One-Out MELHORADO com {total_de_casos} casos...")

for i in tqdm(range(total_de_casos), desc="Testando Casos"):
    caso_atual_como_consulta = all_cases[i]
    diagnostico_real = caso_atual_como_consulta["description"]["diagnostico"]
    query_symptoms = caso_atual_como_consulta["description"]["sintomas"]
    
    base_de_casos_loo = all_cases[:i] + all_cases[i+1:]

    # Calcula as similaridades usando a nova função ponderada.
    similarities = [
        weighted_jaccard_similarity(query_symptoms, case["description"]["sintomas"], idf_weights)
        for case in base_de_casos_loo
    ]
    
    if not similarities:
        continue
        
    indice_do_mais_similar = np.argmax(similarities) # Usar np.argmax é mais seguro
    caso_recuperado = base_de_casos_loo[indice_do_mais_similar]
    diagnostico_previsto = caso_recuperado["description"]["diagnostico"]

    if diagnostico_previsto == diagnostico_real:
        acertos += 1

acuracia = (acertos / total_de_casos) * 100

print("\n" + "="*50)
print("📊 RESULTADO DO TESTE LEAVE-ONE-OUT (COM TF-IDF) 📊")
print("="*50)
print(f"Total de Casos Testados: {total_de_casos}")
print(f"Total de Acertos: {acertos}")
print(f"Nova Acurácia do Sistema: {acuracia:.2f}%")
print("="*50)

