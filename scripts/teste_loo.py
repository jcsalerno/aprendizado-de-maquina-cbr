import pandas as pd
from tqdm import tqdm # Biblioteca para mostrar uma barra de progresso (instale com: pip install tqdm)

# --------------------------------------------------------------------------------
# REUTILIZANDO AS FUNÇÕES E A LÓGICA DO SEU SISTEMA CBR
# (Copiamos as mesmas funções do outro arquivo)
# --------------------------------------------------------------------------------

def jaccard_similarity(set1, set2):
    if not set1 and not set2: return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

def global_similarity_fn(query, case):
    weights = {"sintomas": 0.7, "diagnostico": 0.3}
    sintomas_sim = jaccard_similarity(query["sintomas"], case["description"]["sintomas"])
    # No teste LOO, a consulta não tem um diagnóstico "verdadeiro", então focamos nos sintomas.
    # Ou, para um teste mais justo, podemos simular que o diagnóstico da consulta é o mesmo do caso.
    # Vamos usar uma abordagem mais simples e focar apenas nos sintomas para a recuperação.
    # Isso evita que o sistema "trapaceie" usando o próprio diagnóstico da consulta.
    # Portanto, a similaridade global será apenas a similaridade dos sintomas.
    return jaccard_similarity(query["sintomas"], case["description"]["sintomas"])

# 1. Carregar os dados e criar a base de casos
df = pd.read_csv("../data/kaggle-base.csv")
symptom_columns = [col for col in df.columns if col != "prognosis"]

all_cases = []
for _, row in df.iterrows():
    diagnostico = row["prognosis"]
    all_cases.append({
        "description": {
            "sintomas": {s for s in symptom_columns if row[s] == 1},
            "diagnostico": diagnostico
        },
        "solution": {
            "tratamento": f"Tratamento padrão para {diagnostico}"
        }
    })

# --------------------------------------------------------------------------------
# IMPLEMENTAÇÃO DO TESTE LEAVE-ONE-OUT
# --------------------------------------------------------------------------------

total_de_casos = len(all_cases)
acertos = 0

print(f"🚀 Iniciando o teste Leave-One-Out com {total_de_casos} casos...")

# Usamos tqdm para ter uma barra de progresso bonita no terminal.
for i in tqdm(range(total_de_casos), desc="Testando Casos"):
    # Passo 1: Pegue um caso para ser a "consulta"
    caso_atual_como_consulta = all_cases[i]
    diagnostico_real = caso_atual_como_consulta["description"]["diagnostico"]
    
    # A consulta para o sistema CBR são apenas os sintomas do caso atual.
    query = {"sintomas": caso_atual_como_consulta["description"]["sintomas"]}

    # Passo 2: Crie a base de casos para esta iteração (todos os casos, EXCETO o atual)
    base_de_casos_loo = all_cases[:i] + all_cases[i+1:]

    # Passo 3: Recupere o caso mais similar (k=1)
    similarities = [global_similarity_fn(query, case) for case in base_de_casos_loo]
    
    # Encontra o índice do caso mais similar na base_de_casos_loo
    if not similarities:
        continue # Pula se a lista de similaridades estiver vazia
        
    indice_do_mais_similar = similarities.index(max(similarities))
    caso_recuperado = base_de_casos_loo[indice_do_mais_similar]
    diagnostico_previsto = caso_recuperado["description"]["diagnostico"]

    # Passo 4: Verifique se foi um acerto
    if diagnostico_previsto == diagnostico_real:
        acertos += 1

# Passo 5: Calcule e exiba a acurácia final
acuracia = (acertos / total_de_casos) * 100

print("\n" + "="*50)
print("📊 RESULTADO DO TESTE LEAVE-ONE-OUT 📊")
print("="*50)
print(f"Total de Casos Testados: {total_de_casos}")
print(f"Total de Acertos: {acertos}")
print(f"Acurácia do Sistema: {acuracia:.2f}%")
print("="*50)

