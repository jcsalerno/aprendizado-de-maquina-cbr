import pandas as pd

def jaccard_similarity(set1, set2):
    """Calcula a similaridade de Jaccard entre dois conjuntos."""
    if not set1 and not set2: return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

def global_similarity_fn(query, case):
    """Calcula a similaridade global ponderada entre a consulta e um caso."""
    weights = {"sintomas": 0.7, "diagnostico": 0.3}
    
    # Similaridade local para os sintomas
    sintomas_sim = jaccard_similarity(query["sintomas"], case["description"]["sintomas"])
    
    # Similaridade local para o diagnóstico
    diagnostico_sim = 1.0 if query["diagnostico"] == case["description"]["diagnostico"] else 0.0
    
    # Agregação ponderada para a similaridade global
    return (sintomas_sim * weights["sintomas"]) + (diagnostico_sim * weights["diagnostico"])

df = pd.read_csv("../data/kaggle-base.csv")
symptom_columns = [col for col in df.columns if col != "prognosis"]

cases = []
for _, row in df.iterrows():
    diagnostico = row["prognosis"]
    cases.append({
        "description": {
            "sintomas": {s for s in symptom_columns if row[s] == 1},
            "diagnostico": diagnostico
        },
        "solution": {
            "tratamento": f"Tratamento padrão para {diagnostico}"
        }
    })

#Definir a consulta (query)
query = {
    "sintomas": {"itching", "skin_rash"},
    "diagnostico": "Fungal infection"
}

# Passo de Recuperação (Retrieve) Manual
# Calculamos a similaridade entre a consulta e TODOS os casos da base.
similarities = [global_similarity_fn(query, case) for case in cases]

# Criamos o ranking dos casos mais similares (k=3).
# `enumerate` nos dá o índice e o valor, e `sorted` ordena pela similaridade.
ranking = sorted(enumerate(similarities), key=lambda item: item[1], reverse=True)
top_3_ranking = ranking[:3]

# Passo de Reuso (Reuse) Manual
print("🔎 Casos mais semelhantes (implementação manual completa):")
for idx, sim in top_3_ranking:
    retrieved_case = cases[idx]
    # Reuso da solução do caso recuperado
    print(f"-> Diagnóstico: {retrieved_case['description']['diagnostico']} | Similaridade: {sim:.2f}")
    print(f"   Tratamento Sugerido: {retrieved_case['solution']['tratamento']}")

