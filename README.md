# Sistema CBR para Diagnóstico de Doenças com Base em Sintomas

## Visão Geral do Projeto

Este projeto foi desenvolvido como parte da disciplina de **Aprendizado de Máquina** do Programa de Pós-Graduação em Ciência da Computação (PPGCC), sob a orientação do **Prof. Luís Alvaro**.

O trabalho implementa um sistema de **Raciocínio Baseado em Casos (CBR)**, uma técnica de *Lazy Learning*, para realizar o diagnóstico preliminar de doenças. O sistema recebe um conjunto de sintomas e, ao consultar uma base de conhecimento de casos passados, recupera o diagnóstico mais provável.

O desenvolvimento deste projeto foi uma jornada em duas fases que destaca um princípio fundamental da ciência de dados: **a qualidade e a escala dos dados são os fatores mais críticos para o sucesso de um modelo de aprendizado.** Partimos de um protótipo com baixa performance (2.38% de acurácia) para uma solução final otimizada que alcançou **100% de acurácia** no dataset de teste.

A biblioteca Python `cbrkit` foi utilizada como base estrutural, mas a lógica de similaridade, o núcleo do sistema, foi implementada de forma customizada para garantir robustez e permitir otimizações avançadas como a ponderação de sintomas com TF-IDF.

## Estrutura do Repositório

```
.
├── data/
│   └── Training.csv     
|   └── kaggle-base.csv 
└── src/
    ├── cbr_sintomas_doenca.py  # Script de demonstração do conceito CBR.
    ├── teste_loo.py            # Script de teste inicial (usado com a base pequena).
    ├── teste_loo_melhorado.py  # Script de avaliação final com TF-IDF.
    └── diagnostico.py          # Script utilitário para depuração da biblioteca.
```

## O Dataset: "Disease Prediction using Machine Learning"

O sucesso deste projeto está diretamente ligado ao dataset utilizado.

*   **Fonte:** [Disease Prediction using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning ) por Kaushil H. D. no Kaggle.
*   **Estrutura:** O dataset contém **4.920 registros** (casos). Cada caso possui:
    *   **132 colunas de sintomas:** Com valores binários (1 para presente, 0 para ausente).
    *   **1 coluna `prognosis`:** Contendo o nome da doença diagnosticada (41 doenças únicas no total).

Essa estrutura de "problema (sintomas) -> solução (diagnóstico)" é o formato ideal para um sistema de Raciocínio Baseado em Casos.

## A Arquitetura da Solução

A solução final é um sistema CBR que implementa uma métrica de similaridade sofisticada para comparar casos.

1.  **Engenharia de Features com TF-IDF:** Antes de qualquer comparação, o sistema analisa todos os 4.920 casos para calcular o peso **IDF (Inverse Document Frequency)** de cada sintoma. Sintomas raros e específicos recebem um peso alto, enquanto sintomas comuns e ambíguos recebem um peso baixo. Isso cria uma camada de "inteligência" que informa ao sistema quais sintomas são mais importantes para o diagnóstico.

2.  **Similaridade de Jaccard Ponderada:** O núcleo do sistema é uma função de similaridade customizada. Em vez de usar a Jaccard clássica (que apenas conta sintomas em comum), nossa implementação calcula a **soma dos pesos IDF** dos sintomas na interseção e na união. Isso garante que a concordância em um sintoma raro e importante contribua muito mais para a similaridade do que a concordância em um sintoma comum.

3.  **Validação com Leave-One-Out (LOO):** O sistema foi rigorosamente testado com o método de validação cruzada LOO. Para cada um dos 4.920 casos, ele foi temporariamente removido da base e usado como uma consulta para testar se o sistema conseguia prever seu diagnóstico corretamente.

## Como Executar o Projeto

### Pré-requisitos

*   Python 3.x
*   Pip (gerenciador de pacotes do Python)

### 1. Instalação

Primeiro, clone o repositório e navegue até a pasta do projeto:
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

Crie e ative um ambiente virtual (altamente recomendado ):
```bash
python -m venv venv
# No Linux/macOS:
source venv/bin/activate
# No Windows:
venv\Scripts\activate
```

Instale as dependências necessárias:
```bash
pip install pandas tqdm scikit-learn
```

### 2. Executando a Avaliação Final

Para executar o teste completo do sistema e replicar o resultado de 100% de acurácia, rode o script principal de avaliação.

**Atenção:** Este processo é computacionalmente intensivo e pode levar alguns minutos para ser concluído.

```bash
python src/teste_loo_melhorado.py
```

### Resultado Esperado

Após a execução, você verá a seguinte saída no terminal, confirmando a eficácia do modelo:

```
🚀 Iniciando o teste Leave-One-Out MELHORADO com 4920 casos...
Testando Casos: 100%|██████████████████████| 4920/4920 [XX:XX<00:00, XX.XXit/s]

==================================================
📊 RESULTADO DO TESTE LEAVE-ONE-OUT (COM TF-IDF) 📊
==================================================
Total de Casos Testados: 4920
Total de Acertos: 4920
Nova Acurácia do Sistema: 100.00%
==================================================
```


# cbr-aprendizado-de-maquina
# aprendizado-de-maquina-cbr
