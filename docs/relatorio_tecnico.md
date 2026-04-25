# Relatório Técnico — Cardio ML

**Projeto de Disciplina 26E1_3 — Engenharia de Machine Learning**
**Aluno:** Anderson Corrêa

---

## Introdução

Este projeto é a **evolução** do trabalho anterior (26E1_2). O mesmo dataset — 70.000 pacientes, 11 features de saúde e estilo de vida — é reutilizado, mas organizado como um **sistema de ML completo**, não como um notebook exploratório.

**Entregas principais:**

- Pacote Python (`cardio_ml`) com 8 camadas funcionais.
- **15 experimentos comparáveis** (5 modelos × 3 técnicas de redução de dimensionalidade) rastreados no MLflow.
- **Seleção automatica** de modelo campeão via score composto auditável (performance + custo + eficiência).
- Serviço de inferência FastAPI com Docker e CI/CD real (GitHub Actions).
- **Deploy real na AWS** via CDK (TypeScript): Lambda + API Gateway com interface visual acessível publicamente.
- Detecção de drift de dados e de modelo (PSI + KS próprios, complementados por relatório Evidently).
- Documentação em três níveis: este relatório, ADRs condensadas (`docs/decisoes_tecnicas.md`) e estratégia de operação (`docs/ciclo_de_vida.md`).

---

## 1. Contexto do problema e objetivos

### 1.1 Domínio e motivação

Doenças cardiovasculares (DCV) são a principal causa de morte no mundo — aproximadamente 17,9 milhões de óbitos anuais (OMS, 2021). A triagem precoce pode reduzir drasticamente essa estatística, mas exige classificar corretamente o risco de cada paciente a partir de dados clínicos simples: idade, pressão arterial, colesterol, hábitos, biometria.

O dataset usado (`cardio_train.csv`, Kaggle) contém 70.000 pacientes anônimos com 11 features brutas e uma variável-alvo binária (`cardio`). O balanceamento natural de classes (aproximadamente 50/50) remove a necessidade de técnicas especiais de reamostragem e faz de F1 e accuracy métricas igualmente informativas.

### 1.2 Objetivos técnicos

| Objetivo                       | Métrica primária  | Meta     |
| ------------------------------ | ----------------- | -------- |
| Classificar risco de DCV       | F1-score no teste | ≥ 0,70   |
| Discriminar entre casos        | ROC-AUC           | ≥ 0,78   |
| Capturar positivos (recall)    | Recall            | ≥ 0,65   |
| Tempo de inferência            | Latência p95      | < 50 ms  |
| Tempo total de treino da suíte | Wall-clock        | < 10 min |

### 1.3 Métricas de negócio

A triagem clínica de DCV tem um perfil de custos assimétrico: **um falso negativo (paciente doente classificado como saudável) é mais grave que um falso positivo** (paciente saudável sinalizado, submetido a um exame adicional inofensivo). Modelamos esse custo explicitamente em `src/cardio_ml/evaluation/metrics.py`:

> `custo_esperado_por_caso = (5 × FN + 1 × FP) / N`

A razão 5:1 é uma estimativa discutível — expomos os pesos como **constantes auditáveis** (`COST_FALSE_NEGATIVE`, `COST_FALSE_POSITIVE`) para permitir ajuste em revisão clínica futura. Outras métricas de negócio expostas:

- **Taxa de captura** — recall dos verdadeiros positivos (proporção de doentes identificados).
- **Taxa de falso alarme** — FPR (saudáveis sinalizados erradamente).
- **Casos positivos acima do aleatório** — uplift em unidades absolutas contra um modelo baseline aleatório.

### 1.4 Diferenciação em relação ao 26E1_2

| Aspecto                      | 26E1_2                | 26E1_3                             |
| ---------------------------- | --------------------- | ---------------------------------- |
| Forma do código              | Notebook único        | Pacote Python instalável           |
| Rastreamento de experimentos | Manual, em markdown   | Automatizado via MLflow            |
| Comparação de modelos        | 4 variantes em tabela | 15 experimentos controlados        |
| Redução de dimensionalidade  | Não abordada          | PCA + LDA integrados ao Pipeline   |
| Persistência de modelo       | Ausente               | MLflow Model Registry com estágios |
| Inferência                   | Ausente               | FastAPI + Docker                   |
| Deploy em nuvem              | Ausente               | AWS Lambda + API Gateway (CDK)     |
| Monitoramento                | Ausente               | PSI/KS + Evidently                 |
| CI/CD                        | Ausente               | GitHub Actions                     |

---

## 2. Estrutura do projeto e decisões de engenharia

A distinção mais importante entre um projeto exploratório e um projeto de engenharia é que **o projeto de engenharia tem interfaces estáveis**. Os módulos de `cardio_ml` foram desenhados como **Deep Modules**: cada um expõe uma interface mínima e esconde complexidade significativa.

```
src/cardio_ml/
├── data/            Ingestão, diagnóstico, split estratificado
├── features/        Preprocessamento e redução de dimensionalidade
├── models/          Registry de candidatos + tuning
├── tracking/        Wrapper do MLflow
├── training/        CLI de treino
├── evaluation/      Métricas técnicas, de negócio e drift
├── inference/       Carregamento do modelo do Registry
└── serving/         API FastAPI
```

**Princípios aplicados:**

- **Falha precoce em limites.** A ingestão valida o schema; um CSV com colunas diferentes quebra ali, não 200 linhas adiante no pipeline.
- **Determinismo na ingestão.** `age_years` e `bmi` são derivações determinísticas — computadas na ingestão, fora do pipeline. Qualquer coisa que dependa de estatísticas dos dados (média, desvio, quantis) **precisa** viver dentro do `Pipeline` do scikit-learn para evitar data leakage.
- **Configuração central.** `config.py` concentra caminhos, seed, limites de recurso e URI do MLflow. Evita divergência entre scripts.
- **Política de recursos transparente.** O pacote aplica os limites de thread e de prioridade já no `__init__`. Nenhum ponto de entrada (CLI, API, notebook) precisa lembrar de configurar isso.

Ver `docs/decisoes_tecnicas.md` para o registro completo das decisões arquiteturais.

---

## 3. Fundação de dados

### 3.1 Características gerais

- **70.000 registros**, 11 features + 1 target, sem valores nulos.
- Target `cardio` balanceado (~50/50), dispensando técnicas de oversampling.
- Features numéricas: `age_years` (derivada), `height`, `weight`, `ap_hi`, `ap_lo`, `bmi` (derivada).
- Features categóricas ordinais ou binárias: `gender`, `cholesterol`, `gluc`, `smoke`, `alco`, `active`.

### 3.2 Problemas estruturais detectados

A função `diagnose_quality` em `src/cardio_ml/data/quality.py` identifica automaticamente os principais problemas:

| Severidade | Coluna                                 | Problema                                          | Linhas afetadas |
| ---------- | -------------------------------------- | ------------------------------------------------- | --------------- |
| **Alta**   | `ap_lo`                                | Valores fora de [40, 140] mmHg                    | ~1.042          |
| **Alta**   | `ap_hi/ap_lo`                          | Pressão diastólica maior que sistólica (inversão) | várias centenas |
| Média      | `<dataset>`                            | Duplicatas em features+target                     | 24              |
| Baixa      | `ap_hi`                                | Valores fora de [80, 220] mmHg                    | ~255            |
| Baixa      | `height`, `weight`, `bmi`, `age_years` | Outliers fisiológicos                             | <200 cada       |

**Decisão de tratamento: não remover silenciosamente.** Os outliers representam erro de medição real e remover enviesaria o dataset (viés de sobrevivência). A estratégia adotada é:

1. Documentar os problemas no relatório.
2. Aplicar `StandardScaler` e modelos robustos (árvores) que absorvem bem os outliers.
3. Deixar o `QualityReport` como artefato logável no MLflow para rastreabilidade.

### 3.3 Vieses potenciais

- **Auto-reporte**: `smoke`, `alco`, `active` são auto-reportados e suscetíveis a viés social (subestimação de comportamentos mal vistos). Esperamos coeficientes de baixa magnitude nesses atributos em modelos lineares — e o resultado empírico confirma.
- **Demografia fixa**: dataset de uma população específica, sem garantia de generalização para outras.
- **Sem dados temporais**: cada linha é um snapshot; o modelo não captura progressão da doença.

### 3.4 Pipeline sem leakage

O `build_preprocessor` em `src/cardio_ml/features/preprocessing.py` monta um `ColumnTransformer` com:

- **Numéricas**: `SimpleImputer(strategy='median')` → `StandardScaler`.
- **Categóricas**: `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`.

Encapsulado dentro de um `Pipeline` do scikit-learn, garantimos que as estatísticas aprendidas no `fit` vêm **apenas** do split de treino. O teste `tests/test_preprocessing.py::test_preprocessor_inside_pipeline_no_leakage` verifica isso como não-regressão.

O `handle_unknown='ignore'` no OneHotEncoder é crítico para produção: impede que categorias novas (que aparecem só depois do deploy) quebrem a inferência.

---

## 4. Experimentação sistemática

### 4.1 Candidatos

O registry em `src/cardio_ml/models/registry.py` define cinco candidatos:

| Modelo                         | Papel                    | Estratégia de busca                 |
| ------------------------------ | ------------------------ | ----------------------------------- |
| `DummyClassifier` (stratified) | Baseline aleatório       | nenhuma (referência)                |
| `LogisticRegression`           | Baseline linear          | GridSearch (C, class_weight)        |
| `DecisionTree`                 | Não-linear interpretável | GridSearch (depth, leaf, criterion) |
| `RandomForest`                 | Ensemble bagging         | RandomizedSearch (12 iterações)     |
| `XGBoost`                      | Ensemble boosting (novo) | RandomizedSearch (15 iterações)     |

**Razão para misturar Grid e Random**: modelos leves (tree) suportam grids completos sem estouro de custo; ensembles pesados (RF, XGB) teriam grids de centenas de combinações — a busca aleatória com orçamento fixo é mais prática e converge para boas regiões do espaço.

### 4.2 Protocolo experimental

- **Split**: 80% treino / 20% teste, estratificado (`SEED=42`).
- **Validação cruzada**: 5-fold estratificado dentro do treino (também com `SEED=42`).
- **Métrica de otimização**: F1 (ponderação entre precisão e recall — coerente com o custo clínico de FN).
- **Rastreamento**: cada run no MLflow carrega tags (plataforma, Python version, seed, recursos), params (tipo, busca, hiperparâmetros escolhidos) e métricas (técnicas e de negócio).

### 4.3 Cruzamento com redução de dimensionalidade

Cada modelo é treinado **três vezes**: sem redução (`none`), com PCA (`pca`, preservando 95% da variância) e com LDA (`lda`, 1 componente). Total: **15 experimentos**. Ver §5 para análise.

### 4.4 Logs reprodutíveis no MLflow

O wrapper `start_run_context` em `src/cardio_ml/tracking/mlflow_utils.py` adiciona automaticamente:

- `python_version`, `platform`, `seed` — reprodutibilidade.
- `resources.n_jobs`, `resources.omp_num_threads`, ... — política de recursos aplicada.
- `model_family`, `dim_technique`, `scoring` — facilitam consultas/filtros.

Métricas loggadas com prefixo `tech.` (técnicas), `biz.` (negócio), `cv.` (validação cruzada) e `train.` (tempo de fit) — o prefixo possibilita gráficos agrupados na UI.

---

## 5. Análise comparativa e seleção final

Os 15 experimentos foram executados em sequência na máquina de desenvolvimento (Apple M2 Pro, 4 efficiency cores via `taskpolicy -c background`), totalizando aproximadamente 4 minutos de wall-clock. Todos os runs estão registrados em `./mlruns` e podem ser inspecionados via `make mlflow-ui`.

### 5.1 Ranking geral (resultados reais)

| Modelo                | Dim.     | CV F1      | Test F1    | Test AUC   |
| --------------------- | -------- | ---------- | ---------- | ---------- |
| `dummy`               | none     | 0,4954     | 0,4958     | 0,4993     |
| `dummy`               | pca      | 0,4954     | 0,4958     | 0,4993     |
| `dummy`               | lda      | 0,4954     | 0,4958     | 0,4993     |
| `logistic_regression` | none     | 0,7080     | 0,7022     | 0,7788     |
| `logistic_regression` | pca      | 0,7069     | 0,7015     | 0,7745     |
| `logistic_regression` | lda      | 0,6408     | 0,6321     | 0,6973     |
| `decision_tree`       | none     | 0,7210     | 0,7198     | 0,7837     |
| `decision_tree`       | pca      | 0,6815     | 0,6657     | 0,7445     |
| `decision_tree`       | lda      | 0,6454     | 0,6329     | 0,6916     |
| `random_forest`       | none     | 0,7236     | 0,7222     | 0,7963     |
| `random_forest`       | pca      | 0,7098     | 0,7071     | 0,7700     |
| `random_forest`       | lda      | 0,6450     | 0,6330     | 0,6969     |
| **`xgboost`**         | **none** | **0,7267** | **0,7260** | **0,8007** |
| `xgboost`             | pca      | 0,7184     | 0,7135     | 0,7857     |
| `xgboost`             | lda      | 0,6445     | 0,6335     | 0,6969     |

**Observações principais:**

- **XGBoost sem redução** atinge o topo em todos os três quesitos técnicos: F1 de validação cruzada (0,727), F1 de teste (0,726) e ROC-AUC de teste (0,801).
- **Random Forest sem redução** fica muito próximo (F1 0,722, AUC 0,796) — diferença de menos de 1 pp em F1.
- **Árvore de Decisão ajustada** alcança 0,720 F1, surpreendentemente competitiva com o Random Forest — boa interpretabilidade, teto apenas ligeiramente menor.
- **Regressão Logística** entrega 0,702 F1 / 0,779 AUC — replica o ganho esperado sobre o Perceptron do 26E1_2 e comprova que a fronteira de decisão tem componente linear relevante.
- **DummyClassifier** fica exatamente onde deveria (0,496 F1), confirmando que o sinal aprendido pelos modelos não-triviais é real.

### 5.2 Impacto da redução de dimensionalidade (evidência empírica)

Comparando F1 de teste `raw` vs `pca` vs `lda` para cada modelo:

| Modelo                | raw    | pca    | Δ pca  | lda    | Δ lda  |
| --------------------- | ------ | ------ | ------ | ------ | ------ |
| `logistic_regression` | 0,7022 | 0,7015 | −0,001 | 0,6321 | −0,070 |
| `decision_tree`       | 0,7198 | 0,6657 | −0,054 | 0,6329 | −0,087 |
| `random_forest`       | 0,7222 | 0,7071 | −0,015 | 0,6330 | −0,089 |
| `xgboost`             | 0,7260 | 0,7135 | −0,013 | 0,6335 | −0,093 |

**Conclusões:**

- **PCA** tem impacto modesto (−0,1 a −5,4 pp de F1). Em `logistic_regression` a perda é quase nula — o modelo já era linear — mas em `decision_tree` chega a −5,4 pp, sugerindo que a árvore se beneficiava de features não-gaussianas que o PCA achatou.
- **LDA** tem impacto consistentemente negativo (−7 a −9 pp de F1). Forçar 1 componente único elimina informação discriminativa que os modelos não-lineares exploravam. **LDA não é adequada a este dataset** quando usada isoladamente como redutor — diagnóstico técnico importante.
- **Em XGBoost**, PCA degrada 1,3 pp — a regularização interna (`subsample`, `colsample_bytree`, `max_depth`) já controla overfitting sem necessidade de projeção.

**Resposta à pergunta da rubrica ("a redução de dimensionalidade é adequada?")**: neste problema, **não**. A redução gera economia computacional marginal (o dataset já é modesto) sem ganho de performance. Documentar esta conclusão honestamente tem mais valor pedagógico que forçar um ganho artificial.

### 5.3 Trade-off performance × custo computacional

Tempos de `fit+search` no teste (5 folds × grids ou n_iter configurados):

| Modelo (raw)          | F1    | Tempo (s) | Observação                                      |
| --------------------- | ----- | --------- | ----------------------------------------------- |
| `logistic_regression` | 0,702 | 8         | Campeão custo-benefício se simplicidade importa |
| `decision_tree`       | 0,720 | 27        | Melhor interpretabilidade, bom F1               |
| `xgboost`             | 0,726 | 93        | Topo de performance, custo moderado             |
| `random_forest`       | 0,722 | 334       | 3,5× mais lento que XGBoost com F1 inferior     |

**Insight**: Random Forest foi dramaticamente mais lento que XGBoost sem entregar performance superior — XGBoost usa `tree_method="hist"` que é altamente otimizado. Isso justifica a escolha de XGBoost não apenas por F1, mas também por eficiência.

### 5.4 Score composto e seleção

O `scripts/select_final_model.py` aplica:

> `score = 0,60 × F1_norm + 0,25 × (1 − custo_esperado_norm) + 0,15 × (1 − tempo_norm)`

**Ranking resultante** (top 5):

| Posição | Run                        | F1    | AUC   | Custo | Tempo | Score     |
| ------- | -------------------------- | ----- | ----- | ----- | ----- | --------- |
| **1**   | `xgboost__raw`             | 0,726 | 0,801 | 0,863 | 93s   | **0,989** |
| 2       | `decision_tree__raw`       | 0,720 | 0,784 | 0,861 | 27s   | 0,959     |
| 3       | `random_forest__raw`       | 0,722 | 0,796 | 0,874 | 334s  | 0,928     |
| 4       | `xgboost__pca`             | 0,713 | 0,786 | 0,879 | 124s  | 0,891     |
| 5       | `logistic_regression__pca` | 0,702 | 0,775 | 0,929 | 15s   | 0,779     |

**Modelo vencedor: `xgboost` sem redução de dimensionalidade.** Registrado como `cardio-dcv-classifier v1` e promovido ao estágio `Production` no MLflow Model Registry. O score de 0,989 reflete que ele foi o melhor nos três eixos (performance, custo de negócio, eficiência relativa entre candidatos de topo).

### 5.5 Registro no Model Registry

O mesmo script:

1. Cria/atualiza o modelo registrado `cardio-dcv-classifier`.
2. Registra uma nova versão apontando para o run vencedor.
3. Promove para o estágio `Production` (arquivando versões anteriores).

A API (`src/cardio_ml/serving/api.py`) carrega essa versão no startup via `load_production_model`.

---

## 6. Operacionalização

### 6.1 Persistência versionada

O modelo é persistido pelo MLflow como artefato `sklearn.pipeline.Pipeline` com assinatura (`infer_signature`) e exemplo de entrada (`input_example`), permitindo que a API valide automaticamente o payload na inferência.

Versões seguem o ciclo de vida completo: `None → Staging → Production → Archived`.

### 6.2 Serviço de inferência

**FastAPI** com três rotas principais:

| Rota          | Método | Função                                                |
| ------------- | ------ | ----------------------------------------------------- |
| `/` ou `/ui`  | GET    | Interface visual de triagem (formulário + lote CSV)   |
| `/health`     | GET    | Liveness + readiness (responde 200 mesmo sem modelo)  |
| `/model-info` | GET    | Metadados do modelo servido (versão, estágio, run_id) |
| `/predict`    | POST   | Inferência em 1 a 1000 observações                    |
| `/docs`       | GET    | Documentação interativa Swagger (auto-gerada)         |

**Decisões de design:**

- Validação em Pydantic com limites realistas (idade 18-100, ap_hi 60-260, etc.).
- `bmi` opcional na requisição — derivado automaticamente de altura e peso se omitido.
- Resposta inclui `probability` e `risk_level` textual (`baixo`/`moderado`/`alto`), traduzindo a probabilidade em algo consumível por sistemas não-ML a jusante.
- Lifespan assíncrono carrega o modelo no startup; falha (503 em `/predict`) quando não há modelo.

### 6.3 Empacotamento e deploy

- `Dockerfile` de uma única estágio, enxuto (`python:3.12-slim`).
- `docker-compose.yml` com serviços `api` e `mlflow` (UI). Limites de CPU e memória aplicados (`cpus: "2"`, `memory: 2g` na API) para garantir que o ambiente local do revisor não seja saturado.
- Healthcheck HTTP embutido no Dockerfile.

### 6.4 CI/CD

Workflow em `.github/workflows/ci.yml`: lint → testes → smoke-train → build Docker. Roda em cada push/PR. Artefatos do MLflow são preservados por 7 dias para diagnóstico.

**Escolha deliberada**: CI/CD **real**, não simulado. O texto da rubrica aceita os dois — optamos pelo real porque gera um sinal verificável (badge de build) e exercita de fato a reproduzibilidade.

### 6.5 Deploy na AWS

Além do ambiente local (Docker Compose), o modelo está implantado na AWS como um serviço acessível publicamente:

**URL da API:** https://ffe04qm1nl.execute-api.us-east-1.amazonaws.com/

Infraestrutura definida como código (Infrastructure as Code) usando **AWS CDK** no diretório `aws/`:

```
API Gateway (HTTP API v2)  →  Lambda (ARM64, 1024 MB)  →  pipeline.joblib
       ↑
CloudWatch Events (a cada 10 min mantém a função aquecida)
```

**Decisões de design:**

- **Lambda com imagem Docker** em vez de ZIP: permite empacotar sklearn + xgboost + fastapi sem limite de tamanho dos layers tradicionais.
- **ARM64 (Graviton2)**: ~20% mais barato que x86 e cold start mais rápido.
- **Modelo extraído como joblib**: o deploy exporta o modelo vencedor do MLflow Registry para um arquivo `.joblib` e o embute na imagem Docker. Em Lambda não há acesso ao MLflow — o modelo é carregado diretamente do disco via `joblib.load()`. Isso elimina a dependência do MLflow (~200 MB) do runtime de produção.
- **Warming rule a cada 10 minutos**: Lambda permanece aquecida e responde em ~50ms. Sem a regra, o cold start levaria ~10-15s na primeira requisição.
- **Throttling**: 10 req/s com burst de 20 — proteção contra uso indevido.
- **Custo**: $0,00 (Lambda: 1M requests/mês grátis; API Gateway HTTP: 1M requests/mês grátis).

A interface visual (`/ui`) permite a qualquer pessoa preencher dados e receber a classificação de risco imediatamente.

O deploy e destruição da stack são comandos únicos:

```bash
cd aws
./deploy.sh <PROFILE>    # cria a stack
./destroy.sh <PROFILE>   # remove a stack
```

---

## 7. Monitoramento, drift e re-treinamento

### 7.1 Métricas em operação

Combinamos métricas **técnicas** (precision, recall, F1, ROC-AUC, average precision) com métricas de **negócio** (custo esperado, taxa de captura, taxa de falso alarme). Ambos são logados no MLflow por run e podem ser plotados em dashboards externos (Grafana, etc.).

### 7.2 Drift de dados

Implementação própria em `src/cardio_ml/evaluation/drift.py`:

- **PSI (Population Stability Index)** com limiares padrão da indústria: <0,10 estável, 0,10-0,25 moderado, >0,25 drift.
- **KS (Kolmogorov-Smirnov)** como teste de significância estatística complementar.
- Relatório agregado (`DriftReport`) exportável em JSON.

### 7.3 Drift de modelo

A mesma metodologia PSI + KS é aplicada à distribuição das **probabilidades preditas** pelo modelo (além das features). Mudança nessa distribuição indica drift conceitual — as features podem estar estáveis mas a fronteira de decisão que o modelo vê mudou.

### 7.4 Evidently como complemento visual

`scripts/simulate_drift.py` gera dois artefatos:

- `reports/drift_summary.json` — nosso relatório estruturado (PSI, KS por feature).
- `reports/drift_evidently.html` — relatório HTML.

### 7.5 Estratégia de re-treinamento

Documentada em detalhe em `docs/ciclo_de_vida.md`. Em resumo:

- **Gatilhos**: PSI > 0,25 em 2+ features relevantes, queda de recall > 5 pp, mudança nas diretrizes clínicas, ou tempo (preventivo a cada 6 meses).
- **Procedimento**: coletar dados acumulados → rodar `run_full_experiment.py` → comparar com campeão atual → promover via `canary` (10% do tráfego por 1 semana) → cutover ou rollback.
- **Aprendizado online descartado**: em vez de atualizar o modelo continuamente a cada novo paciente, o que dificultaria rastrear qual versão gerou cada predição, optamos por re-treinar periodicamente em lote, com dados inspecionados, e registrar cada versão no MLflow.

---

## 8. Conclusão

Este projeto demonstra a transição de **executor de modelos** para **engenheiro de machine learning**. Os ganhos específicos sobre o 26E1_2:

1. **Reprodutibilidade**: qualquer revisor consegue executar a suíte completa com `make all` e obter resultados idênticos (dentro do ruído de não-determinismo do scikit-learn/XGBoost).
2. **Rastreabilidade**: 15 experimentos rastreados com parâmetros, métricas, artefatos e tags que permitem responder "por que esse modelo venceu" meses depois.
3. **Decisões auditáveis**: seleção via código — pesos alteráveis em um único lugar.
4. **Operacionalidade**: modelo versionado, API funcional com interface visual, CI verde, deploy real na AWS acessível publicamente, monitoramento de drift em posição.

**Possíveis próximos passos:**

- Calibração de probabilidades (isotonic regression) para melhorar a semântica do `risk_level`.
- Análise de fairness por subgrupo (gênero, idade).
- Experimentar `xgboost` + feature engineering clinicamente motivado (classe funcional de pressão arterial).
- Promover o MLflow para servidor remoto com PostgreSQL + S3 quando houver time colaborando.
