# Cardio ML — Engenharia de Machine Learning para Triagem de DCV

**Aluno:** Anderson Corrêa

**Disciplina:** Operacionalização de Modelos com MLOps [26E1_3]

**Apresentação do projeto (vídeo):** https://www.youtube.com/watch?v=PjY1hlMIBKU

Este repositório é a **evolução** do [projeto anterior (26E1_2)](https://github.com/andersonfpcorrea/26E1_2). O foco deixa de ser o _algoritmo isolado_ e passa a ser o _projeto de ML como um sistema_: pipelines reutilizáveis, rastreamento de experimentos, modelo versionado, serviço de inferência, monitoramento e estratégia de re-treinamento.

**Acesse a aplicação implantada:** https://ffe04qm1nl.execute-api.us-east-1.amazonaws.com/

---

## Arquitetura

### Fluxo do pipeline

```
                    ┌───────────────────────────────────┐
                    │         make experiment           │
                    └───────────────┬───────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │   Preprocessamento                │
                    │   StandardScaler + OneHotEncoder  │
                    └───────────────┬───────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
        ┌──────────┐         ┌──────────┐         ┌──────────┐
        │   raw    │         │   PCA    │         │   LDA    │
        │ (sem     │         │ (95%     │         │ (1       │
        │ redução) │         │ variânc.)│         │ compon.) │
        └────┬─────┘         └────┬─────┘         └────┬─────┘
             │                    │                    │
             ▼                    ▼                    ▼
        5 modelos ×1         5 modelos ×1         5 modelos ×1
        Dummy, LogReg,       Dummy, LogReg,       Dummy, LogReg,
        Tree, RF, XGB        Tree, RF, XGB        Tree, RF, XGB
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────────────┐
                    │       MLflow (15 runs)             │
                    │   params + métricas + modelos      │
                    └───────────────┬───────────────────┘
                                    │
                                    ▼  make select
                    ┌───────────────────────────────────┐
                    │   Score composto                   │
                    │   60% F1 + 25% custo + 15% tempo  │
                    └───────────────┬───────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │   Model Registry                  │
                    │   cardio-dcv v1 → Production      │
                    └───────────────┬───────────────────┘
                                    │
                      ┌─────────────┼─────────────┐
                      ▼                           ▼
               ┌─────────────┐          ┌──────────────┐
               │ make serve  │          │ AWS Lambda   │
               │ localhost   │          │ (deploy.sh)  │
               │ :8000/ui    │          │ URL pública  │
               └─────────────┘          └──────────────┘
```

### Estrutura do código

```
src/cardio_ml/
├── data/            # Ingestão, diagnóstico de qualidade, split estratificado
├── features/        # ColumnTransformer + wrappers PCA/LDA
├── models/          # Registry de 5 candidatos + tuning (Grid/Random Search)
├── tracking/        # Wrapper do MLflow (tags e artefatos padronizados)
├── training/        # CLI cardio-train
├── evaluation/      # Métricas técnicas + de negócio + PSI/KS para drift
├── inference/       # Carregamento do modelo do Registry
└── serving/         # FastAPI (/predict, /health, /model-info, /ui)
```

---

## Quickstart

### 1. Pré-requisitos

- Python 3
- [`uv`](https://docs.astral.sh/uv/) (gerenciador de dependências)
- [`Docker`](https://docs.docker.com/get-started/get-docker/) (opcional, para rodar via container)
- [`make`](https://www.gnu.org/software/make/) (já incluso no macOS/Linux; no Windows use [chocolatey](https://community.chocolatey.org/packages/make) ou WSL)

### 2. Instalar dependências

```bash
make install    # cria .venv/ local e instala dependências
```

### 3. Rodar a suíte completa de experimentos

```bash
make experiment
```

Isso treina **5 modelos × 3 técnicas de redução = 15 experimentos**. Cada experimento registra automaticamente seus parâmetros, métricas e o modelo treinado.

Para visualizar e comparar os resultados:

```bash
make mlflow-ui
# Abre em http://localhost:5001
```

Na interface do MLflow é possível ordenar por métrica (ex: F1), filtrar por modelo, comparar runs lado a lado e inspecionar os hiperparâmetros escolhidos para cada experimento.

### 4. Selecionar o modelo vencedor

```bash
make select
```

Aplica o score composto (60% F1 + 25% custo esperado + 15% tempo de treino), registra o vencedor no MLflow Model Registry e promove para o estágio `Production`.

### 5. API de inferência

**Opção A — localmente:**

```bash
make serve
# Interface visual: http://localhost:8000/ui
# Swagger (API):    http://localhost:8000/docs
```

**Opção B — via Docker:**

```bash
docker compose up --build
# API + UI:  http://localhost:8000/ui
# MLflow:    http://localhost:5001
```

### 6. Exemplo de requisição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [{
      "age_years": 55.0,
      "height": 170.0,
      "weight": 78.0,
      "ap_hi": 140.0,
      "ap_lo": 90.0,
      "gender": 1,
      "cholesterol": 2,
      "gluc": 1,
      "smoke": 0,
      "alco": 0,
      "active": 1
    }]
  }'
```

Resposta:

```json
{
  "model_version": "1",
  "predictions": [
    {
      "predicted_class": 1,
      "probability": 0.78,
      "risk_level": "alto"
    }
  ]
}
```

### 7. Simular drift

```bash
make drift
# Gera: reports/drift_summary.json e reports/drift_evidently.html
```

---

## Rastreabilidade com a rubrica

| Requisito                                 | Onde está atendido                                                                                   |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Parte 1 — estruturação do projeto         | `src/cardio_ml/`, `pyproject.toml`, `Makefile`                                                       |
| Parte 2 — fundação e diagnóstico de dados | `src/cardio_ml/data/quality.py`, `notebooks/01_diagnostico_dados.ipynb`                              |
| Parte 3 — experimentação sistemática      | `scripts/run_full_experiment.py`, `mlruns/`                                                          |
| Parte 4 — redução de dimensionalidade     | `src/cardio_ml/features/dimensionality.py` (PCA + LDA)                                               |
| Parte 5 — seleção final                   | `scripts/select_final_model.py`, `notebooks/02_analise_experimentos.ipynb`                           |
| Parte 6 — operação                        | `src/cardio_ml/serving/`, `scripts/simulate_drift.py`, `docs/ciclo_de_vida.md`, `aws/` (deploy real) |

---

## Uso de recursos

O projeto foi construído para rodar **sem saturar a máquina** do usuário/revisor. Três camadas combinadas:

1. **Variáveis de ambiente** (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc.) limitam threads BLAS — funciona em Linux, macOS e Windows.
2. **`N_JOBS=4`** limita a paralelização do scikit-learn e XGBoost.
3. **`psutil`** rebaixa a prioridade do processo (cross-platform).

`make train` e `make experiment` usam wrappers específicos do sistema operacional:

- **macOS**: `taskpolicy -c background` (fixa o processo em 4 efficiency cores).
- **Linux**: `nice -n 19 ionice -c 3`.
- **Windows**: sem wrapper — use Docker para aplicar limites.

Limites configuráveis via variáveis de ambiente (`CARDIO_N_JOBS`, `CARDIO_NICE`, `CARDIO_SEED`).

---

## Comandos úteis

| Comando           | Descrição                                     |
| ----------------- | --------------------------------------------- |
| `make install`    | Instala o pacote + dependências de dev        |
| `make lint`       | Verifica estilo com ruff                      |
| `make test`       | Roda a suíte de testes                        |
| `make experiment` | Executa os 15 experimentos completos          |
| `make select`     | Registra o modelo vencedor no MLflow Registry |
| `make serve`      | Sobe a API FastAPI em `localhost:8000`        |
| `make mlflow-ui`  | Abre a UI do MLflow em `localhost:5001`       |
| `make drift`      | Simula e relata drift com PSI/KS + Evidently  |
| `make clean`      | Remove caches e artefatos de build            |
| `make all`        | de `install` até `select`                     |

---

## Documentação complementar

- [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md) — **documento central** avaliado pela rubrica.
- [`docs/decisoes_tecnicas.md`](docs/decisoes_tecnicas.md) — ADRs condensadas.
- [`docs/ciclo_de_vida.md`](docs/ciclo_de_vida.md) — estratégia de monitoramento, drift e re-treino.

---

## CI/CD de exemplo

O workflow em `.github/workflows/ci.yml` executa em cada push/PR:

1. Lint (ruff)
2. Testes unitários (pytest)
3. Smoke-train (modelos rápidos, em amostra) para garantir que o pipeline inteiro ainda funciona
4. Build da imagem Docker

Artefatos do MLflow são preservados por 7 dias para diagnóstico.

---

## Deploy na AWS

O diretório `aws/` é um projeto CDK (TypeScript) que implanta a API usando **Lambda + API Gateway**.

```
                    ┌──────────────────────────────────────┐
                    │            API Gateway               │
                    │          (HTTP API v2)               │
                    │    throttle: 10 req/s, burst 20      │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │         Lambda (ARM64)               │
                    │    1024 MB / timeout 30s             │
                    │                                      │
                    │  ┌────────────────────────────────┐  │
                    │  │  handler.py (mangum)           │  │
                    │  │       ↓                        │  │
                    │  │  FastAPI app                   │  │
                    │  │       ↓                        │  │
                    │  │  pipeline.joblib → .predict()  │  │
                    │  └────────────────────────────────┘  │
                    └──────────────────────────────────────┘
                                       ▲
                                       │ ping /health
                    ┌──────────────────────────────────────┐
                    │     CloudWatch Events                │
                    │     rate(10 minutes)                 │
                    │     mantém a função aquecida         │
                    └──────────────────────────────────────┘
```

### Pré-requisitos

- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- [AWS CDK CLI](https://docs.aws.amazon.com/cdk/v2/guide/getting-started.html#getting-started-install)
- [Docker](https://docs.docker.com/get-started/get-docker/)

### Deploy

```bash
# Autenticar na AWS
aws login --profile <YOUR_PROFILE>

# Deploy
cd aws
npm i                          # instala dependências CDK
./deploy.sh <YOUR_PROFILE>     # exporta modelo + cdk deploy
```

O script de deploy faz o seguinte:

1. Verifica credenciais AWS do `profile` informado
2. Exporta o modelo campeão do MLflow para `aws/src/lambda/model/pipeline.joblib`
3. Build da imagem Docker
4. Implanta a stack na AWS (Lambda + API Gateway)
5. Renderiza no terminal a URL pública da API

### Testar a API implantada na AWS

Exemplos:

```bash
# Health check
curl https://<api-id>.execute-api.us-east-1.amazonaws.com/health

# Interface visual
open https://<api-id>.execute-api.us-east-1.amazonaws.com/ui

# Swagger
open https://<api-id>.execute-api.us-east-1.amazonaws.com/docs

# Predição
curl -X POST https://<api-id>.execute-api.us-east-1.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{"patients":[{"age_years":55,"height":170,"weight":78,"ap_hi":140,"ap_lo":90,"gender":1,"cholesterol":2,"gluc":1,"smoke":0,"alco":0,"active":1}]}'
```

### Testes unitários do handler

```bash
cd aws
npm run lambda:setup  # cria .venv local com uv + instala dependências Python
npm t                 # roda os testes unitários do handler
```

### Destruir a stack

```bash
cd aws
./destroy.sh <YOUR_PROFILE>   # destrói a stack
```
