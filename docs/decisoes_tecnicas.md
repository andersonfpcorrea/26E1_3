# Decisões Técnicas — Registros Arquiteturais (ADRs Condensadas)

Este documento registra as decisões arquiteturais mais impactantes do projeto, em formato ADR (Architecture Decision Record) condensado: **contexto, decisão, alternativas consideradas e consequências**.

---

## ADR-01 — Pacote Python instalável em vez de scripts soltos

**Contexto.** No projeto anterior (26E1_2) toda a lógica vivia em um único notebook Jupyter. Isso dificulta testes, reuso entre etapas (treino, inferência, drift) e versionamento dos artefatos.

**Decisão.** Reestruturar como um pacote Python (`cardio_ml`) instalável via `pip install -e .`, com `pyproject.toml` e separação clara entre camadas (`data`, `features`, `models`, `tracking`, `training`, `evaluation`, `inference`, `serving`).

**Alternativas consideradas.**

- Manter scripts soltos em `scripts/` sem empacotamento: rejeitada por não permitir imports limpos (`from cardio_ml.inference import ...`) nem testes unitários confiáveis.
- Monorepo com múltiplos pacotes: exagero para o escopo da disciplina.

**Consequências.** +Clareza arquitetural; +testabilidade; +reusabilidade entre treino e inferência; −curva de aprendizado ligeiramente maior para quem espera ver tudo em um único notebook.

---

## ADR-02 — MLflow local com store em arquivo

**Contexto.** A rubrica exige rastreamento de parâmetros, métricas e versões de modelos. Um MLflow remoto exigiria infraestrutura (banco de dados, servidor) que não agrega valor pedagógico.

**Decisão.** Usar MLflow com backend de arquivo em `./mlruns`. O Model Registry também fica local, acessível via UI (`make mlflow-ui`) ou API Python.

**Alternativas consideradas.**

- MLflow com backend PostgreSQL + artefatos em S3: arquitetura de produção, mas superdimensionada.
- Weights & Biases / Neptune: ferramentas equivalentes, mas requerem conta remota e fogem da stack pedida pela disciplina.

**Consequências.** Reproduzibilidade completa por qualquer revisor (`git clone` → `make all`); porém, o store local não escala para times grandes — decisão explicitamente aceita para o escopo deste projeto.

---

## ADR-03 — Escolha das duas técnicas de redução de dimensionalidade: PCA e LDA

**Contexto.** A Parte 4 exige duas técnicas entre PCA, LDA e t-SNE, com justificativa explícita.

**Decisão.** Usar PCA e LDA. t-SNE foi descartado.

**Justificativa individual.**

- **PCA** é não-supervisionada, remove colinearidade (relevante porque `bmi` é derivado de `height` e `weight`) e deixa passar 95% da variância com menos componentes que o total — controla overfitting e eficiência computacional.
- **LDA** é supervisionada, maximiza a separação entre classes; num problema binário balanceado ela projeta para um único componente ótimo para separação.

**Por que t-SNE foi descartado.** t-SNE é um método de visualização, não de projeção reusável. Sua API `fit_transform` não produz uma projeção estável de novos pontos, o que o torna inadequado como etapa de um pipeline de produção. Seria aceitável apenas em um notebook de exploração; como artefato produtizável quebra.

**Consequências.** +Ambas as técnicas se integram ao `Pipeline` do scikit-learn com `.transform` consistente; +LDA reforça a narrativa pedagógica ("supervisionada vs. não supervisionada"); −perdemos a visualização bidimensional típica do t-SNE (pode ser adicionada apenas no notebook de exploração se desejado).

---

## ADR-04 — XGBoost adicionado como modelo candidato

**Contexto.** O projeto anterior cobria Perceptron, Árvore de Decisão e Random Forest. Esperam-se avanços reais em 26E1_3.

**Decisão.** Adicionar XGBoost (gradient boosting) ao registry, junto com LogisticRegression (baseline linear mais bem-calibrado que Perceptron).

**Alternativas consideradas.**

- LightGBM: performance equivalente ao XGBoost em dados tabulares; a escolha entre os dois é praticamente intercambiável — optamos por XGBoost por ser o mais adotado no ecossistema scikit-learn.
- Stacking (ensemble de modelos heterogêneos): todos os nossos melhores modelos convergem para ~73–75% F1, indicando um teto do dataset. Stacking ajuda quando modelos cometem erros diferentes — quando todos platôam no mesmo nível, a complexidade adicional não compensa.

**Consequências.** +Possibilidade real de performance superior; +contato com o algoritmo campeão em competições de dados tabulares; −dependência extra, mas justificável.

---

## ADR-05 — Métricas de negócio explicitamente ponderadas (custo 5×1)

**Contexto.** Em triagem clínica de doenças cardiovasculares, um falso negativo (paciente com doença classificado como saudável) é tipicamente mais custoso que um falso positivo (paciente saudável sinalizado, que passa por um exame adicional).

**Decisão.** Expor pesos como constantes auditáveis em `evaluation/metrics.py`: `COST_FALSE_NEGATIVE = 5.0` e `COST_FALSE_POSITIVE = 1.0`. O custo esperado por caso entra na função de seleção do modelo campeão (ADR-06).

**Alternativas consideradas.**

- Não modelar custos: simplista demais para um projeto que se propõe a ligar ML a negócio.
- Aprender pesos a partir de dados de outcome real: fora do escopo (dataset não contém essa informação).

**Consequências.** +Auditável; +reforça a dimensão de negócio na rubrica; −os pesos são estimativas e podem ser ajustados em revisão clínica real.

---

## ADR-06 — Score composto para seleção do modelo final

**Contexto.** Selecionar o "melhor modelo" por uma métrica única (ex: F1) ignora custo computacional e alinhamento com o negócio.

**Decisão.** Score composto:

> `score = 0.60 * F1_norm + 0.25 * (1 − custo_esperado_norm) + 0.15 * (1 − tempo_treino_norm)`

Todas as métricas são min-max normalizadas; custo e tempo entram invertidos (menor é melhor). Em caso de empate, desempata por ROC-AUC.

**Alternativas consideradas.**

- Seleção manual: pouco auditável, não reprodutível.
- Otimização multi-objetivo (Pareto): identifica modelos na fronteira ótima, mas não escolhe entre eles — adia a decisão em vez de tomá-la. Com pesos explícitos (60/25/15) a seleção é automática, reprodutível e auditável.

**Consequências.** Decisão reprodutível; pesos podem ser alterados no código para explorar cenários.

---

## ADR-07 — FastAPI em vez de Flask/Streamlit

**Contexto.** A rubrica pede um serviço de inferência.

**Decisão.** FastAPI com Pydantic.

**Alternativas consideradas.**

- **Flask**: em Flask, cada campo do request precisa ser validado manualmente (`if "age_years" not in request.json: abort(400)`). Com FastAPI + Pydantic, a validação é declarativa — definimos os tipos e limites uma vez (`age_years: float = Field(ge=18, le=100)`) e payloads inválidos são rejeitados automaticamente com mensagens de erro detalhadas. Flask também não gera documentação Swagger automaticamente — exigiria uma extensão adicional (flask-restx, flasgger).
- **Streamlit**: gera interfaces visuais rapidamente, mas o resultado é uma aplicação web monolítica que só humanos podem usar. Não expõe endpoints HTTP que outros sistemas possam consumir. Um sistema hospitalar não consegue chamar um app Streamlit — precisa de uma API com `POST /predict` que aceite JSON e retorne JSON.

**Consequências.** +Documentação Swagger gerada automaticamente a partir dos tipos Pydantic; +validação de entrada em uma linha por campo; +compatível com Mangum para deploy em Lambda sem alteração de código; −dependência a mais em relação a Flask, mas justificável pelo ganho em produtividade e robustez.

---

## ADR-08 — Drift detectado com PSI+KS próprios E relatório Evidently

**Contexto.** Rubrica pede "detecção de drift de dados e de modelo por comparação estatística".

**Decisão.** Implementar PSI e KS em `evaluation/drift.py` (algoritmo próprio, curto, testado) e gerar relatório HTML com Evidently.

**Alternativas consideradas.**

- Apenas Evidently: resolveria a rubrica mas não demonstra compreensão do cálculo subjacente.
- Apenas código próprio: perderia a qualidade visual do relatório da ferramenta madura.

**Consequências.** Cobre a rubrica nos dois eixos — domínio técnico e maturidade operacional.

---

## ADR-09 — CI/CD real no GitHub Actions

**Contexto.** A rubrica permite "simulado ou real".

**Decisão.** Real. Workflow executa lint, testes, smoke-train e build Docker em cada push/PR.

**Alternativas consideradas.**

- YAML só de exemplo sem execução: atende o texto da rubrica mas não o espírito.

**Consequências.** Badge verde de CI no repositório; quem clona consegue verificar reprodutibilidade em infraestrutura efêmera.

---

## ADR-10 — Limites de recurso multiplataforma

**Contexto.** A máquina de desenvolvimento (e a dos revisores) precisa continuar utilizável durante treinos longos.

**Decisão.** Stack em três camadas:

1. Env vars BLAS (`OMP_NUM_THREADS` etc.) — portáveis entre SOs.
2. `n_jobs=4` explícito em `config.py`.
3. `psutil.nice()` — cross-platform.

No Makefile, wrappers específicos de cada SO (macOS `taskpolicy`, Linux `nice/ionice`) adicionam uma camada de isolamento extra sem quebrar compatibilidade.

**Alternativas consideradas.**

- Apenas Docker com `--cpus`: funciona mas exige Docker. Mantivemos como opção.
- Apenas `taskpolicy`: quebra em Linux/Windows.

**Consequências.** Projeto utilizável sem degradar performance da máquina.

---

## ADR-11 — Deploy na AWS com Lambda + API Gateway via CDK

**Contexto.** A rubrica pede "expor o modelo por meio de um serviço de inferência" e "integrar o deploy a um pipeline de CI/CD simulado ou real". Um serviço acessível apenas em `localhost` demonstra a capacidade técnica, mas não a operacionalização completa.

**Decisão.** Deployar o modelo como uma função Lambda (container ARM64) exposta via API Gateway HTTP, com infraestrutura definida como código em AWS CDK. A stack inclui uma regra de warming (CloudWatch Events a cada 10 min) e throttling (10 req/s).

**Alternativas consideradas.**

- **EC2 (t2.micro free tier)**: servidor dedicado, mas exige gerenciar uptime, patches e segurança.
- **ECS Fargate**: robusto para produção, mas custo mínimo de ~$10/mês para um container always-on.
- **Render/Railway (free tier)**: simples de configurar, mas fora do ecossistema AWS e com limites de sleep após inatividade.
- **Apenas localhost + Docker Compose**: funciona para demonstração local, mas o revisor não consegue acessar sem clonar e rodar.

**Por que Lambda:**

- Custo real: $0,00 dentro do free tier (1M requests/mês + 400K GB-segundos).
- Sem gerenciamento de servidor — escala automaticamente e desliga quando não há tráfego.
- A warming rule elimina o cold start para o uso esperado (avaliação pelo professor).
- Destruição com um comando (`./destroy.sh`) — sem risco de esquecer um recurso rodando.

**Por que CDK em TypeScript (e não Python):**

- CDK foi criado em TypeScript — melhor suporte de IDE, mais documentação e exemplos.
- A linguagem da infraestrutura e do runtime são preocupações separadas: TypeScript define o que deployar, Python define o que roda.
- Padrão comum na indústria: CDK TypeScript + Lambda Python.

**Por que o modelo é extraído como joblib (e não carregado via MLflow):**

- MLflow adiciona ~200 MB de dependências (SQLAlchemy, Flask, etc.) ao runtime. Desnecessário para inferência.
- O deploy extrai o Pipeline vencedor do Registry uma vez (`joblib.dump`) para a imagem Docker.
- Na execução da funcão Lambda, `joblib.load()` carrega o mesmo objeto Python — mesmas transformações, mesmas predições.
- Versão do scikit-learn fixada no Dockerfile para evitar incompatibilidade de serialização.

**Consequências.** +API acessível publicamente com URL fixa; +custo zero; +infraestrutura versionada no repositório; +destruição garantida com um comando; −modelo congelado na imagem (atualizar exige re-deploy, não apenas promover no Registry).
