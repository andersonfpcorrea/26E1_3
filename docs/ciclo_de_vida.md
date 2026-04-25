# Ciclo de Vida do Modelo em Produção

Este documento descreve como o modelo de triagem de DCV é **monitorado, auditado e re-treinado** após o deploy. Cobre a Parte 6 da rubrica ("Operacionalização e Simulação de Produção").

A premissa central: **um sistema de ML em produção é vivo**. O desempenho vai degradar naturalmente à medida que a população atendida muda, novos hábitos de saúde surgem e a prática clínica evolui. O trabalho não termina quando o modelo é implantado.

---

## 1. Métricas monitoradas

### 1.1 Métricas técnicas (`evaluation/metrics.py`)

| Métrica                  | O que mede                                           | Gatilho de alerta                            |
| ------------------------ | ---------------------------------------------------- | -------------------------------------------- |
| `tech.accuracy`          | Proporção de acertos                                 | queda > 3 pp vs. baseline histórico          |
| `tech.precision`         | Acertos entre os classificados positivos             | queda > 3 pp                                 |
| `tech.recall`            | Positivos capturados entre os reais                  | queda > 5 pp (mais sensível — custo clínico) |
| `tech.f1`                | Média harmônica precisão/recall                      | queda > 3 pp                                 |
| `tech.roc_auc`           | Discriminação independente de threshold              | queda > 0.02                                 |
| `tech.average_precision` | Área sob curva PR (útil para classes desbalanceadas) | queda > 0.03                                 |

### 1.2 Métricas de negócio (`evaluation/metrics.py`)

| Métrica                            | Fórmula                      | Significado                                                  |
| ---------------------------------- | ---------------------------- | ------------------------------------------------------------ |
| `biz.expected_cost_per_case`       | `(5 × FN + 1 × FP) / N`      | custo médio esperado — traduz confusão em dor                |
| `biz.capture_rate`                 | `TP / (TP + FN)`             | proporção de doentes identificados                           |
| `biz.false_alarm_rate`             | `FP / (FP + TN)`             | proporção de saudáveis sinalizados                           |
| `biz.positives_caught_over_random` | `TP − prevalência × (TP+FN)` | casos positivos identificados a mais vs. um modelo aleatório |

Os pesos do custo esperado (5×1) são auditáveis e ajustáveis no módulo de métricas.

---

## 2. Detecção de drift

Implementada em `evaluation/drift.py`. Dois mecanismos complementares por feature numérica:

### 2.1 PSI (Population Stability Index)

Limiares de referência adotados:

| PSI         | Classificação | Ação                         |
| ----------- | ------------- | ---------------------------- |
| < 0.10      | `estavel`     | nenhuma                      |
| 0.10 — 0.25 | `moderado`    | investigar; log no dashboard |
| > 0.25      | `drift`       | alerta — acionar playbook    |

### 2.2 Kolmogorov-Smirnov

Teste não-paramétrico. Guardamos a estatística `D` e o `p-valor`. Usado em conjunto com PSI porque:

- PSI sumariza magnitude em um número único, comparável entre features.
- KS fornece significância estatística, evitando alarme falso em janelas pequenas.

### 2.3 Drift de modelo (predição)

A mesma metodologia é aplicada à **distribuição das probabilidades preditas** pelo modelo. Uma mudança brusca aqui indica que o modelo está enxergando padrões diferentes, mesmo se os dados brutos parecerem estáveis (drift concept). Implementado no `compute_drift(..., ref_predictions, cur_predictions)`.

### 2.4 Relatório Evidently

Complemento visual para auditoria humana: `scripts/simulate_drift.py` gera `reports/drift_evidently.html`.

---

## 3. Cadência de monitoramento

| Frequência     | Ação                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------- |
| **Tempo real** | Logs de latência, taxa de erro HTTP, throughput da API                                      |
| **Diário**     | Comparação de distribuição das features do dia vs. janela de referência (últimas 4 semanas) |
| **Semanal**    | Verificação de drift de predição; revisão do custo esperado se houver feedback de outcome   |
| **Mensal**     | Auditoria completa: subgrupos (gênero, idade), revisão dos pesos 5:1 com equipe clínica     |
| **Trimestral** | Decisão formal sobre re-treinamento (ver §4)                                                |

---

## 4. Estratégia de re-treinamento

### 4.1 Gatilhos

Re-treinar **quando pelo menos um** dos seguintes ocorrer:

1. PSI > 0.25 em 2 ou mais features numéricas relevantes (`ap_hi`, `age_years`, `bmi`, `cholesterol`).
2. Recall cai mais de 5 pontos percentuais na janela rolante de 30 dias vs. baseline do Registry.
3. Custo esperado por caso sobe acima de um limite pactuado com a equipe clínica.
4. Mudança explícita na prática clínica ou nas diretrizes que mudem o custo relativo de FN vs. FP.
5. Tempo: re-treinamento preventivo a cada 6 meses, mesmo sem drift detectado.

### 4.2 Procedimento

1. Coletar os dados acumulados em produção (últimos `N` meses).
2. Rodar `make experiment` com o dataset ampliado.
3. Rodar `make select` — comparar o candidato vencedor com o modelo em Produção atual.
4. Promover apenas se a diferença for estatisticamente relevante (holdout + teste binomial).
5. Implantar como uma versão separada, configurando um **alias com peso** (weighted alias) para direcionar ~10% do tráfego para a versão nova e ~90% para a versão atual.
6. Monitorar métricas da nova versão durante 1 semana (latência, erros, drift nas predições).
7. Se as métricas forem satisfatórias, mover 100% do tráfego para a nova versão. Caso contrário, remover o peso — a versão anterior continua intacta.

**Nota:** o script `./deploy.sh` deste repositório faz deploy direto (substitui a versão atual de uma vez). Para produção real, seria necessário estender o CDK para criar aliases com pesos configuráveis e automatizar o aumento gradual de tráfego.

### 4.3 Aprendizado contínuo

Este projeto intencionalmente **não** usa aprendizado online (atualização de pesos em streaming) porque:

- Dataset tabular não se beneficia significativamente de streaming.
- Modelos em streaming são difíceis de auditar em contexto clínico.
- Estabilidade de comportamento é mais importante que adaptação instantânea.

Em compensação, o processo de re-treino batch é **barato** (~minutos de CPU) e totalmente rastreado no MLflow, o que atende ao objetivo de evolução contínua sem os riscos do online learning.

---

## 5. Relação com o MLflow Model Registry

Cada re-treino resulta em uma **nova versão** no Registry. Fluxo de vida de uma versão:

1. `None` → recém-registrada pelo `select_final_model.py`.
2. `Staging` → validação manual ou canário.
3. `Production` → em uso pela API (rotas `/predict`).
4. `Archived` → modelos antigos mantidos para auditoria.

**Localmente** (`make serve`): a API carrega automaticamente a versão em `Production` no startup (ver `inference/predict.py::_resolve_model_version`). Promover uma nova versão e reiniciar o servidor é suficiente — sem rebuild.

**Na AWS** (Lambda): o modelo é extraído do Registry e embutido na imagem Docker durante o deploy. Atualizar o modelo em produção exige re-executar `./deploy.sh`, que exporta a nova versão, reconstrói a imagem e atualiza a Lambda automaticamente.

---

## 6. Riscos conhecidos e mitigação

| Risco                                                                            | Mitigação                                                                                |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Drift silencioso (features estáveis, relação feature-target muda)                | Monitorar distribuição das probabilidades preditas, não só das features                  |
| Feedback loop (modelo rejeitando pacientes que depois não aparecem no re-treino) | Manter 5% de decisões sob amostragem aleatória para coleta de verdade de campo           |
| Viés em subgrupos                                                                | Auditoria mensal por gênero e faixa etária; alerta se F1 divergir > 5 pp entre grupos    |
| Dependência do único modelo em Produção                                          | Versões anteriores permanecem `Archived` no Registry — rollback em segundos via promoção |

---

## 7. Limites explícitos

Este projeto é uma **simulação de operação**, não um sistema clínico real. Para uso em produção seriam necessárias:

- Aprovação regulatória (ANVISA, HIPAA-like) e validação clínica longitudinal.
- Segurança: autenticação, auditoria de acesso, criptografia em trânsito e em repouso.
- SLA e observabilidade em grau de produção (Prometheus, OpenTelemetry).
- Plano de resposta a incidentes.

A arquitetura atual **não impede** essas adições futuras — as camadas estão deliberadamente isoladas para permitir evolução.
