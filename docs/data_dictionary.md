# Dataset de Doenças Cardiovasculares - Dicionário de Dados

**Fonte:** Kaggle (sulianova/cardiovascular-disease-dataset)
**URL:** https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

## Visão Geral do Dataset

- **Observações:** 70.000 pacientes
- **Features:** 11 atributos + 1 ID + 1 variável-alvo
- **Domínio:** Cardiologia / Doenças Cardiovasculares
- **Tarefa:** Classificação Binária (presença/ausência de doença cardiovascular)
- **Arquivo:** cardio_train.csv
- **Delimitador:** Ponto e vírgula (;)

---

## Descrição das Features

| #   | Feature         | Tipo           | Descrição                          | Valores/Intervalo                |
| --- | --------------- | -------------- | ---------------------------------- | -------------------------------- |
| 1   | **id**          | Identificador  | ID do paciente                     | Inteiro único                    |
| 2   | **age**         | Numérico       | Idade em DIAS (converter: age/365) | ~30-65 anos                      |
| 3   | **gender**      | Categórico     | Gênero                             | 1 = Feminino, 2 = Masculino      |
| 4   | **height**      | Numérico       | Altura em cm                       | 55-250 cm                        |
| 5   | **weight**      | Numérico       | Peso em kg                         | 10-200 kg                        |
| 6   | **ap_hi**       | Numérico       | Pressão arterial sistólica         | mm Hg                            |
| 7   | **ap_lo**       | Numérico       | Pressão arterial diastólica        | mm Hg                            |
| 8   | **cholesterol** | Categórico     | Nível de colesterol                | 1=Normal, 2=Acima, 3=Muito acima |
| 9   | **gluc**        | Categórico     | Nível de glicose                   | 1=Normal, 2=Acima, 3=Muito acima |
| 10  | **smoke**       | Binário        | Tabagismo                          | 0=Não, 1=Sim                     |
| 11  | **alco**        | Binário        | Consumo de álcool                  | 0=Não, 1=Sim                     |
| 12  | **active**      | Binário        | Atividade física                   | 0=Inativo, 1=Ativo               |
| 13  | **cardio**      | Binário (Alvo) | Doença cardiovascular              | 0=Ausente, 1=Presente            |

---

## Contexto Clínico

### O que é Doença Cardiovascular?

Doença cardiovascular (DCV) inclui condições que afetam o coração e vasos sanguíneos:

- Doença arterial coronariana
- Insuficiência cardíaca
- Acidente vascular cerebral (AVC)
- Hipertensão

É a principal causa de morte no mundo.

### Diretrizes de Pressão Arterial

| Categoria             | Sistólica (ap_hi) | Diastólica (ap_lo) |
| --------------------- | ----------------- | ------------------ |
| Normal                | <120              | <80                |
| Elevada               | 120-129           | <80                |
| Hipertensão Estágio 1 | 130-139           | 80-89              |
| Hipertensão Estágio 2 | >=140             | >=90               |
| Crise                 | >180              | >120               |

---

## Pré-processamento Necessário

### 1. Conversão de Idade (dias para anos)

```python
df['age_years'] = df['age'] / 365
```

### 2. Cálculo do IMC

```python
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
```

### 3. Limpeza de Pressão Arterial

```python
# Remover leituras impossíveis
df = df[(df['ap_hi'] > 0) & (df['ap_hi'] < 300)]
df = df[(df['ap_lo'] > 0) & (df['ap_lo'] < 200)]
df = df[df['ap_lo'] < df['ap_hi']]  # diastólica deve ser < sistólica
```

### 4. Outliers de Altura/Peso

```python
df = df[(df['height'] > 100) & (df['height'] < 220)]
df = df[(df['weight'] > 30) & (df['weight'] < 200)]
```

---

## Oportunidades de Feature Engineering

1. **Categorias de IMC:** Abaixo do peso (<18,5), Normal (18,5-24,9), Sobrepeso (25-29,9), Obeso (>=30)
2. **Pressão de Pulso:** ap_hi - ap_lo (indicador de rigidez arterial)
3. **Grupos de Idade:** Agrupamento em décadas

---

## Distribuição de Classes

- Classe 0 (Sem DCV): ~35.000 (~50%)
- Classe 1 (DCV Presente): ~35.000 (~50%)

Classes estão balanceadas - não é necessário reamostragem.

---

## Por que Este Dataset?

| Aspecto           | UCI Cleveland | Cardiovascular Disease |
| ----------------- | ------------- | ---------------------- |
| Registros         | 303           | **70.000**             |
| Features          | 13            | 11                     |
| Valores faltantes | Sim           | Não                    |
| Escala            | Pequeno       | Mundo real             |

O dataset maior demonstra:

- Validação cruzada em escala
- Divisões treino/teste realistas
- Considerações computacionais
- Significância estatística

---

## Referências

1. Dataset: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
2. OMS Doenças Cardiovasculares: https://www.who.int/health-topics/cardiovascular-diseases
