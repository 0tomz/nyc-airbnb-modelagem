# NYC Airbnb — EDA e Regressão

Projeto de Machine Learning em duas etapas sobre o dataset **New York City Airbnb Open Data**: análise exploratória (APS1) e regressão do preço anunciado por noite (APS2).

## Visão Geral

- **Dataset:** New York City Airbnb Open Data.
- **Problema:** regressão para previsão de `price`, o preço anunciado por noite em USD.
- **Autores:** Maria Eduarda Oliveira Galdino e Tomaz Santos Almeida.
- **Prazo da APS2 na rubrica:** **26/10**, conforme `Projeto_de_Machine_Learning (1).ipynb` (o enunciado não informa o ano).
- **Site do projeto (APS1 e APS2):** <https://0tomz.github.io/nyc-airbnb-modelagem/>.
- **Site anterior (APS1):** <https://meog38.github.io/nyc-airbnb-eda-pipeline/>.
- **Relatório web:** `index.html` (APS1 e APS2).
- **Notebook APS1:** `Meu_projeto_EDA_ML_revisado (2).ipynb` — entrega final da EDA, preservada.
- **Notebook APS2:** `APS2_Regressao_NYC_Airbnb.ipynb` — modelagem, avaliação, explicabilidade, sensibilidades e discussão crítica.
- **Orientações da disciplina:** `Projeto_de_Machine_Learning (1).ipynb`.

O pré-processamento da APS1 é mantido dentro dos pipelines de modelagem, então cada fold aprende imputação, escala e encoding somente nos dados de treino.

## O que mudou na APS2

| Arquivo | Situação | O que contém |
|---|---|---|
| `APS2_Regressao_NYC_Airbnb.ipynb` | novo | Notebook completo da APS2, com saídas salvas |
| `src/nyc_airbnb_eda/modeling.py` | novo | Pipelines de modelo (`build_model_pipeline`, `build_log_target_pipeline`) e nomes de features |
| `tests/test_modeling.py` | novo | Testes de `modeling.py` |
| `reports/figures/aps2/` | novo | Nove figuras geradas pelo notebook |
| `reports/aps2_results.json` | novo | Números finais gravados pelo notebook (fonte do README e do site) |
| `index.html`, `styles.css` | alterados | Seção da APS2 no relatório web |
| `requirements.txt`, `pyproject.toml` | alterados | Incluem `shap` e `ipykernel` |
| `README.md`, `.gitignore` | alterados | Documentação; ignora `.matplotlib/` e `.jupyter_runtime/` |
| `Projeto_de_Machine_Learning (1).ipynb` | orientações | Enunciado e rubrica da disciplina |

O `preprocessing.py`, o CLI e o notebook da APS1 não foram alterados.

## Estrutura

```text
.
├── data/raw/                  # coloque AB_NYC_2019.csv aqui (não é versionado)
├── reports/
│   ├── figures/               # gráficos da APS1
│   │   └── aps2/              # gráficos da APS2: comparação, curva, teste, coeficientes, importâncias, SHAP
│   └── aps2_results.json      # números finais da APS2
├── src/nyc_airbnb_eda/
│   ├── data.py                # carregamento do dataset
│   ├── eda.py                 # tabelas e visualizações da APS1
│   ├── preprocessing.py       # split e pré-processamento (APS1)
│   ├── modeling.py            # pipelines de modelo da APS2
│   └── cli.py                 # ponto de entrada da APS1
├── tests/                     # testes unitários (pré-processamento e modelagem)
├── index.html, styles.css     # relatório web
├── Meu_projeto_EDA_ML_revisado (2).ipynb
├── APS2_Regressao_NYC_Airbnb.ipynb
├── Projeto_de_Machine_Learning (1).ipynb
├── pyproject.toml
└── requirements.txt
```

## Dataset

O arquivo `AB_NYC_2019.csv` pode ser baixado no Kaggle: <https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data>

O notebook da APS2 lê `data/raw/AB_NYC_2019.csv` e para com erro claro se o arquivo não existir. O CSV bruto não é versionado (`.gitignore`). O CLI da APS1, sem `--input`, tenta uma cópia pública definida em `src/nyc_airbnb_eda/data.py` quando não acha o arquivo local.

## Execução Local

No PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.nyc_airbnb_eda.cli
```

O `requirements.txt` inclui `seaborn` e `shap`, usados na APS2. O CLI da APS1 gera tabelas em `reports/` e figuras em `reports/figures/`.

Para a APS2, deixe o CSV em `data/raw/AB_NYC_2019.csv`, abra `APS2_Regressao_NYC_Airbnb.ipynb` **na raiz do repositório** (o notebook confere a pasta) pelo VS Code ou Jupyter, escolha o kernel do ambiente acima e execute todas as células (Restart & Run All). **Tempo de execução: cerca de 21 minutos** num PC com 20 núcleos; as buscas da Random Forest, a curva de aprendizado, o SHAP e as sensibilidades são as partes mais pesadas. O notebook grava as figuras em `reports/figures/aps2/` e os números em `reports/aps2_results.json`.

Para indicar outro caminho de CSV no CLI da APS1:

```powershell
python -m src.nyc_airbnb_eda.cli --input caminho\AB_NYC_2019.csv
```

## Relatório Web

A apresentação prioriza os resultados da regressão original. A EDA completa fica em
uma seção expansível de apoio. Os gráficos de apresentação têm rótulos maiores,
diagnósticos separados e links para abrir as imagens em tamanho completo.

Para visualizar localmente:

```powershell
python -m http.server 8765 --bind 127.0.0.1
```

Abra <http://127.0.0.1:8765/>. Para regenerar os gráficos de apresentação:

```powershell
python scripts/render_presentation_figures.py
```

O script usa os resultados salvos em `reports/aps2_results.json` e reajusta somente
a configuração final publicada para redesenhar os diagnósticos, conferindo o MAE
original. Não executa novas buscas nem altera o notebook ou as métricas da APS2.
As imagens ficam em `reports/figures/apresentacao/` e a conferência em
`reports/presentation_checks.json`.

Os experimentos posteriores estão preservados separadamente na branch local
`experimentos/modelagem-robusta`. A entrega principal usa a modelagem original.

A página estática em `index.html` apresenta os resultados das duas etapas:

- resumo do dataset;
- dicionário das 16 colunas;
- auditoria de valores ausentes, inconsistências e extremos;
- estatísticas descritivas das variáveis numéricas relevantes;
- visualizações univariadas, bivariadas e multivariadas;
- relações entre variáveis categóricas e `price`;
- análise de PCA das variáveis numéricas;
- arquitetura do pipeline de pré-processamento;
- comparação de Dummy, Ridge e Random Forest, com e sem `log1p` no alvo;
- avaliação final, curva de aprendizado, resíduos, importâncias, coeficientes e explicações SHAP da APS2.

O CSS principal está em `styles.css`, e os estilos essenciais também estão embutidos no HTML para evitar falhas de renderização em ambientes estáticos.

## Principais Achados

### APS1 — EDA

- A base original tem 48.895 anúncios e 16 colunas.
- O alvo de modelagem é `price`, preço anunciado por noite.
- Existem 11 registros com `price = 0`, removidos como inconsistência do alvo.
- As ausências em `reviews_per_month` e `last_review` representam anúncios sem avaliações; por isso `reviews_per_month` recebe imputação zero.
- `price`, `minimum_nights`, reviews e contagens apresentam caudas longas.
- Manhattan e Brooklyn concentram a maior parte dos anúncios.
- `Entire home/apt` e `Private room` dominam os tipos de acomodação.
- As correlações numéricas isoladas com preço são fracas.
- Tipo de acomodação, borough e bairro mostram diferenças relevantes de preço mediano.
- O PCA ajuda a visualizar a estrutura numérica, mas não separa claramente faixas de preço.

### APS2 — Regressão

Métrica principal: **MAE em dólares**. Escolha de modelo e de hiperparâmetros por validação cruzada de cinco folds (`KFold`, `random_state=42`) **só no treino**; o teste (20%, o mesmo split da APS1) aparece só na avaliação final.

| Modelo | MAE CV ± desvio entre folds (US$) | Configuração |
|---|---|---|
| Dummy (mediana) | 83,66 ± 1,36 | referência |
| Ridge | 70,78 ± 1,49 | `alpha` = 100 |
| Random Forest | 63,42 ± 1,85 | `max_depth` sem limite, `min_samples_leaf` 5, `max_features` 0,20 |
| Ridge + `log1p(price)` | 59,87 ± 1,67 | `alpha` = 10 |
| **Random Forest + `log1p(price)`** (final) | 55,81 ± 1,95 | `max_depth` 20, `min_samples_leaf` 1, `max_features` 0,20 |

- **Mudança no pipeline:** `log1p(price)` no alvo (`TransformedTargetRegressor`; o `predict` devolve dólares). Reduziu o MAE CV da floresta em 12,0% (−US$ 7,61), melhor em 5 dos 5 folds. O Ridge com log (US$ 59,87) já supera a floresta sem log (US$ 63,42).
- **Teste:** MAE de **US$ 57,53**, contra US$ 85,93 do Dummy (33,1% menos erro). MAE CV na seleção: US$ 55,81.
- **Onde erra:** decil de maior preço com MAE de US$ 306,67 e resíduo médio `real − previsto` de +US$ 303,48 (subestima). Os 62 anúncios de US$ 1.000 ou mais respondem por 23,5% do erro absoluto. Anúncios sem avaliações têm MAE de US$ 93,34, contra US$ 48,35 dos demais.
- **Interpretação:** `room_type` e localização dominam nos três métodos. Importância da floresta (agregada): `room_type` 0,399, `longitude` 0,147, `latitude` 0,104, `neighbourhood` 0,070. SHAP médio: `room_type` 0,372, `longitude` 0,128, `availability_365` 0,092, `neighbourhood_group` 0,086. Ridge + log, maiores coeficientes (em `log1p(price)`, não em dólares): `room_type = Entire home/apt` +0,611, `neighbourhood_group = Staten Island` −0,557, `neighbourhood = Tribeca` +0,529. As explicações são associativas, não causais.
- **Sensibilidades (não substituem o split oficial):** retirar os anúncios com `minimum_nights > 365` do treino não muda o MAE de teste (US$ 57,51 sem esses anúncios contra US$ 57,48 com todos). Dividir por anfitrião (`GroupKFold`) em vez de por linha aumenta o MAE CV de US$ 55,79 para US$ 58,42 (4,7%) nas 5 sementes testadas.

## Pré-Processamento

As decisões de preparação foram reunidas com `Pipeline` e `ColumnTransformer`:

- remover registros com `price <= 0`;
- separar treino e teste antes de aprender imputação, escala ou encoding;
- imputar `reviews_per_month` com zero;
- imputar demais variáveis numéricas com mediana;
- imputar variáveis categóricas com moda;
- aplicar `log1p` em variáveis assimétricas de contagem;
- padronizar numéricas com `StandardScaler`;
- aplicar one-hot encoding em `neighbourhood_group`, `neighbourhood` e `room_type`;
- usar `min_frequency=20` e `handle_unknown="ignore"` para controlar categorias raras e novas;
- (APS2) comparar a regressão com `price` original e com `log1p(price)` no alvo, avaliando o MAE sempre em dólares;
- (APS2) escolher modelo e transformação somente pela validação cruzada; usar o teste reservado só na avaliação final.

## Cobertura Técnica

| Etapa | Evidência no projeto |
|---|---|
| Inspeção inicial | Dicionário de dados, tipos conceituais, dimensões, ausências e inconsistências |
| Análise univariada | Estatísticas descritivas e gráficos para variáveis numéricas e categóricas |
| Análise bivariada/multivariada | Correlações, scatter plots, boxplots conjuntos e preço por categoria |
| Pré-processamento | Missing values, outliers, encoding, escala e split sem vazamento |
| Redução de dimensionalidade | PCA numérico exploratório com interpretação dos componentes |
| Reprodutibilidade | Código modular em `src/`, testes em `tests/` e pipeline executável por CLI |
| Modelos APS2 | DummyRegressor, Ridge e RandomForestRegressor; buscas CV de hiperparâmetros; Ridge e floresta também com `log1p(price)` |
| Avaliação APS2 | MAE em validação cruzada e teste; curva de aprendizado; real × previsto, resíduos × previsto e distribuição dos resíduos; erro por decil |
| Explicabilidade APS2 | Coeficientes do Ridge, importâncias da floresta e SHAP global e em três previsões, cada uma com texto |
| Sensibilidades APS2 | `minimum_nights > 365` e divisão por anfitrião (`GroupKFold`) |
| Discussão APS2 | Duas limitações do modelo, duas dos dados e duas ações de operação, com referências |

## Limitações conhecidas

- A divisão oficial é por linha, para manter a comparação com a APS1; a divisão por anfitrião mostra um erro maior (4,7%).
- A validação por blocos geográficos não foi feita.
- A busca da floresta é aleatória (20 de 144 combinações); a validação cruzada não é aninhada.
- O notebook demora porque as buscas e o SHAP rodam do zero a cada execução.
- A data efetiva de submissão deve ser preenchida pelo grupo; ela não é a data de elaboração do notebook nem o prazo da rubrica (26/10).

## GitHub Pages

A branch `codex/aps2-regression` foi integrada na `main`. O workflow em `.github/workflows/pages.yml` publica o site a cada push na `main`, na URL <https://0tomz.github.io/nyc-airbnb-modelagem/>.

Para habilitar o site, um administrador deve abrir **Settings > Pages**, selecionar **GitHub Actions** em *Build and deployment* e executar **Deploy static page to GitHub Pages** na aba **Actions**. O site anterior da APS1 continua no repositório `Meog38/nyc-airbnb-eda-pipeline`.

Arquivos necessários para a página:

- `index.html`
- `styles.css`
- `.nojekyll`
- `reports/figures/*.png`
- `reports/figures/aps2/*.png`
- `Meu_projeto_EDA_ML_revisado (2).ipynb`
- `APS2_Regressao_NYC_Airbnb.ipynb`

## Referências

- Kaggle. New York City Airbnb Open Data: <https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data>
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.
- Documentação do scikit-learn: `Pipeline`, `ColumnTransformer`, `TransformedTargetRegressor`, `OneHotEncoder`, `StandardScaler`, `PCA`, `GridSearchCV`, `RandomizedSearchCV`, `GroupKFold`, curvas de aprendizado e persistência de modelos.
- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- A discussão crítica do notebook da APS2 cita, com referência completa: Duan (1983), Strobl et al. (2007), Cawley & Talbot (2010), Meinshausen (2006), Kaufman et al. (2012), Roberts et al. (2017), Sculley et al. (2015), Gama et al. (2014), Lundberg et al. (2020) e o NIST AI RMF Playbook.
