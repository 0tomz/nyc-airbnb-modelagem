# NYC Airbnb — EDA e Regressão

Projeto de Machine Learning dividido em duas etapas para o dataset **New York City Airbnb Open Data**: análise exploratória (APS1) e regressão do preço anunciado por noite (APS2).

## Visão Geral

- **Dataset:** New York City Airbnb Open Data.
- **Problema:** regressão para previsão de `price`, o preço anunciado por noite em USD.
- **Autores:** Maria Eduarda Oliveira Galdino e Tomaz Santos Almeida.
- **Prazo informado para a APS2:** 30/10/2026.
- **Site do projeto:** <https://meog38.github.io/nyc-airbnb-eda-pipeline/>.
- **Relatório web:** `index.html` (APS1 e APS2).
- **Notebook APS1:** `Meu_projeto_EDA_ML_revisado (2).ipynb` — entrega final da EDA, preservada.
- **Notebook APS2:** `APS2_Regressao_NYC_Airbnb.ipynb` — modelagem, avaliação, explicabilidade e discussão crítica.

O pré-processamento da APS1 é mantido dentro dos pipelines de modelagem para que cada fold aprenda imputação, escala e encoding somente nos dados de treino.

## Estrutura

```text
.
├── data/raw/                  # coloque AB_NYC_2019.csv aqui, se quiser usar cópia local
├── reports/
│   ├── figures/               # gráficos dos relatórios APS1 e APS2
│   │   └── aps2/              # validação, resíduos, importâncias e SHAP
│   └── *.csv                  # tabelas resumidas geradas pelo pipeline
├── src/nyc_airbnb_eda/
│   ├── data.py                # carregamento do dataset
│   ├── eda.py                 # tabelas e visualizações
│   ├── preprocessing.py       # split e transformações
│   ├── modeling.py            # composição de pré-processamento e estimadores
│   └── cli.py                 # ponto de entrada
├── tests/                     # testes unitários
├── index.html                 # página estática do relatório
├── styles.css                 # estilos da página
├── Meu_projeto_EDA_ML_revisado (2).ipynb
├── APS2_Regressao_NYC_Airbnb.ipynb
├── pyproject.toml
└── requirements.txt
```

## Dataset

O arquivo `AB_NYC_2019.csv` pode ser baixado no Kaggle:

<https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data>

O arquivo `data/raw/AB_NYC_2019.csv` é usado pelo notebook APS2. O CSV bruto não é versionado pelo Git (`.gitignore`). Quando nenhuma cópia local é encontrada, o pipeline da APS1 tenta carregar uma cópia pública definida em `src/nyc_airbnb_eda/data.py`.

## Execução Local

No PowerShell:

```powershell
.\setup.ps1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.nyc_airbnb_eda.cli
```

Abra `APS2_Regressao_NYC_Airbnb.ipynb` na raiz do repositório pelo VS Code ou Jupyter e execute as células em ordem. O notebook requer o CSV em `data/raw/AB_NYC_2019.csv`, SHAP e um kernel Python com as dependências instaladas.

Para indicar outro caminho de CSV:

```powershell
python -m src.nyc_airbnb_eda.cli --input caminho\AB_NYC_2019.csv
```

O comando da APS1 gera tabelas em `reports/` e figuras em `reports/figures/`. O notebook da APS2 salva as figuras de modelagem em `reports/figures/aps2/`.

## Relatório Web

A página estática em `index.html` apresenta os resultados das duas etapas:

- resumo do dataset;
- dicionário das 16 colunas;
- auditoria de valores ausentes, inconsistências e extremos;
- estatísticas descritivas das variáveis numéricas relevantes;
- visualizações univariadas, bivariadas e multivariadas;
- relações entre variáveis categóricas e `price`;
- análise de PCA das variáveis numéricas;
- arquitetura do pipeline de pré-processamento.
- comparação de Dummy, Ridge e Random Forest;
- avaliação final, curva de aprendizado, resíduos, importâncias e explicações SHAP da APS2.

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

- Métrica principal: MAE, em dólares, com validação cruzada de cinco folds no treino.
- MAE CV: Dummy mediana **US$ 83,66 ± 1,36**; Ridge **US$ 70,78 ± 1,33**; Random Forest **US$ 63,74 ± 1,71**.
- A transformação `log1p(price)` foi retunada na Random Forest e reduziu o MAE CV para **US$ 56,12**, ganho de **11,95%** frente à versão sem transformação.
- MAE no conjunto de teste reservado: **US$ 57,78**.
- O erro no decil de preço mais alto foi **US$ 305,11**, com resíduo médio real − previsto de **+US$ 301,59**; o modelo subestima anúncios caros.
- Variáveis de localização e tipo de acomodação tiveram destaque nas importâncias da floresta e no SHAP. As explicações são associativas, não causais.

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
- usar `min_frequency=20` e `handle_unknown="ignore"` para controlar categorias raras e novas.
- comparar a regressão com `price` original e com `log1p(price)` no alvo, avaliando MAE sempre na escala original em USD.
- escolher o modelo e a transformação somente pela validação cruzada; usar o teste reservado para a avaliação final.

## Cobertura Técnica

| Etapa | Evidência no projeto |
|---|---|
| Inspeção inicial | Dicionário de dados, tipos conceituais, dimensões, ausências e inconsistências |
| Análise univariada | Estatísticas descritivas e gráficos para variáveis numéricas e categóricas |
| Análise bivariada/multivariada | Correlações, scatter plots, boxplots conjuntos e preço por categoria |
| Pré-processamento | Missing values, outliers, encoding, escala e split sem vazamento |
| Redução de dimensionalidade | PCA numérico exploratório com interpretação dos componentes |
| Reprodutibilidade | Código modular em `src/`, testes em `tests/` e pipeline executável por CLI |
| Modelos APS2 | DummyRegressor, Ridge e RandomForestRegressor; buscas CV de hiperparâmetros |
| Avaliação APS2 | MAE em validação cruzada e teste; curva de aprendizado e três gráficos de resíduos/regressão |
| Explicabilidade APS2 | Coeficientes Ridge, importâncias da floresta e SHAP global/local |
| Discussão APS2 | Limitações do modelo e dos dados; propostas de monitoramento e manutenção |

## GitHub Pages

O workflow em `.github/workflows/pages.yml` publica a página a cada push na branch `main`. Se o repositório ainda não estiver configurado, abra **Settings > Pages**, selecione **GitHub Actions** em *Build and deployment* e execute o workflow novamente.

Arquivos necessários para a página:

- `index.html`
- `styles.css`
- `.nojekyll`
- `reports/figures/*.png`
- `Meu_projeto_EDA_ML_revisado (2).ipynb`
- `APS2_Regressao_NYC_Airbnb.ipynb`
- `reports/figures/aps2/*.png`

## Referências

- Kaggle. New York City Airbnb Open Data: <https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data>
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.
- Documentação do scikit-learn para `Pipeline`, `ColumnTransformer`, `OneHotEncoder`, `StandardScaler` e `PCA`.
- Documentação do scikit-learn para `GridSearchCV`, MAE, curvas de aprendizado, `GroupKFold` e persistência de modelos.
- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- NIST. *AI Risk Management Framework Playbook — Manage*.
