# income-ml-platform

A small production-style ML platform: Dockerized MySQL for storage, a reusable preprocessing/training pipeline built around scikit-learn's `Pipeline` API, an Optuna-tuned voting ensemble, a FastAPI service for inference, and a Streamlit client that talks to it over HTTP. The classification task itself — predicting whether someone earns above $50K from census attributes — is almost incidental; it's the vehicle for building out the surrounding infrastructure: a database layer, a packaged training pipeline, a served model, and a client, wired together the way you'd actually ship this rather than the way you'd leave it in a notebook.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-4479A1?logo=mysql&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006ACC)
![LightGBM](https://img.shields.io/badge/LightGBM-9ACD32)
![Optuna](https://img.shields.io/badge/Optuna-0078D7)
![SHAP](https://img.shields.io/badge/SHAP-explainability-red)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white)

## Why this exists

Most public repos built around the Adult Census dataset stop at a notebook with a `.fit()` call and a printed accuracy score. This one treats the notebook as the research phase and everything after it as a separate concern: once a preprocessing decision or a model is validated in `ml.ipynb`, it gets pulled out into a module that both the training script and — indirectly — the API depend on. Nothing that runs in production lives only in a notebook cell.

## Architecture

![Workflow](ml_pipeline_workflow_.svg)



`fastapi_app.py` wraps its database connection in a `try/except` at startup instead of assuming MySQL is reachable. Locally, predictions get logged to the same MySQL container used for training data. On Hugging Face Spaces, where there's no database container, the connection attempt fails once, the API notices, and it keeps serving predictions without logging instead of refusing to boot. That one detail is the difference between code written for a fixed environment and code written to run in more than one.

## Engineering notes

- **Modular by necessity, not convention.** `database.py`, `data_preprocessor.py`, `ml_pipeline.py`, `ml_utils.py`, and `optuna_config.py` each own one concern. `save_model.py` — the actual training entry point — is five lines because everything it calls already exists as a tested unit.
- **`ML_Pipeline` and `DataBase_Manager` are classes, not scripts.** `ML_Pipeline.__init__` runs `prepare_data` and `build_model` up front, then exposes `train_model`, `evaluate`, and `save_model` as separate steps you can call independently — useful when you want to evaluate without immediately overwriting `model.pkl`.
- **One preprocessing object, three consumers.** The same `preprocessing_pipeline` (`ColumnTransformer` + `RobustScaler`) is threaded through Optuna's objective function, the final training run, and the serialized model. There's no separate "training-time" and "inference-time" transform to keep in sync — the classic source of train/serve skew in ML systems.
- **Config isn't scattered.** Hyperparameter search spaces and the final tuned values both live in `optuna_config.py`, so nothing about the model's configuration is hardcoded three files away from where it's used.
- **Graceful degradation over hard dependency.** The FastAPI/MySQL fallback described above means the same image can run in a fully local Docker Compose setup or standalone on a platform with no database at all.

## Machine learning

### Exploratory analysis

`eda.ipynb` runs on `edatoolkit`'s `Inspector` class from the start — it separates columns into categorical, numerical, and high-cardinality ("categorical but cardinal") groups automatically, which is what later drives which encoder each column gets. The initial pass already flags data quality problems worth acting on: 15 columns, north of 32,000 rows, a handful of exact duplicates, and — after cross-checking `education` against `education.num` — full agreement between the two, which is what confirms `education.num` is safe to drop as a redundant encoding rather than something that might carry extra signal.

Two statistical tools do most of the work in deciding what matters:

- **Categorical features vs. target.** A straight Chi-square test isn't trustworthy at this sample size — with 30K+ rows almost anything comes back "significant" — so the notebook uses **Cramer's V** for effect size and compares observed vs. expected counts from the contingency tables directly. Education and marital status/relationship come out moderate-to-strong: higher education levels and `Married-civ-spouse` status are heavily over-represented in the >50K group, `Never-married` and `Own-child` are heavily concentrated in ≤50K. Occupation shows a similar pattern — `Exec-managerial` and `Prof-specialty` skew high-income, `Adm-clerical` skews low. Workclass, race, and sex land weak-to-moderate; the effect is real but smaller.
- **Numerical features vs. target.** None of the numerical columns are normally distributed (checked with `edatoolkit`'s `NormalityAnalyzer` via skewness, kurtosis, and Q-Q plots against histograms and box plots), which rules out a t-test and points to the **Mann-Whitney U** test instead. `age` and `hours.per.week` show small-to-moderate effect sizes — both increase the odds of >50K, consistent with seniority and full-time-plus work correlating with income. `fnlwgt` comes back with essentially zero effect, which is expected — it's a census sampling weight, not a personal attribute — and is the statistical basis for dropping it later rather than just a judgment call. `capital.gain`/`capital.loss` are trickier: both are zero for most of the population (median and even the 75th percentile sit at 0), but the *mean* is markedly higher in the >50K group, meaning the rare large capital events are concentrated among higher earners. That skew is also what `edatoolkit`'s `OutlierAnalyzer` later caps at the 4th/96th percentile instead of trimming outright — the extreme values are informative, not noise.

### Cleaning

`data_preprocessor.py` drops a handful of internally contradictory rows — `sex = Male` with `relationship = Wife` and the mirror case, plus `marital.status = Married-civ-spouse` paired with a relationship that contradicts it. These get dropped rather than imputed: when two categorical fields disagree, there's no principled way to decide which one is wrong. Rare categories are merged with domain logic instead of dumped into a generic bucket — `Never-worked`/`Without-pay` into `Other`, `Armed-Forces` into `Protective-serv`, `Married-AF-spouse` into `Married-civ-spouse`. `fnlwgt` (a census sampling weight, not a feature) and `education.num` (a redundant numeric copy of `education`) are dropped outright.

### Imputation

The only missing values are in `workclass`, `occupation`, and `native.country` — all categorical, all encoded as `"?"` in the raw data. Instead of mode imputation, each column gets its own `LGBMClassifier` trained on the other features, predicting the missing category directly. LightGBM handles categoricals natively, so there's no manual encoding step, and it's fast enough to retrain per column without slowing the pipeline down. Columns are imputed in order of increasing missing count (`native.country` → `workclass` → `occupation`) so each model can use the previously imputed columns as complete features.

### Outliers and encoding

`capital.gain` and `capital.loss` are heavily zero-inflated — most people report zero, a small tail reports large values — so they're capped at the 4th/96th percentile (Winsorization) rather than trimmed, which keeps the rows. Categorical features are one-hot encoded, `education` gets an explicit ordinal encoding since it has a real progression from `Preschool` to `Doctorate`, and everything is scaled with `RobustScaler` to absorb whatever outliers survive capping.

### Model selection

Eight baseline models, no tuning:

| Model | F1 | ROC-AUC | Accuracy |
|---|---|---|---|
| CatBoost | 0.716 | 0.930 | 0.874 |
| LightGBM | 0.716 | 0.930 | 0.873 |
| XGBoost | 0.715 | 0.929 | 0.872 |
| Gradient Boosting | 0.685 | 0.923 | 0.866 |
| KNN | 0.689 | 0.889 | 0.856 |
| AdaBoost | 0.648 | 0.907 | 0.854 |
| Random Forest | 0.669 | 0.896 | 0.849 |
| Logistic Regression | 0.639 | 0.897 | 0.844 |

AdaBoost and `GradientBoostingClassifier` were dropped before tuning — same boosting logic as the other three, lower scores, no reason to spend Optuna trials on them.

Tuning (Optuna, 50 trials per model, maximizing F1, run through the same preprocessing pipeline used everywhere else):

| Model | Best F1 |
|---|---|
| XGBoost | 0.7364 |
| LightGBM | 0.7247 |
| CatBoost | 0.7244 |
| KNN | 0.7106 |
| Logistic Regression | 0.6667 |
| Random Forest | 0.6561 |

### Ensemble

The deployed model is a soft `VotingClassifier`: XGBoost, LightGBM, KNN, and Logistic Regression. CatBoost, despite scoring close to LightGBM, was left out on purpose — it's driven by the same boosting mechanics as the other two GBMs, takes noticeably longer to train, and tends to be wrong about the same cases they're wrong about. Adding it would cost training time without buying the ensemble anything. KNN and Logistic Regression stay in specifically because they *don't* think like tree-based models — KNN is instance-based, Logistic Regression is linear — so their errors are less correlated with the boosting models', which is what a voting ensemble actually needs to gain anything over its best single member. Several other combinations were tried before landing on this one.

### Explainability

SHAP on the tuned LightGBM model: `marital.status` leads global importance (+0.86), then `age`, then `relationship`; `native.country` sits at the bottom (+0.03), which tracks — it's high-cardinality and overwhelmingly `United-States`. A waterfall plot on a single sample confirmed the same top features dominate locally, which is a useful sanity check in itself: it means the model isn't relying on different signals depending on the row.

## Evaluation

Held-out test set:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 (≤50K) | 0.90 | 0.94 | 0.92 | 4907 |
| 1 (>50K) | 0.79 | 0.68 | 0.73 | 1577 |

Accuracy 0.88, ROC-AUC 0.92. Class 1 metrics are weaker mainly because of class imbalance (~76/24 split), not a modeling flaw. The learning curve shows train and validation F1 converging as training size grows — no overfitting gap to worry about — which is why the final model gets refit on the full dataset before being serialized.

## Project structure

```
.
├── database/
│   ├── database.py            # SQLAlchemy connection manager
│   └── load_data.py           # loads the raw CSV into MySQL
├── dataset/                   # raw CSV (adult.csv)
├── edatoolkit/                # EDA package used in the notebooks
├── fastapi_app/
│   └── fastapi_app.py
├── notebooks/
│   ├── eda.ipynb
│   └── ml.ipynb
├── requirements/
│   ├── requirements-api.txt   # slim set for the Docker image
│   └── requirements.txt       # full dev environment
├── streamlit_app/
│   └── streamlit_app.py
├── workflow_diagram/
│   └── ml_pipeline_workflow_.svg
├── .dockerignore
├── .env
├── .gitignore
├── data_preprocessor.py
├── docker-compose.yaml
├── Dockerfile
├── LICENSE
├── ml_pipeline.py
├── ml_utils.py
├── model.pkl
├── optuna_config.py
├── README.md
└── save_model.py
```

## Running it locally

Requires Docker and Docker Compose. Python 3.10+ if you want to run the notebooks or scripts outside containers.

Create a `.env` in the project root:

```
MYSQL_DATABASE=census_income_db
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password
MYSQL_ROOT_PASSWORD=your_root_password
```

Then:

```bash
docker compose up --build
```

This brings up `income_db` (MySQL, persistent volume) and `income_fastapi` (built from the local `Dockerfile`, port 8000, connects to MySQL over the `database` service name). To seed the table, run `load_data.py` once locally with `DB_HOST=localhost` in your `.env`.

Streamlit isn't in `docker-compose.yaml` — run it separately:

```bash
pip install -r requirements/requirements.txt
streamlit run streamlit_app/streamlit_app.py
```

By default it points at the hosted API on Hugging Face. Point `API_URL` in `streamlit_app.py` at `http://localhost:8000/predict` to test against your local stack instead.

To retrain: `python save_model.py` — pulls `census_income` from MySQL, runs `ML_Pipeline`, prints a classification report, writes `model.pkl`.

## Live

| Service | URL |
|---|---|
| Streamlit | https://elvin-aliyev-census-income-streamlit.hf.space |
| FastAPI | https://elvin-aliyev-census-income-api.hf.space/predict |

## Stack

**Language** Python
**ML** scikit-learn, XGBoost, LightGBM, CatBoost (comparison only)
**Data** Pandas, NumPy, SQLAlchemy
**Explainability** SHAP
**Tuning** Optuna
**API** FastAPI, Pydantic, Uvicorn
**Frontend** Streamlit
**Database** MySQL, PyMySQL
**Infra** Docker, Docker Compose
**Deployment** Hugging Face Spaces
**Tooling** Jupyter, python-dotenv, Joblib

## References

- Dataset: [Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income) (Kaggle)
- [edatoolkit](https://github.com/elvinaliyev2006/edatoolkit)

## License

MIT — see [LICENSE](LICENSE).