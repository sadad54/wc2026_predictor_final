# 2026 FIFA World Cup Prediction Engine

A production-quality machine learning pipeline for predicting 2026 FIFA World
Cup match outcomes, tournament progression, and individual player awards.

## Architecture

- **Dixon-Coles Poisson model** — statistical goals model (attack/defense parameters)
- **Custom Elo rating system** — built from scratch, tuned for international football  
- **XGBoost + SHAP** — gradient boosting with feature-level explainability
- **Random Forest** — tree ensemble for diversity in the final stack
- **Stacked ensemble** — meta-learner that learns optimal model weights
- **Monte Carlo simulation** — 10,000 full tournament simulations per run

## Project Structures
worldcup-predictor/
├── config/         # Central YAML configuration
├── data/           # Raw, processed, and external data (tracked by DVC)
├── models/         # Serialised models and evaluation metrics
├── notebooks/      # Exploratory analysis and visualisations
├── src/            # All source code
└── tests/          # Unit tests (pytest)
## Setup

```bash
git clone <your-repo-url>
cd worldcup-predictor
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env   # fill in Kaggle credentials
python validate_setup.py
```

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Project architecture & setup | ✅ Complete |
| 1 | Data collection | 🔄 Next |
| 2 | Preprocessing | ⏳ Pending |
| 3 | Feature engineering | ⏳ Pending |
| 4 | Model building | ⏳ Pending |
| 5 | Validation | ⏳ Pending |
| 6 | Match simulator | ⏳ Pending |
| 7 | Tournament simulator | ⏳ Pending |
| 8 | Dashboard & portfolio | ⏳ Pending |

## Running the tests

```bash
pytest tests/ -v
```