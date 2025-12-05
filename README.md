# CMSE 830 Midterm Project

### To access the app, [Click Here!](https://blank-app-6ed1ez2skeb.streamlit.app/)

## Features
- Interactive EDA: parallel coordinates and clustered heatmaps for multivariate inspection.
- Feature engineering: log transforms, z-standardization, ratio and polynomial features, and one-hot encoding for categorical variables (`S/C`, `Div`, `Conf`).
- Modeling: two baseline models (Linear Regression and Decision Tree) with test-set evaluation and cross-validated metrics.

## Repository structure
- `streamlit_app.py` — main Streamlit application. Use this to run the app locally.
- `SS.xlsx`, `Bio.xlsx`, `Misc.xlsx` — input data files (not included in this repo). These are expected to be placed in the repository root when running the app.
- `requirements.txt` — Python dependencies.

## Data sources and notes
- `SS.xlsx` — Season statistics exported from NHL stats (per-player season-level features such as `GP`, `G`, `A`, `P`, `S`, `PIM`, `TOI/GP`, etc.).
- `Bio.xlsx` — Biographical and draft data (height, weight, country, draft year/round/overall).
- `Misc.xlsx` — Additional per-player stats included in the final dataset such as `Hits`, `BkS` (blocked shots), `GvA` (giveaways), and `TkA` (takeaways).

## Data dictionary (selected / important fields)
| Column | Type | Description |
|---|---|---|
| `Player` | string | Player full name (identifier for display/hover only) |
| `Team` | string | Team abbreviation |
| `GP` | int | Games played |
| `G` | int | Goals |
| `A` | int | Assists |
| `P` | int | Points (Goals + Assists) — model target |
| `P/GP` | float | Points per game |
| `EVG`, `EVP` | int | Even-strength goals / points |
| `PPG`, `PPP` | int | Power-play goals / points |
| `S` | int | Shots on goal |
| `S%` | float | Shooting percentage |
| `TOI/GP` | float | Time on ice per game |
| `FOW%` | float | Face-off win percentage (mostly for forwards; often missing for defensemen) |
| `Hits`, `BkS`, `GvA`, `TkA` | int | Misc stats from `Misc.xlsx`: hits, blocked shots, giveaways, takeaways |
| `Ctry` | string | Country of origin |
| `Ht`, `Wt` | int | Height (inches), Weight (lbs) |
| `Draft Yr`, `Round`, `Overall` | int | Draft metadata (may contain missing values) |
| `S/C` | categorical | Shoots: `L` / `R` (one-hot encoded option available) |
| `Div` | categorical | Division derived from `Team` (ATL / MET / CEN / PAC) |
| `Conf` | categorical | Conference derived from `Team` (EC / WC) |


## Feature engineering available in-app
- Log transform (`_log`) — uses `np.log1p` to handle zeros.
- Z-score standardization (`_z`) — centered and scaled column-wise.
- Ratio features — user-specified A/B ratio column creation.
- Polynomial features — optional generation (degree configurable) for numeric columns (careful: can explode column count).
- One-hot encoding — for `S/C`, `Div`, `Conf` (options to drop first level or include NaN indicator).

## Modeling approach
- Target: `P` (Points). This is fixed in the app's modeling workflow.
- Features: user-selected numeric columns.
- Train/test split: fixed 50% test set (test_size = 0.5) for consistent evaluation.
- Models implemented:
   - Linear Regression
   - Decision Tree Regressor (for non-linear interactions)
- Evaluation metrics:
   - RMSE (root mean squared error)
   - MAE (mean absolute error)
   - R2 on test set
   - Cross-validated R2 on training set (5-fold) as a rough measure of generalization
