# Used Car Value Retention Prediction (CMPSC 445)

Predicting the value retention ratio of used cars:

    value retention ratio = current listing price / original MSRP

## Folders

- `src/` - Python scripts, one per step of the pipeline
- `data/raw/` - raw scraped CSVs (saved right away so we don't re-scrape)
- `data/processed/` - cleaned and merged datasets
- `plots/` - saved charts

## Setup

Open a terminal in this folder and activate the virtual environment:

    .\venv\Scripts\Activate.ps1

Install the packages (only needed once):

    pip install -r requirements.txt

## Pipeline (run in order)

Run everything from the project folder.

1. `python src/scrape_autolist.py` - scrapes Autolist -> `data/raw/autolist_raw.csv` (a few minutes)
2. `python src/scrape_craigslist.py` - scrapes Craigslist -> `data/raw/craigslist_raw.csv` (about an hour; can be stopped and re-run, it skips listings it already has)

3. `python src/scrape_msrp.py` - collects trim-level MSRP from J.D. Power -> `data/raw/msrp_raw.csv` (about 20 minutes; re-run it to retry any model years that got blocked)

Add `test` at the end of any scraper command to do a tiny test run.

4. `python src/clean_merge.py` - cleans both sources, merges them, attaches MSRP, computes `value_retention` -> `data/processed/cars_clean.csv` (a few seconds)
5. `python src/explore_data.py` - summary statistics -> `data/processed/summary_stats.txt`, and 8 charts -> `plots/`
6. `python src/feature_engineering.py` - age, description features, encodings, train/test split -> `data/processed/cars_features.csv`
7. `python src/train_models.py` - trains Linear Regression, Random Forest, Gradient Boosting; 5-fold CV + test scores -> `data/processed/model_results.csv`, charts -> `plots/`
8. `python src/analyze_features.py` - real-unit coefficients, permutation feature ranking, depreciation curves -> `data/processed/feature_analysis.txt`, charts -> `plots/`

`src_backup/` is a copy of the code from before the cleanup, kept for comparison.

`src/config.py` holds the shared settings (makes/models, cities, delays).
