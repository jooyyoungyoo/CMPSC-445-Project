# Results Summary: Used Car Value Retention (CMPSC 445)

Notes for writing the report. All numbers come from the scripts in `src/`
(see README.md for the run order). Charts are in `plots/`.

**Target:** value retention ratio = current listing price / original MSRP

---

## 1. Data collection

| Source | What it is | How we got it | Raw rows |
|---|---|---|---|
| Autolist | National listing site (dealer cars) | Its JSON search API (`/api/v2/search`), 4 pages x 20 cars per model | 954 |
| Craigslist | Classifieds (private owners + dealers) | Search pages in 10 big cities, then each listing page (specs + description) | 959 (478 owner, 481 dealer) |
| J.D. Power | MSRP for each trim | One page per make/model/year, 2008-2026 | 2,514 trim prices |

- **4 makes x 3 models:** Toyota (Camry, Corolla, RAV4), Honda (Civic, Accord, CR-V), Ford (F-150, Escape, Explorer), Chevrolet (Silverado 1500, Equinox, Malibu)
- **Original attributes (more than 10):** price, year, make, model, trim, mileage, body type, transmission, drivetrain, fuel type, cylinders, exterior color, condition, title status, seller type, state, listing date, description
- **Scraping etiquette:** 3-second delay between requests, robots.txt checked, saved after every search, scrapers resume instead of re-downloading, automatic stop on 403/429

---

## 2. Preprocessing and integration

Rows left after each cleaning step (`src/clean_merge.py`):

| Step | Autolist | Craigslist | Total |
|---|---|---|---|
| Raw combined | 954 | 959 | 1,913 |
| Wrong model removed (e.g. Corolla Cross, Silverado 2500) | 954 | 948 | 1,902 |
| Bad prices removed (missing, $0 placeholders, under $1,000) | 851 | 899 | 1,750 |
| Kept model years 2008-2026 (years with MSRP data) | 835 | 736 | 1,571 |
| Bad mileage removed (missing, over 400k, fake "under 100") | 822 | 735 | 1,557 |
| "Parts only" title removed | 822 | 734 | 1,556 |
| Duplicates removed (5 same VIN, 4 same description) | 821 | 726 | 1,547 |
| Extreme ratios removed (under 0.05 or over 1.5) | **821** | **720** | **1,541** |

**Integration problems fixed:**
- Different column names (`odometer` vs `mileage`, `type` vs `body_type`, `drive` vs `drivetrain`)
- Prices as text (`"$6,900"`) vs numbers (`6900`)
- Craigslist has no AWD option -> combined AWD/4WD/4X4 into `awd/4wd`
- `grey` vs `gray`, `gasoline`/`flex fuel` vs `gas`, `"4 cylinders"` vs `4.0`
- Craigslist gives a city, Autolist gives a state -> converted cities to states
- Trim names don't match J.D. Power ("LE" vs "Sedan 4D LE I4") -> matched on whole words; 1,087 rows matched a trim, 454 used that model year's median trim MSRP

---

## 3. Data exploration (`summary_stats.txt`)

- **1,541 cars:** median price $14,500, median mileage 98,000, median model year 2018
- **Value retention:** median 0.51 (the typical car sells for about half its original MSRP), range 0.06-1.43
- **Balanced:** 112-145 listings per model, about half from each source
- **The sources have different kinds of cars:**

| | Autolist | Craigslist |
|---|---|---|
| Median year | 2020 | 2016 |
| Median mileage | 73k | 121k |
| Median price | $17.8k | $9.6k |
| Median retention | 0.61 | 0.35 |

- **Correlation with value retention:** mileage -0.73, model year +0.68, MSRP +0.15, cylinders -0.07
- **Missing data:** condition 69%, title status 54%, description 53% (Autolist has none of these), color 15%, drivetrain 11%, cylinders 11%

Charts: `price_distribution.png`, `mileage_distribution.png`, `retention_distribution.png`,
`make_model_counts.png`, `year_counts.png`, `retention_by_year.png`, `retention_vs_mileage.png`,
`retention_by_model.png`

---

## 4. Feature engineering (`src/feature_engineering.py`)

- **Engineered features:** `vehicle_age` (year listed - model year), `miles_per_year`, `branded_title`
- **From descriptions** (Craigslist only, % of listings that mention it):
  `desc_one_owner` 11.1%, `desc_clean_title` 34.3%, `desc_accident_free` 13.1%,
  `desc_needs_work` 7.5%, `desc_maintained` 16.0%, `desc_premium_count` (average 0.61)
- **One-hot encoding** for low-cardinality categories (make, model, body type, drivetrain, fuel,
  transmission, color, condition, seller type, source) with `drop_first=True`
- **High-cardinality feature: `state` (49 values)**
  - *Frequency encoding:* share of training cars from that state. Uses no target, so it can't leak.
  - *Target encoding:* smoothed mean value retention of training cars from that state,
    `(n * state_mean + 10 * overall_mean) / (n + 10)`. Computed on **training rows only**.
  - Why not one-hot: it would add 48 sparse columns, and many states have fewer than 10 cars (overfitting risk)
  - Test-set correlation with the target: target encoding 0.33 vs frequency encoding -0.13
  - Note: the target encoding partly stands in for the source. The low-retention states (IL, NY, GA, PA, TX)
    are the Craigslist cities, which have older cars.
- **Result:** 56 numeric features, no missing values, fixed 80/20 split (1,233 train / 308 test, seed 42)

---

## 5. Model results (`model_results.csv`, `model_comparison.png`, `predicted_vs_actual.png`)

| Model | CV R^2 (+/- std) | Test MAE | Test RMSE | Test R^2 | Test MAE in $ |
|---|---|---|---|---|---|
| Baseline (predict the average) | -0.001 | 0.190 | 0.229 | -0.005 | $5,621 |
| Linear Regression | 0.821 (0.026) | 0.077 | 0.105 | 0.788 | $2,309 |
| Random Forest | 0.803 (0.031) | 0.071 | **0.096** | **0.823** | $2,124 |
| **Gradient Boosting** | **0.837** (0.028) | **0.069** | 0.097 | 0.819 | **$2,075** |

- The best model was **chosen by 5-fold cross-validation** (Gradient Boosting). The test set was only used to report the final score.
- An MAE of 0.069 means predictions are off by about 6.9% of MSRP (about $2,075) on average.
- Random Forest and Gradient Boosting are statistically tied on the test set (a 0.004 R^2 difference, smaller than the CV std).
- Linear Regression is close behind, so the relationship is mostly linear in age and mileage.
  The tree models gain a little from curves (depreciation slows down for older cars).

---

## 6. Feature ranking and depreciation (`feature_analysis.txt`)

**Grouped permutation importance** (Gradient Boosting, test set, drop in R^2 when shuffled), `feature_ranking.png`:

| Rank | Feature | R^2 drop |
|---|---|---|
| 1 | mileage | 0.628 |
| 2 | vehicle_age | 0.233 |
| 3 | model | 0.104 |
| 4 | make | 0.050 |
| 5 | msrp | 0.022 |
| 6 | body_type | 0.017 |
| 7 | seller_type | 0.017 |
| 8 | state_target_enc | 0.014 |
| 9 | desc_premium_count | 0.014 |
| 10 | source | 0.011 |
| ... | description phrases, color | about 0 |

**Linear regression in real units** (holding everything else the same), `linear_coefficients.png`:
- +1 year of age -> **-0.023** (about 2.3% of MSRP per year)
- +10,000 miles -> **-0.019** (about 1.9% of MSRP)
- Branded title (rebuilt/salvage/lien) -> **-0.103** (10% of MSRP, but only about 3% of cars have one)
- Private owner vs dealer -> **-0.036**
- +$1,000 original MSRP -> -0.004 (more expensive trims lose value slightly faster)

**Depreciation curves**, `depreciation_curves.png` (model, everything else held the same):
- Mileage: 0.75 at 0 miles -> 0.51 at 100k -> 0.31 at 200k, then levels off
- Age: steep drop in years 0-2, flatter in years 3-6, then a steady drop again

**By make**, `depreciation_by_make.png` (real medians):

| Age (years) | Toyota | Honda | Chevrolet | Ford |
|---|---|---|---|---|
| 0-1 | 0.86 | 0.91 | 0.79 | 0.76 |
| 4-5 | 0.73 | 0.77 | 0.54 | 0.54 |
| 8-9 | 0.55 | 0.57 | 0.41 | 0.37 |
| 12-13 | 0.40 | 0.41 | 0.19 | 0.23 |

- **By model:** median retention, highest to lowest: Corolla 0.68, Civic 0.63, RAV4 0.62, Silverado 0.57,
  CR-V 0.55, Camry 0.54, Accord 0.53, F-150 0.49, Malibu 0.42, Explorer 0.38, Escape 0.37, Equinox 0.34

---

## 7. Discussion points

### Most important factors affecting value
1. **Mileage** is the single strongest predictor. It's more precise than age: two 8-year-old cars can have 40k or 160k miles.
2. **Age** is second. Mileage and age are correlated (older cars have more miles), so when both are in the
   model, mileage gets most of the credit.
3. **Model and make:** Toyota and Honda keep roughly 15-20 more percentage points of MSRP than Ford and Chevrolet at the same age.
4. **Smaller effects:** original MSRP, body type, seller type, and title status (big per car, but rare).

### Major drivers of depreciation
- Using up the car (miles) and time (age) drive almost all of the depreciation
- Brand reputation for reliability: the Toyota/Honda vs Ford/Chevy gap appears in the first 2-3 years and stays
- Sedans and SUVs from domestic brands (Malibu, Equinox, Escape, Explorer) lose value fastest; trucks and Japanese compacts the slowest
- Damage history (a branded title) costs about 10% of MSRP on its own
- Private-sale cars sell for less than dealer cars (no dealer markup or reconditioning)

### Challenges in data collection and integration
- **Bot blocking:** 11 of 14 listing sites tested blocked automated requests (403 errors, "Just a moment..." bot-check pages):
  Cars.com, CarGurus, CarMax, Edmunds, Autotrader, TrueCar, Carfax, AutoNation, eBay, CarsDirect, EchoPark.
  Facebook Marketplace requires a login.
- **robots.txt vs terms of use:** Craigslist, Autolist, and J.D. Power allow these pages in robots.txt but forbid
  scraping in their terms of use. We kept the volume small and polite for this class project.
- **Hidden data formats:** Autolist loads its data from a JSON API, and J.D. Power puts its prices inside escaped JavaScript data,
  so we had to look at the raw responses instead of the visible HTML.
- **Unreliable sources:** J.D. Power sometimes returned random 403s (we needed retry logic and a second pass);
  MotorTrend quietly showed 2025 prices on its 2005 pages, so we didn't use it.
- **Integration:** different column names, units, and category labels across sources; trim names that don't
  match between listings and MSRP data; free-text model names on Craigslist ("Chevy", "crv", just "Toyota")
- **The sources don't cover the same things:** only Craigslist has descriptions, condition, and title status; only Autolist has MPG and interior color

### Limitations of the dataset
- **Size and scope:** 1,541 cars, 12 popular models, 2008+ only (no MSRP before 2008), 10 Craigslist cities
- **Asking price, not sale price:** listing prices are what sellers *hope* to get
- **Snapshot in time:** all data is from October 2026, so it says nothing about market changes over time
- **MSRP isn't adjusted for inflation:** older cars' ratios compare today's dollars to old dollars, which makes old cars look like they kept slightly more value than they really did
- **MSRP matching:** 29% of cars used the model-year median MSRP because the trim was missing or didn't match;
  special versions (GR Corolla, RAV4 Prime, Civic Type R) were compared to the regular model's MSRP, giving ratios above 1
- **Source differences:** Craigslist cars are older and cheaper, so "source" and "state" partly measure the type of car
- **Lots of missing values:** condition (69%) and title status (54%) are missing, and descriptions only exist for Craigslist
- **Possible bad prices:** some very cheap Craigslist listings may be down-payment ads or scams

### Limitations of the models
- **Description features added almost nothing** (R^2 drop of 0.001 or less): they only exist for 47% of rows, the phrases are rare,
  and other columns (title status, seller type) already cover the same information. Simple keyword matching
  also misses things like negation ("not one owner") and different wording.
  - **Missing is treated as "not mentioned":** Autolist has no descriptions, so all 821 Autolist cars get 0 for every
    description feature. A `has_description` flag would be exactly the same as `source_craigslist` (821/720 split),
    which the models already use, so the models can tell "no text" apart from "text that doesn't mention it."
  - In the linear model, `source_craigslist` absorbs the overall difference between the sources, so the description
    coefficients are effectively learned from Craigslist cars only. The "0 means no text" problem mostly affects the
    **permutation importance ranking** (shuffling these features mixes in Autolist's all-zero rows), not the model itself.
  - **Fairer check on Craigslist rows only** (720 cars, all with descriptions, 5-fold CV): adding the description
    features raises R^2 from 0.697 to 0.702 (Linear Regression) and from 0.686 to 0.695 (Gradient Boosting).
  - **Fold-by-fold check** (same folds with and without the features, 3 shuffles x 5 folds = 15 comparisons):
    the version with description features won 12/15 folds (Linear, mean gain +0.004, std 0.008) and 11/15
    (Gradient Boosting, mean gain +0.009, std 0.016). The gain leans positive but is **small and within the CV
    variation**, so it's not a consistent improvement.
- **Mileage and age are correlated**, so their separate effects are hard to split; linear coefficients for correlated features can be unstable
- **Small leak in cross-validation:** state target encoding was computed on all training rows, so CV scores are slightly optimistic
  (the test-set scores don't have this problem)
- **No hyperparameter tuning:** fixed, reasonable settings were used
- **Redundant columns in linear regression:** the make columns are exact sums of the model columns, and mileage
  (up to ~300,000) is on a very different scale from the 0/1 columns. Without StandardScaler, the solver hits
  rounding problems and test R^2 drops from 0.788 to 0.785, so the linear model keeps the scaler.
- **The biggest errors** are special versions with ratios above 1 and very cheap Craigslist listings

### Possible future work
- More sources and years (and other countries), plus inflation-adjusted MSRP
- Better trim matching (or decoding the VIN to get the exact trim)
- Better text features (TF-IDF, handling negation) and the condition information in photos
- Hyperparameter tuning (grid search) and target encoding done inside each CV fold
- Collect sold prices instead of asking prices

---

## 8. Conclusion (key takeaways)
- A Gradient Boosting model predicts value retention with **R^2 of about 0.82** and an average error of **about 7% of MSRP (about $2,100)**,
  compared to 19% for a predict-the-average baseline.
- **Mileage and age** explain most of the depreciation (about 2% of MSRP per 10k miles and about 2.3% per year, holding the other fixed).
- **Brand matters:** Toyota and Honda keep about 15-20 more percentage points of MSRP than Ford and Chevrolet at the same age.
- **Description keywords barely helped** (even on Craigslist cars alone, the gain was within the CV variation), which suggests simple keyword matching isn't enough to get value out of seller descriptions.

---

## Files
| Step | Script | Output |
|---|---|---|
| Scrape Autolist | `src/scrape_autolist.py` | `data/raw/autolist_raw.csv` |
| Scrape Craigslist | `src/scrape_craigslist.py` | `data/raw/craigslist_raw.csv` |
| Collect MSRP | `src/scrape_msrp.py` | `data/raw/msrp_raw.csv` |
| Clean + merge | `src/clean_merge.py` | `data/processed/cars_clean.csv` |
| Explore | `src/explore_data.py` | `data/processed/summary_stats.txt`, 8 charts |
| Features | `src/feature_engineering.py` | `data/processed/cars_features.csv` |
| Models | `src/train_models.py` | `data/processed/model_results.csv`, 2 charts |
| Analysis | `src/analyze_features.py` | `data/processed/feature_analysis.txt`, 4 charts |
