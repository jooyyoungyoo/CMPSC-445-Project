# Step 9: Which features matter most, and how do cars depreciate?
#
# Three ways of looking at it:
#   1. Linear regression coefficients in real units (per year, per 10k miles, ...)
#   2. Grouped permutation importance on the test set: shuffle a feature and
#      see how much worse the model gets. One-hot columns that belong together
#      (like all the model_ columns) are shuffled together as one group.
#   3. Depreciation curves: predicted value retention as age or mileage changes,
#      plus the real median value retention by age for each make
#
# Run from the project folder:
#     python src/analyze_features.py

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts to files without opening windows
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import config
import train_models   # reuse the same data loading and model settings as Step 8

CLEAN_FILE = os.path.join(config.PROJECT_FOLDER, "data", "processed", "cars_clean.csv")
SUMMARY_FILE = os.path.join(config.PROJECT_FOLDER, "data", "processed", "feature_analysis.txt")
PLOTS_FOLDER = os.path.join(config.PROJECT_FOLDER, "plots")

RANDOM_SEED = 42
PERMUTATION_REPEATS = 10
TOP_N = 15

# one-hot columns start with these, e.g. "model_Camry" belongs to the "model" group
ONE_HOT_GROUPS = ["make", "model", "body_type", "drivetrain", "fuel_type", "transmission",
                  "exterior_color", "condition", "seller_type", "source"]

# yes/no features to show in the coefficient chart
YES_NO_FEATURES = ["desc_one_owner", "desc_clean_title", "desc_accident_free", "desc_needs_work",
                   "desc_maintained", "desc_premium_count", "branded_title", "seller_type_owner",
                   "source_craigslist"]

# Every line of the summary goes here so we can also save it to a file
summary_lines = []


def log(text=""):
    print(text)
    summary_lines.append(str(text))


def save_chart(figure, file_name):
    path = os.path.join(PLOTS_FOLDER, file_name)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print("    saved", path)


def group_name(column):
    # "model_Camry" -> "model", "vehicle_age" -> "vehicle_age"
    for group in ONE_HOT_GROUPS:
        if column.startswith(group + "_"):
            return group
    return column


# ---------------------------------------------------------------
# 1. Linear regression coefficients in real units
# ---------------------------------------------------------------

def linear_coefficients(X_train, y_train, feature_columns):
    # Fit a plain linear regression on the raw features so each
    # coefficient is in real units (per year, per mile, per dollar, ...)
    plain_model = LinearRegression()
    plain_model.fit(X_train, y_train)
    coefficients = pd.Series(plain_model.coef_, index=feature_columns)

    # scale a few of them to units that are easier to talk about
    effects = {}
    effects["+1 year of age"] = coefficients["vehicle_age"]
    effects["+10,000 miles"] = coefficients["mileage"] * 10000
    effects["+1,000 miles/year"] = coefficients["miles_per_year"] * 1000
    effects["+$1,000 of MSRP"] = coefficients["msrp"] * 1000
    for column in YES_NO_FEATURES:
        if column == "desc_premium_count":
            effects[column + " (+1 feature)"] = coefficients[column]   # it's a 0-3 count
        else:
            effects[column + " (yes vs no)"] = coefficients[column]

    log("=" * 70)
    log("1. LINEAR REGRESSION COEFFICIENTS (real units)")
    log("=" * 70)
    log("Change in value retention, holding all other features the same:")
    for name in effects:
        log(f"  {name}: {round(effects[name], 4)}")
    log()
    return effects


def plot_linear_coefficients(effects):
    names = list(effects.keys())
    values = list(effects.values())

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(names, values, color="steelblue")
    axis.axvline(0, color="black", linewidth=1)
    axis.invert_yaxis()  # keep the same order as the list, top to bottom
    axis.set_title("Linear regression: effect of each feature on value retention")
    axis.set_xlabel("Change in value retention (negative = loses value)")
    save_chart(figure, "linear_coefficients.png")


# ---------------------------------------------------------------
# 2. Grouped permutation importance (the main ranking)
# ---------------------------------------------------------------

def grouped_permutation_importance(model, X_test, y_test, feature_columns):
    # put the columns into groups
    groups = {}
    for column in feature_columns:
        name = group_name(column)
        if name not in groups:
            groups[name] = []
        groups[name].append(column)

    starting_r2 = r2_score(y_test, model.predict(X_test))
    random_generator = np.random.RandomState(RANDOM_SEED)

    results = []
    for name in groups:
        drops = []
        for repeat in range(PERMUTATION_REPEATS):
            shuffled = X_test.copy()
            new_order = random_generator.permutation(len(X_test))
            # shuffle every column in the group with the SAME new order
            for column in groups[name]:
                shuffled[column] = X_test[column].values[new_order]
            new_r2 = r2_score(y_test, model.predict(shuffled))
            drops.append(starting_r2 - new_r2)
        results.append({
            "feature_group": name,
            "columns": len(groups[name]),
            "r2_drop_mean": np.mean(drops),
            "r2_drop_std": np.std(drops),
        })

    results = pd.DataFrame(results)
    results = results.sort_values("r2_drop_mean", ascending=False)

    log("=" * 70)
    log("2. GROUPED PERMUTATION IMPORTANCE (Gradient Boosting, test set)")
    log("=" * 70)
    log(f"Test R^2 before shuffling: {round(starting_r2, 3)}")
    log("How much R^2 drops when each feature group is shuffled:")
    log(results.round(4).to_string(index=False))
    log()
    return results


def plot_permutation_importance(results):
    top = results.head(TOP_N)

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(top["feature_group"], top["r2_drop_mean"], color="steelblue")
    axis.invert_yaxis()  # most important at the top
    axis.set_title("Feature ranking: drop in test R² when shuffled (Gradient Boosting)")
    axis.set_xlabel("Drop in R² (bigger = more important)")
    save_chart(figure, "feature_ranking.png")


# ---------------------------------------------------------------
# 3. Depreciation curves
# ---------------------------------------------------------------

def average_prediction(model, X_test, column, value):
    # set one feature to the same value for every test car and average the predictions
    changed = X_test.copy()
    changed[column] = value
    # miles_per_year depends on age and mileage, so update it too
    # (clip(lower=1) uses at least 1 year, same as in feature_engineering.py)
    years_for_division = changed["vehicle_age"].clip(lower=1)
    changed["miles_per_year"] = changed["mileage"] / years_for_division
    predictions = model.predict(changed)
    return predictions.mean()


def depreciation_curves(model, X_test):
    ages = list(range(0, 16))
    age_predictions = []
    for age in ages:
        age_predictions.append(average_prediction(model, X_test, "vehicle_age", age))

    mileages = list(range(0, 250001, 10000))
    mileage_predictions = []
    for miles in mileages:
        mileage_predictions.append(average_prediction(model, X_test, "mileage", miles))

    log("=" * 70)
    log("3. DEPRECIATION CURVES (Gradient Boosting, everything else held the same)")
    log("=" * 70)
    log("Predicted value retention by vehicle age:")
    for i in range(len(ages)):
        line = f"  age {ages[i]}: {round(age_predictions[i], 3)}"
        if i > 0:
            change = age_predictions[i] - age_predictions[i - 1]
            line = line + f"   ({round(change, 3)} from the year before)"
        log(line)
    log()
    log("Predicted value retention by mileage:")
    for i in range(0, len(mileages), 5):
        log(f"  {mileages[i]} miles: {round(mileage_predictions[i], 3)}")
    log()
    return ages, age_predictions, mileages, mileage_predictions


def plot_depreciation_curves(ages, age_predictions, mileages, mileage_predictions):
    # two separate panels since age and mileage have different x scales
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    axes[0].plot(ages, age_predictions, color="steelblue", marker="o")
    axes[0].set_title("Predicted value retention vs. age")
    axes[0].set_xlabel("Vehicle age (years)")
    axes[0].set_ylabel("Predicted value retention")

    axes[1].plot(mileages, mileage_predictions, color="steelblue")
    axes[1].set_title("Predicted value retention vs. mileage")
    axes[1].set_xlabel("Mileage (miles)")

    figure.suptitle("Depreciation curves (Gradient Boosting, all other features held the same)")
    save_chart(figure, "depreciation_curves.png")


def depreciation_by_make():
    # real data, not the model: median value retention by age for each make
    data = pd.read_csv(CLEAN_FILE)
    listed_dates = pd.to_datetime(data["listed_date"])
    ages = listed_dates.dt.year - data["year"]
    data["vehicle_age"] = ages.clip(lower=0)   # no negative ages, same as feature_engineering.py

    # group ages into 2-year buckets so each point has enough cars.
    # "//" is whole-number division: ages 0-1 -> 0, ages 2-3 -> 2, ages 4-5 -> 4, ...
    data["age_bucket"] = (data["vehicle_age"] // 2) * 2
    buckets = sorted(data["age_bucket"].unique())
    makes = ["Toyota", "Honda", "Chevrolet", "Ford"]

    log("=" * 70)
    log("4. MEDIAN VALUE RETENTION BY AGE AND MAKE (real data)")
    log("=" * 70)
    log("age bucket   " + "   ".join(makes))

    figure, axis = plt.subplots(figsize=(9, 5))
    # one line per make; skip buckets with fewer than 5 cars (too noisy)
    plot_points = {}
    for make in makes:
        plot_points[make] = ([], [])

    for bucket in buckets:
        line = f"  {bucket}-{bucket + 1}:"
        for make in makes:
            rows = data[(data["age_bucket"] == bucket) & (data["make"] == make)]
            median_retention = rows["value_retention"].median()
            line = line + f"   {round(median_retention, 2)}"
            if len(rows) >= 5:
                plot_points[make][0].append(bucket + 0.5)   # middle of the 2-year bucket
                plot_points[make][1].append(median_retention)
        log(line)
    log()

    for make in makes:
        axis.plot(plot_points[make][0], plot_points[make][1], marker="o", label=make)

    axis.set_title("Median value retention by age and make (real listings)")
    axis.set_xlabel("Vehicle age (years, 2-year groups)")
    axis.set_ylabel("Median value retention")
    axis.legend()
    save_chart(figure, "depreciation_by_make.png")


def main():
    os.makedirs(PLOTS_FOLDER, exist_ok=True)

    X_train, y_train, X_test, y_test, feature_columns = train_models.load_data()
    models = train_models.make_models()

    print("Training Gradient Boosting (same settings as Step 8)...")
    gradient_boosting = models["Gradient Boosting"]
    gradient_boosting.fit(X_train, y_train)
    print()

    effects = linear_coefficients(X_train, y_train, feature_columns)
    permutation_results = grouped_permutation_importance(gradient_boosting, X_test, y_test, feature_columns)
    ages, age_predictions, mileages, mileage_predictions = depreciation_curves(gradient_boosting, X_test)

    print("Making charts...")
    plot_linear_coefficients(effects)
    plot_permutation_importance(permutation_results)
    plot_depreciation_curves(ages, age_predictions, mileages, mileage_predictions)
    depreciation_by_make()

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print("Saved summary to", SUMMARY_FILE)
    print("Done!")


if __name__ == "__main__":
    main()
