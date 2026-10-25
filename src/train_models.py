# Step 8: Train and compare 3 regression models.
#
#   1. Linear Regression   - one coefficient per feature, easy to explain
#   2. Random Forest       - averages many decision trees
#   3. Gradient Boosting   - adds trees one at a time, each fixing the last one's mistakes
#
# Each model is checked two ways:
#   - 5-fold cross-validation on the training rows (fair comparison)
#   - final score on the test rows the models never saw
# using MAE, RMSE, and R^2.
#
# Run from the project folder:
#     python src/train_models.py

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts to files without opening windows
import matplotlib.pyplot as plt

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config

FEATURES_FILE = os.path.join(config.PROJECT_FOLDER, "data", "processed", "cars_features.csv")
RESULTS_FILE = os.path.join(config.PROJECT_FOLDER, "data", "processed", "model_results.csv")
PREDICTIONS_FILE = os.path.join(config.PROJECT_FOLDER, "data", "processed", "test_predictions.csv")
PLOTS_FOLDER = os.path.join(config.PROJECT_FOLDER, "plots")

RANDOM_SEED = 42


def make_models():
    models = {}

    # The baseline just predicts the average every time. A real model has to beat it.
    models["Baseline (average)"] = DummyRegressor(strategy="mean")

    # StandardScaler puts every feature on the same scale before fitting.
    # Mathematically it shouldn't change the predictions, but here it does a little:
    # mileage goes up to ~300,000 while most features are 0/1, and the make columns
    # are exact sums of the model columns. Without scaling, the solver hits rounding
    # problems and gets a slightly worse fit (test R^2 0.785 instead of 0.788).
    models["Linear Regression"] = make_pipeline(StandardScaler(), LinearRegression())

    models["Random Forest"] = RandomForestRegressor(
        n_estimators=300,        # number of trees
        min_samples_leaf=2,      # each leaf needs at least 2 cars (less overfitting)
        random_state=RANDOM_SEED,
        n_jobs=-1,               # use all CPU cores
    )

    models["Gradient Boosting"] = GradientBoostingRegressor(
        n_estimators=300,        # number of trees added one after another
        learning_rate=0.05,      # how much each new tree is allowed to change things
        max_depth=3,             # small trees
        random_state=RANDOM_SEED,
    )
    return models


def load_data():
    data = pd.read_csv(FEATURES_FILE)
    train = data[data["split"] == "train"]
    test = data[data["split"] == "test"]

    # everything except the split and the target is a feature
    feature_columns = []
    for column in data.columns:
        if column != "split" and column != "value_retention":
            feature_columns.append(column)

    X_train = train[feature_columns]
    y_train = train["value_retention"]
    X_test = test[feature_columns]
    y_test = test["value_retention"]
    return X_train, y_train, X_test, y_test, feature_columns


def cross_validation_scores(model, X_train, y_train):
    # split the training rows into 5 parts; train on 4, score on the 5th, repeat 5 times
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_validate(model, X_train, y_train, cv=folds,
                            scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"])

    # sklearn gives errors as negative numbers (so "bigger is better"), so flip the sign
    cv_mae = -scores["test_neg_mean_absolute_error"].mean()
    cv_rmse = -scores["test_neg_root_mean_squared_error"].mean()
    cv_r2 = scores["test_r2"].mean()
    cv_r2_std = scores["test_r2"].std()
    return cv_mae, cv_rmse, cv_r2, cv_r2_std


def test_scores(y_test, predictions, msrp_test):
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # turn the ratio error into dollars: (ratio error) x (that car's MSRP)
    dollar_errors = np.abs(predictions - y_test) * msrp_test
    mae_dollars = dollar_errors.mean()
    return mae, rmse, r2, mae_dollars


# ---------------------------------------------------------------
# Charts
# ---------------------------------------------------------------

def plot_model_comparison(results):
    # one small chart per metric, since they have different scales
    metrics = ["test_MAE", "test_RMSE", "test_R2"]
    titles = ["MAE (lower is better)", "RMSE (lower is better)", "R² (higher is better)"]
    model_names = list(results["model"])

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for i in range(len(metrics)):
        axis = axes[i]
        values = list(results[metrics[i]])
        axis.barh(model_names, values, color="steelblue")

        # print the value next to each bar - the models are too close to tell apart otherwise
        for j in range(len(values)):
            axis.text(max(values[j], 0), j, " " + str(round(values[j], 3)), va="center")

        axis.set_title(titles[i])
        axis.set_xlim(0, max(values) * 1.25)

    axes[0].invert_yaxis()  # first model at the top (shared by all 3 charts)
    figure.suptitle("Model comparison on the test set")
    figure.tight_layout()
    path = os.path.join(PLOTS_FOLDER, "model_comparison.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print("    saved", path)


def plot_predicted_vs_actual(y_test, predictions, model_name):
    figure, axis = plt.subplots(figsize=(6.5, 6))
    axis.scatter(y_test, predictions, s=14, alpha=0.5, color="steelblue", label="test cars")

    # perfect predictions would land on this diagonal line
    biggest = max(y_test.max(), predictions.max()) + 0.05
    axis.plot([0, biggest], [0, biggest], color="black", linestyle="--", label="perfect prediction")

    axis.set_xlim(0, biggest)
    axis.set_ylim(0, biggest)
    axis.set_title("Predicted vs. actual value retention\n(" + model_name + ", test set)")
    axis.set_xlabel("Actual value retention")
    axis.set_ylabel("Predicted value retention")
    axis.legend(loc="upper left")
    figure.tight_layout()
    path = os.path.join(PLOTS_FOLDER, "predicted_vs_actual.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print("    saved", path)


def main():
    X_train, y_train, X_test, y_test, feature_columns = load_data()
    print("Training rows:", len(X_train), "  Test rows:", len(X_test), "  Features:", len(feature_columns))
    print()

    models = make_models()
    results = []
    all_predictions = {}

    for name in models:
        model = models[name]
        print("Training", name, "...")

        cv_mae, cv_rmse, cv_r2, cv_r2_std = cross_validation_scores(model, X_train, y_train)

        # now train on all the training rows and score on the test rows
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        all_predictions[name] = predictions
        mae, rmse, r2, mae_dollars = test_scores(y_test, predictions, X_test["msrp"])

        results.append({
            "model": name,
            "cv_MAE": cv_mae,
            "cv_RMSE": cv_rmse,
            "cv_R2": cv_r2,
            "cv_R2_std": cv_r2_std,
            "test_MAE": mae,
            "test_RMSE": rmse,
            "test_R2": r2,
            "test_MAE_dollars": mae_dollars,
        })

    # Pick the best model by its CROSS-VALIDATION R^2, not the test R^2.
    # The test set should only be used to report the final score, never to make choices.
    best_result = None
    for result in results:
        if result["model"] == "Baseline (average)":
            continue
        if best_result is None or result["cv_R2"] > best_result["cv_R2"]:
            best_result = result
    best_name = best_result["model"]

    results_table = pd.DataFrame(results)
    results_table.to_csv(RESULTS_FILE, index=False)

    print()
    print("=" * 90)
    print("RESULTS (value retention is a ratio, so an MAE of 0.07 means 7% of MSRP)")
    print("=" * 90)
    print(results_table.round(3).to_string(index=False))
    print()
    print("Saved results to", RESULTS_FILE)
    print(f"Best model (by cross-validation): {best_name}  "
          f"cv R^2 = {round(best_result['cv_R2'], 3)}  test R^2 = {round(best_result['test_R2'], 3)}")

    # save test predictions for every model (used for error analysis)
    prediction_table = pd.DataFrame({"actual": y_test.values})
    for name in all_predictions:
        prediction_table[name] = all_predictions[name]
    prediction_table.to_csv(PREDICTIONS_FILE, index=False)

    print()
    print("Making charts...")
    plot_model_comparison(results_table)
    plot_predicted_vs_actual(y_test, all_predictions[best_name], best_name)
    print("Done!")


if __name__ == "__main__":
    main()
