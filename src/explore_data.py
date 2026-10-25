# Step 6: Explore the cleaned dataset and make charts of the data.
#
# Prints summary statistics (and saves them to a text file for the report),
# then saves charts to the plots/ folder.
#
# Run from the project folder:
#     python src/explore_data.py

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts to files without opening windows
import matplotlib.pyplot as plt

import config

DATA_FILE = os.path.join(config.PROJECT_FOLDER, "data", "processed", "cars_clean.csv")
PLOTS_FOLDER = os.path.join(config.PROJECT_FOLDER, "plots")
STATS_FILE = os.path.join(config.PROJECT_FOLDER, "data", "processed", "summary_stats.txt")

NUMERIC_COLUMNS = ["price", "msrp", "value_retention", "year", "mileage", "cylinders"]
CATEGORY_COLUMNS = ["make", "model", "body_type", "transmission", "drivetrain",
                    "fuel_type", "exterior_color", "condition", "title_status",
                    "seller_type", "state"]

# Every line of the stats goes here so we can also save it to a file
stats_lines = []


def log(text=""):
    print(text)
    stats_lines.append(str(text))


def save_chart(figure, file_name):
    path = os.path.join(PLOTS_FOLDER, file_name)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print("    saved", path)


# ---------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------

def print_statistics(data):
    log("=" * 60)
    log("DATASET OVERVIEW")
    log("=" * 60)
    log(f"Rows: {len(data)}   Columns: {len(data.columns)}")
    log()
    log("Rows per source:")
    log(data["source"].value_counts().to_string())
    log()

    log("Numeric columns:")
    numeric_summary = data[NUMERIC_COLUMNS].describe()
    log(numeric_summary.round(2).to_string())
    log()

    log("Median by source:")
    medians_by_source = data.groupby("source")[["price", "mileage", "year", "value_retention"]].median()
    log(medians_by_source.round(2).to_string())
    log()

    log("Missing values (count and percent):")
    for column in data.columns:
        missing_count = data[column].isna().sum()
        if missing_count > 0:
            percent = round(100 * missing_count / len(data), 1)
            log(f"  {column}: {missing_count} ({percent}%)")
    log()

    for column in CATEGORY_COLUMNS:
        log(f"{column} ({data[column].nunique()} unique values):")
        log(data[column].value_counts().to_string())
        log()

    log("Correlation of numeric columns with value_retention:")
    correlation_table = data[NUMERIC_COLUMNS].corr()
    log(correlation_table["value_retention"].round(3).to_string())
    log()

    log("Median value retention by make:")
    medians_by_make = data.groupby("make")["value_retention"].median()
    medians_by_make = medians_by_make.sort_values(ascending=False)
    log(medians_by_make.round(3).to_string())
    log()


# ---------------------------------------------------------------
# Charts
# ---------------------------------------------------------------

def plot_histogram(values, title, x_label, file_name):
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(values, bins=40, color="steelblue", edgecolor="white")

    # dashed line at the median
    median_value = round(values.median(), 2)
    axis.axvline(median_value, color="black", linestyle="--", label=f"median = {median_value}")

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Number of listings")
    axis.legend()
    save_chart(figure, file_name)


def plot_model_counts(data):
    # how many listings of each model, split by source (stacked bars)
    labels = []
    autolist_counts = []
    craigslist_counts = []
    for make in config.MAKES_AND_MODELS:
        for model in config.MAKES_AND_MODELS[make]:
            rows = data[data["model"] == model]
            labels.append(make + " " + model)
            autolist_counts.append(len(rows[rows["source"] == "autolist"]))
            craigslist_counts.append(len(rows[rows["source"] == "craigslist"]))

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(labels, autolist_counts, color="steelblue", label="Autolist")
    # Craigslist bars start where the Autolist bars end
    axis.barh(labels, craigslist_counts, left=autolist_counts, color="orange", label="Craigslist")
    axis.invert_yaxis()  # first model at the top
    axis.set_xlim(0, 200)  # leave room on the right for the legend
    axis.set_title("Listings per make and model")
    axis.set_xlabel("Number of listings")
    axis.legend(loc="lower right")
    save_chart(figure, "make_model_counts.png")


def plot_year_counts(data):
    years = sorted(data["year"].unique())
    autolist_counts = []
    craigslist_counts = []
    for year in years:
        rows = data[data["year"] == year]
        autolist_counts.append(len(rows[rows["source"] == "autolist"]))
        craigslist_counts.append(len(rows[rows["source"] == "craigslist"]))

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(years, autolist_counts, color="steelblue", label="Autolist")
    # Craigslist bars sit on top of the Autolist bars
    axis.bar(years, craigslist_counts, bottom=autolist_counts, color="orange", label="Craigslist")
    axis.set_xticks(years)
    axis.tick_params(axis="x", rotation=45)
    axis.set_title("Listings per model year")
    axis.set_xlabel("Model year")
    axis.set_ylabel("Number of listings")
    axis.legend(loc="upper left")
    save_chart(figure, "year_counts.png")


def plot_retention_by_year(data):
    # one box per model year so we can see depreciation over time
    years = sorted(data["year"].unique())
    groups = []
    for year in years:
        groups.append(data[data["year"] == year]["value_retention"])

    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.boxplot(groups, tick_labels=years)
    axis.tick_params(axis="x", rotation=45)
    axis.set_title("Value retention by model year")
    axis.set_xlabel("Model year")
    axis.set_ylabel("Value retention (price / MSRP)")
    save_chart(figure, "retention_by_year.png")


def plot_retention_vs_mileage(data):
    autolist_rows = data[data["source"] == "autolist"]
    craigslist_rows = data[data["source"] == "craigslist"]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(autolist_rows["mileage"], autolist_rows["value_retention"], s=12, alpha=0.5,
                 color="steelblue", label="Autolist")
    axis.scatter(craigslist_rows["mileage"], craigslist_rows["value_retention"], s=12, alpha=0.5,
                 color="orange", label="Craigslist")
    axis.set_title("Value retention vs. mileage")
    axis.set_xlabel("Mileage (miles)")
    axis.set_ylabel("Value retention (price / MSRP)")
    axis.legend()
    save_chart(figure, "retention_vs_mileage.png")


def plot_retention_by_model(data):
    medians = data.groupby("model")["value_retention"].median()
    medians = medians.sort_values()  # smallest first, so the biggest bar ends up on top

    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.barh(medians.index, medians.values, color="steelblue")
    axis.set_title("Median value retention by model")
    axis.set_xlabel("Median value retention (price / MSRP)")
    save_chart(figure, "retention_by_model.png")


def main():
    data = pd.read_csv(DATA_FILE)
    os.makedirs(PLOTS_FOLDER, exist_ok=True)

    print_statistics(data)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines))
    print("Saved statistics to", STATS_FILE)

    print("Making charts...")
    plot_histogram(data["price"], "Listing price distribution", "Listing price ($)",
                   "price_distribution.png")
    plot_histogram(data["mileage"], "Mileage distribution", "Mileage (miles)",
                   "mileage_distribution.png")
    plot_histogram(data["value_retention"], "Value retention distribution",
                   "Value retention (price / MSRP)", "retention_distribution.png")
    plot_model_counts(data)
    plot_year_counts(data)
    plot_retention_by_year(data)
    plot_retention_vs_mileage(data)
    plot_retention_by_model(data)
    print("Done!")


if __name__ == "__main__":
    main()
