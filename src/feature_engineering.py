# Step 7: Feature engineering.
#
# Turns the cleaned dataset into numbers a model can use:
#   - new numeric features (vehicle age, miles per year)
#   - yes/no features pulled out of the listing descriptions
#   - fills in missing values
#   - one-hot encoding for categories with only a few values
#   - frequency encoding and target encoding for "state" (49 values)
#
# Also decides the train/test split here (80/20), because target encoding
# must only learn from training rows. Otherwise the test answers would
# leak into the features.
#
# Run from the project folder:
#     python src/feature_engineering.py
#
# Output: data/processed/cars_features.csv

import os
import numpy as np
import pandas as pd

import config

INPUT_FILE = os.path.join(config.PROJECT_FOLDER, "data", "processed", "cars_clean.csv")
OUTPUT_FILE = os.path.join(config.PROJECT_FOLDER, "data", "processed", "cars_features.csv")

RANDOM_SEED = 42
TEST_FRACTION = 0.2

# How much to trust a state's own average in target encoding.
# A state with 10 cars gets half its own average and half the overall average.
SMOOTHING = 10

# Phrases to look for in the descriptions (all lowercase)
DESCRIPTION_FEATURES = {
    "desc_one_owner": ["one owner", "1 owner", "single owner", "1-owner", "one-owner"],
    "desc_clean_title": ["clean title", "title clean"],
    "desc_accident_free": ["no accident", "accident free", "accident-free", "never been in an accident",
                           "clean carfax", "carfax clean", "clean car fax", "clean autocheck"],
    "desc_needs_work": ["needs work", "needs repair", "mechanic special", "check engine",
                        "not running", "doesn't run", "does not run", "as-is", "as is"],
    "desc_maintained": ["service record", "maintenance record", "dealer maintained", "dealer serviced",
                        "well maintained", "regularly maintained", "new tires", "new tire"],
}

# Each of these adds 1 to desc_premium_count if mentioned
PREMIUM_FEATURES = {
    "leather": ["leather"],
    "sunroof": ["sunroof", "moonroof", "sun roof", "moon roof"],
    "camera": ["backup camera", "back up camera", "rear camera", "rearview camera"],
}

# Rare categories get grouped into "other" so we don't have tiny one-hot columns
COMMON_BODY_TYPES = ["suv", "sedan", "pickup", "hatchback", "coupe"]
COMMON_COLORS = ["white", "gray", "black", "silver", "blue", "red"]

# Columns that get one-hot encoded
ONE_HOT_COLUMNS = ["make", "model", "body_type", "drivetrain", "fuel_type", "transmission",
                   "exterior_color", "condition", "seller_type", "source"]


def mentions_any(text, phrases):
    # 1 if the text contains any of the phrases, otherwise 0
    if pd.isna(text):
        return 0
    text = str(text).lower()
    for phrase in phrases:
        if phrase in text:
            return 1
    return 0


def make_split_column(number_of_rows):
    # randomly mark 20% of rows as "test" and the rest as "train"
    random_generator = np.random.RandomState(RANDOM_SEED)
    shuffled_positions = random_generator.permutation(number_of_rows)
    test_count = int(number_of_rows * TEST_FRACTION)

    split = ["train"] * number_of_rows
    for i in range(test_count):
        split[shuffled_positions[i]] = "test"
    return split


def add_age_features(data):
    # age = year it was listed - model year (a 2026 car listed in 2026 is 0 years old)
    listed_dates = pd.to_datetime(data["listed_date"])
    listed_year = listed_dates.dt.year   # just the year part of each date
    ages = listed_year - data["year"]
    # clip(lower=0) turns any negative age into 0
    # (a 2027 model listed in 2026 would otherwise be -1 years old)
    data["vehicle_age"] = ages.clip(lower=0)

    # miles per year tells us if the car was driven a lot for its age.
    # clip(lower=1) uses at least 1 year, so brand-new cars don't divide by 0
    years_for_division = data["vehicle_age"].clip(lower=1)
    data["miles_per_year"] = data["mileage"] / years_for_division
    return data


def add_description_features(data):
    # Autolist has no descriptions, so its cars always get 0 here ("not mentioned").
    # We don't need a separate has_description column: it would be exactly the
    # same as source_craigslist, which the models already get.
    for feature_name in DESCRIPTION_FEATURES:
        phrases = DESCRIPTION_FEATURES[feature_name]
        values = []
        for text in data["description"]:
            values.append(mentions_any(text, phrases))
        data[feature_name] = values

    # count how many premium features are mentioned (0 to 3)
    counts = []
    for text in data["description"]:
        count = 0
        for feature in PREMIUM_FEATURES:
            count = count + mentions_any(text, PREMIUM_FEATURES[feature])
        counts.append(count)
    data["desc_premium_count"] = counts
    return data


def add_title_feature(data):
    # 1 if the title is rebuilt / salvage / has a lien, 0 if clean or not reported
    branded = []
    for status in data["title_status"]:
        if status == "rebuilt" or status == "salvage" or status == "lien":
            branded.append(1)
        else:
            branded.append(0)
    data["branded_title"] = branded
    return data


def fill_missing_values(data):
    # cylinders: use the most common number for that model
    for model in data["model"].unique():
        rows_for_model = data["model"] == model
        cylinders_for_model = data[rows_for_model]["cylinders"]
        most_common = cylinders_for_model.mode()[0]   # mode() = most common value
        missing_rows = rows_for_model & data["cylinders"].isna()
        data.loc[missing_rows, "cylinders"] = most_common

    # small number missing -> use the most common value
    data["fuel_type"] = data["fuel_type"].fillna("gas")
    data["transmission"] = data["transmission"].fillna("automatic")

    # lots missing -> keep "unknown" as its own category
    data["drivetrain"] = data["drivetrain"].fillna("unknown")
    data["condition"] = data["condition"].fillna("unknown")
    data["exterior_color"] = data["exterior_color"].fillna("unknown")
    data["body_type"] = data["body_type"].fillna("other")

    # group rare categories into "other"
    new_body_types = []
    for body_type in data["body_type"]:
        if body_type in COMMON_BODY_TYPES:
            new_body_types.append(body_type)
        else:
            new_body_types.append("other")
    data["body_type"] = new_body_types

    new_colors = []
    for color in data["exterior_color"]:
        if color in COMMON_COLORS or color == "unknown":
            new_colors.append(color)
        else:
            new_colors.append("other")
    data["exterior_color"] = new_colors
    return data


def add_state_encodings(data):
    train_rows = data[data["split"] == "train"]

    # Frequency encoding: replace each state with how common it is in the training data
    state_counts = train_rows["state"].value_counts()
    frequencies = []
    for state in data["state"]:
        if state in state_counts:
            frequencies.append(state_counts[state] / len(train_rows))
        else:
            frequencies.append(0)
    data["state_frequency"] = frequencies

    # Target encoding: replace each state with the average value retention of
    # training cars from that state, pulled toward the overall average when
    # a state only has a few cars (smoothing)
    overall_mean = train_rows["value_retention"].mean()
    state_means = train_rows.groupby("state")["value_retention"].mean()
    encoded = []
    for state in data["state"]:
        if state in state_counts:
            count = state_counts[state]
            state_mean = state_means[state]
            smoothed = (count * state_mean + SMOOTHING * overall_mean) / (count + SMOOTHING)
            encoded.append(smoothed)
        else:
            encoded.append(overall_mean)  # state never seen in training
    data["state_target_enc"] = encoded
    return data


def main():
    data = pd.read_csv(INPUT_FILE)
    print("Loaded", len(data), "rows")

    data["split"] = make_split_column(len(data))
    print("Train rows:", (data["split"] == "train").sum(), "  Test rows:", (data["split"] == "test").sum())

    data = add_age_features(data)
    data = add_description_features(data)
    data = add_title_feature(data)
    data = fill_missing_values(data)
    data = add_state_encodings(data)

    # One-hot encoding: each category becomes its own 0/1 column.
    # drop_first=True leaves one category out as the "baseline" so the
    # columns aren't redundant (matters for linear regression).
    one_hot = pd.get_dummies(data[ONE_HOT_COLUMNS], drop_first=True, dtype=int)

    numeric_features = [
        "vehicle_age", "mileage", "miles_per_year", "msrp", "cylinders",
        "desc_one_owner", "desc_clean_title", "desc_accident_free", "desc_needs_work",
        "desc_maintained", "desc_premium_count", "branded_title",
        "state_frequency", "state_target_enc",
    ]
    features = pd.concat([data[numeric_features], one_hot], axis=1)

    # final file: split + target + all the features
    output = pd.concat([data[["split", "value_retention"]], features], axis=1)
    output.to_csv(OUTPUT_FILE, index=False)

    print()
    print("Created", len(features.columns), "features:")
    for column in features.columns:
        print("   ", column)
    print()
    print("Saved to", OUTPUT_FILE)

    # ---- some numbers for the report ----
    print()
    print("Description features (Craigslist rows only, % of listings that mention it):")
    craigslist_rows = data[data["source"] == "craigslist"]
    for column in ["desc_one_owner", "desc_clean_title", "desc_accident_free",
                   "desc_needs_work", "desc_maintained"]:
        percent = round(100 * craigslist_rows[column].mean(), 1)
        print(f"  {column}: {percent}%")
    average_premium = round(craigslist_rows["desc_premium_count"].mean(), 2)
    print(f"  desc_premium_count: average {average_premium}")

    print()
    print("State encodings (top 5 and bottom 5 by target encoding):")
    # every car from the same state has the same encodings, so just take the first one
    state_table = data.groupby("state")[["state_frequency", "state_target_enc"]].first()
    train_rows = data[data["split"] == "train"]
    state_table["train_cars"] = train_rows["state"].value_counts()
    state_table = state_table.sort_values("state_target_enc", ascending=False)
    state_table = state_table.round(3)
    print(state_table.head(5).to_string())
    print("...")
    print(state_table.tail(5).to_string())

    print()
    print("Correlation with value_retention (test rows only, so no leakage):")
    test_rows = output[output["split"] == "test"]
    for column in ["vehicle_age", "mileage", "miles_per_year", "state_frequency", "state_target_enc"]:
        correlation = test_rows[column].corr(test_rows["value_retention"])
        print(f"  {column}: {round(correlation, 3)}")


if __name__ == "__main__":
    main()
