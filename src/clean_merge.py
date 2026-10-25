# Step 5: Clean both listing sources, merge them into one dataset,
# and attach the original MSRP so we can compute the value retention ratio.
#
#     value_retention = listing price / original MSRP
#
# Run from the project folder:
#     python src/clean_merge.py
#
# Output: data/processed/cars_clean.csv

import os
import pandas as pd

import config

RAW = config.RAW_DATA_FOLDER
OUTPUT_FILE = os.path.join(config.PROJECT_FOLDER, "data", "processed", "cars_clean.csv")

# Final column order for the merged dataset
FINAL_COLUMNS = [
    "source", "listing_id", "vin", "year", "make", "model", "trim",
    "price", "mileage", "body_type", "transmission", "drivetrain",
    "fuel_type", "cylinders", "exterior_color", "condition", "title_status",
    "seller_type", "state", "listed_date", "description",
    "msrp", "msrp_match", "value_retention",
]

# Craigslist gives a city "region", so turn it into a state like Autolist has
REGION_TO_STATE = {
    "newyork": "NY", "losangeles": "CA", "chicago": "IL", "houston": "TX",
    "dallas": "TX", "phoenix": "AZ", "philadelphia": "PA", "atlanta": "GA",
    "seattle": "WA", "denver": "CO",
}

# Different ways people write each model name (all lowercase)
MODEL_SPELLINGS = {
    "Camry": ["camry"],
    "Corolla": ["corolla"],
    "RAV4": ["rav4", "rav-4", "rav 4"],
    "Civic": ["civic"],
    "Accord": ["accord"],
    "CR-V": ["cr-v", "crv", "cr v"],
    "F-150": ["f-150", "f150", "f 150"],
    "Escape": ["escape"],
    "Explorer": ["explorer"],
    "Silverado 1500": ["silverado"],
    "Equinox": ["equinox"],
    "Malibu": ["malibu"],
}

# Words that mean it's actually a different vehicle
# (Corolla Cross is a different SUV, Silverado 2500/3500 are heavy-duty trucks)
WRONG_MODEL_WORDS = {
    "Corolla": ["cross"],
    "Accord": ["crosstour"],
    "Silverado 1500": ["2500", "3500", " hd"],
    "Explorer": ["sport trac"],
}

# Words to remove from Craigslist text when looking for the trim
MAKE_WORDS = ["toyota", "honda", "ford", "chevrolet", "chevy"]


# ---------------------------------------------------------------
# Small helper functions for cleaning single values
# ---------------------------------------------------------------

def is_missing(value):
    return pd.isna(value) or str(value).strip() == ""


def parse_money(text):
    # "$6,900" -> 6900
    if is_missing(text):
        return None
    cleaned = str(text).replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_cylinders(text):
    # Craigslist: "4 cylinders" -> 4.  Autolist: 4.0 -> 4
    if is_missing(text):
        return None
    first_part = str(text).split(" ")[0]
    try:
        return int(float(first_part))
    except ValueError:
        return None  # things like "other"


def clean_body_type(text):
    if is_missing(text):
        return None
    text = str(text).lower()
    if text == "truck":
        return "pickup"
    return text


def clean_drivetrain(text):
    # Craigslist only has fwd/rwd/4wd, so AWD and 4WD get combined
    if is_missing(text):
        return None
    text = str(text).lower()
    if text == "awd" or text == "4wd" or text == "4x4":
        return "awd/4wd"
    return text


def clean_fuel(text):
    if is_missing(text):
        return None
    text = str(text).lower()
    if text == "gasoline" or text == "flex fuel":
        return "gas"   # flex fuel cars run on regular gas too
    if text == "natural gas" or text == "electric":
        return "other"
    return text


def clean_color(text):
    if is_missing(text):
        return None
    text = str(text).lower()
    if text == "grey":
        return "gray"
    if text == "custom":
        return "other"
    return text


def clean_text(text):
    # lowercase and strip, or None if missing
    if is_missing(text):
        return None
    return str(text).lower().strip()


def is_right_model(model, text):
    # Check the listing text really mentions the model we searched for
    text = " " + text.lower() + " "
    found = False
    for spelling in MODEL_SPELLINGS[model]:
        if spelling in text:
            found = True
    if not found:
        return False
    if model in WRONG_MODEL_WORDS:
        for bad_word in WRONG_MODEL_WORDS[model]:
            if bad_word in text:
                return False
    return True


def guess_craigslist_trim(make_model_text, model):
    # "toyota camry le awd" -> "le awd"
    words = str(make_model_text).lower().split()
    leftover = []
    for word in words:
        if word in MAKE_WORDS:
            continue
        if word in MODEL_SPELLINGS[model] or word == "silverado" or word == "1500":
            continue
        if word.isdigit():
            continue  # a year like "2017"
        leftover.append(word)
    if len(leftover) == 0:
        return None
    return " ".join(leftover)


# ---------------------------------------------------------------
# Turn each source into the same columns
# ---------------------------------------------------------------

def standardize_autolist(autolist):
    rows = []
    for index, row in autolist.iterrows():
        new_row = {
            "source": "autolist",
            "listing_id": str(row["listing_id"]),
            "vin": row["vin"],
            "year": row["year"],
            "make": row["search_make"],
            "model": row["search_model"],
            "trim": row["trim"],
            "price": row["price"],
            "mileage": row["mileage"],
            "body_type": clean_body_type(row["body_type"]),
            "transmission": clean_text(row["transmission"]),
            "drivetrain": clean_drivetrain(row["drivetrain"]),
            "fuel_type": clean_fuel(row["fuel_type"]),
            "cylinders": parse_cylinders(row["cylinders"]),
            "exterior_color": clean_color(row["exterior_color"]),
            "condition": None,      # Autolist doesn't have this
            "title_status": None,   # or this
            "seller_type": "dealer",
            "state": row["state"],
            "listed_date": str(row["listed_date"])[:10],
            "description": None,    # or descriptions
        }
        rows.append(new_row)
    return pd.DataFrame(rows)


def standardize_craigslist(craigslist):
    rows = []
    for index, row in craigslist.iterrows():
        # use the listing price, or the search page price if it was missing
        price = parse_money(row["price"])
        if price is None:
            price = parse_money(row["search_price"])

        new_row = {
            "source": "craigslist",
            "listing_id": str(row["post_id"]),
            "vin": row["vin"],
            "year": row["year"],
            "make": row["search_make"],
            "model": row["search_model"],
            "trim": guess_craigslist_trim(row["make_model_text"], row["search_model"]),
            "price": price,
            "mileage": parse_money(row["odometer"]),  # "86,058" -> 86058
            "body_type": clean_body_type(row["type"]),
            "transmission": clean_text(row["transmission"]),
            "drivetrain": clean_drivetrain(row["drive"]),
            "fuel_type": clean_fuel(row["fuel"]),
            "cylinders": parse_cylinders(row["cylinders"]),
            "exterior_color": clean_color(row["paint_color"]),
            "condition": clean_text(row["condition"]),
            "title_status": clean_text(row["title_status"]),
            "seller_type": row["seller_type"],
            "state": REGION_TO_STATE[row["region"]],
            "listed_date": str(row["posted_date"])[:10],
            "description": row["description"],
            # only used for the model check below
            "check_text": str(row["make_model_text"]) + " " + str(row["title"]),
        }
        rows.append(new_row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# MSRP matching
# ---------------------------------------------------------------

def build_msrp_lookup(msrp_data):
    # (make, model, year) -> list of (set of trim words, msrp)
    lookup = {}
    for index, row in msrp_data.iterrows():
        key = (row["make"], row["model"], int(row["year"]))
        trim_words = set(str(row["trim_name"]).upper().split())
        if key not in lookup:
            lookup[key] = []
        lookup[key].append((trim_words, row["base_msrp"]))
    return lookup


def median_of(numbers):
    return float(pd.Series(numbers).median())


def find_msrp(lookup, make, model, year, trim):
    # Returns (msrp, how it was matched)
    key = (make, model, int(year))
    if key not in lookup:
        return None, "none"
    trims = lookup[key]

    # Try to match the first word of the listing's trim, like "XLT" or "LE"
    # ("ex-l/ex-l navi" -> "EX-L", so treat "/" like a space)
    if not is_missing(trim):
        first_word = str(trim).upper().replace("/", " ").split()[0]
        matched_prices = []
        for trim_words, price in trims:
            if first_word in trim_words:
                matched_prices.append(price)
        if len(matched_prices) > 0:
            return median_of(matched_prices), "trim"

    # No trim match, so use the middle price of all trims for that year
    all_prices = []
    for trim_words, price in trims:
        all_prices.append(price)
    return median_of(all_prices), "year_median"


# ---------------------------------------------------------------
# Main cleaning steps
# ---------------------------------------------------------------

def report(step_name, data):
    # print how many rows are left after each step (useful for the report)
    autolist_count = len(data[data["source"] == "autolist"])
    craigslist_count = len(data[data["source"] == "craigslist"])
    print(f"{step_name}: total={len(data)}, autolist={autolist_count}, craigslist={craigslist_count}")


def main():
    autolist = pd.read_csv(os.path.join(RAW, "autolist_raw.csv"))
    craigslist = pd.read_csv(os.path.join(RAW, "craigslist_raw.csv"))
    msrp_data = pd.read_csv(os.path.join(RAW, "msrp_raw.csv"))

    # 1. Put both sources into the same columns and combine them
    autolist_clean = standardize_autolist(autolist)
    craigslist_clean = standardize_craigslist(craigslist)
    data = pd.concat([autolist_clean, craigslist_clean], ignore_index=True)
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    report("Raw combined", data)

    # 2. Make sure Craigslist listings are really the model we searched for
    keep = []
    for index, row in data.iterrows():
        if row["source"] == "autolist":
            keep.append(True)
        else:
            keep.append(is_right_model(row["model"], row["check_text"]))
    data = data[keep]
    data = data.drop(columns=["check_text"])
    report("After removing wrong models", data)

    # 3. Remove bad prices (missing, $0 placeholders, $1 scams, typos)
    data = data[data["price"].notna()]
    data = data[data["price"] >= 1000]
    data = data[data["price"] <= 150000]
    report("After removing bad prices", data)

    # 4. Only keep model years we have MSRP for
    data = data[data["year"] >= config.MSRP_FIRST_YEAR]
    data = data[data["year"] <= config.MSRP_LAST_YEAR]
    report("After keeping 2008-2026 model years", data)

    # 5. Remove missing or impossible mileage
    data = data[data["mileage"].notna()]
    data = data[data["mileage"] <= 400000]
    # under 100 miles on a car that's a few years old is probably fake
    fake_mileage = (data["mileage"] < 100) & (data["year"] <= 2023)
    data = data[fake_mileage == False]
    report("After removing bad mileage", data)

    # 6. "parts only" cars aren't really for sale as cars
    data = data[data["title_status"] != "parts only"]
    report("After removing parts-only cars", data)

    # 7. Remove duplicates
    # same VIN listed twice (including on both sites) -> keep the first one.
    # Craigslist rows come first so we keep the one with a description.
    data = data.sort_values("source", ascending=False)  # craigslist before autolist

    # duplicated() marks every repeat after the first one as True.
    # Missing VINs would all count as "the same", so only look at rows that have a VIN.
    has_vin = data["vin"].notna()
    repeated_vin = data.duplicated(subset=["vin"], keep="first")
    duplicate_vin = has_vin & repeated_vin
    print("    duplicate VINs found:", duplicate_vin.sum())
    data = data[duplicate_vin == False]

    # same description posted twice on Craigslist (reposts)
    has_description = data["description"].notna()
    repeated_description = data.duplicated(subset=["description"], keep="first")
    duplicate_text = has_description & repeated_description
    print("    duplicate descriptions found:", duplicate_text.sum())
    data = data[duplicate_text == False]
    report("After removing duplicates", data)

    # 8. Attach MSRP and compute the value retention ratio
    lookup = build_msrp_lookup(msrp_data)
    msrp_list = []
    match_list = []
    for index, row in data.iterrows():
        msrp, match_type = find_msrp(lookup, row["make"], row["model"], row["year"], row["trim"])
        msrp_list.append(msrp)
        match_list.append(match_type)
    data["msrp"] = msrp_list
    data["msrp_match"] = match_list
    data = data[data["msrp"].notna()]
    data["value_retention"] = data["price"] / data["msrp"]
    report("After attaching MSRP", data)

    # 9. Remove impossible ratios (probably a wrong price or a bad match)
    data = data[data["value_retention"] >= 0.05]
    data = data[data["value_retention"] <= 1.5]
    report("After removing extreme ratios", data)

    # Save, sorted so the file is easy to look through
    data = data[FINAL_COLUMNS]
    data = data.sort_values(["make", "model", "year"])
    data = data.reset_index(drop=True)  # renumber rows 0, 1, 2, ...
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    data.to_csv(OUTPUT_FILE, index=False)
    print()
    print("Saved", len(data), "rows to", OUTPUT_FILE)
    print()
    print("How the MSRP was matched:")
    print(data["msrp_match"].value_counts().to_string())


if __name__ == "__main__":
    main()
