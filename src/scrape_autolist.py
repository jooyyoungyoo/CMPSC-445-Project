# Step 3a: Scrape used car listings from Autolist.
#
# Autolist's website loads its search results from a JSON API, so instead of
# parsing HTML we can ask that API directly. Each page gives 20 cars.
#
# Run from the project folder:
#     python src/scrape_autolist.py          (full scrape)
#     python src/scrape_autolist.py test     (1 page of 1 model, to check it works)

import os
import sys
import time
import requests
import pandas as pd

import config

API_URL = "https://www.autolist.com/api/v2/search"
OUTPUT_FILE = os.path.join(config.RAW_DATA_FOLDER, "autolist_raw.csv")

# The API gives ~60 fields per car. These are the ones we want to keep.
# (key in the API response -> column name in our CSV)
FIELDS_TO_KEEP = {
    "id": "listing_id",
    "vin": "vin",
    "year": "year",
    "make_name": "make",
    "model_name": "model",
    "trim": "trim",
    "condition": "condition",
    "price": "price",
    "mileage": "mileage",
    "body_type": "body_type",
    "transmission": "transmission",
    "driveline": "drivetrain",
    "fuel_type": "fuel_type",
    "engine_cylinders": "cylinders",
    "door_count": "doors",
    "combined_mpg": "combined_mpg",
    "city_mpg": "city_mpg",
    "hwy_mpg": "hwy_mpg",
    "normalized_color_exterior": "exterior_color",
    "normalized_color_interior": "interior_color",
    "dealer_name": "dealer_name",
    "city": "city",
    "state": "state",
    "created_at": "listed_date",
}


def get_one_page(make, model, page_number):
    # Ask the API for one page of used cars for this make/model
    params = {
        "make": make,
        "model": model,
        "page": page_number,
        "limit": 20,
        "condition[]": "used",
    }
    response = requests.get(API_URL, params=params, headers=config.HEADERS, timeout=30)
    if response.status_code != 200:
        print("    got status", response.status_code, "- skipping this page")
        return []
    data = response.json()
    return data.get("records", [])


def clean_record(record, make, model):
    # Pull out only the fields we care about
    row = {}
    for api_key in FIELDS_TO_KEEP:
        column_name = FIELDS_TO_KEEP[api_key]
        row[column_name] = record.get(api_key)

    # remember what we searched for, and where the data came from
    row["search_make"] = make
    row["search_model"] = model
    row["source"] = "autolist"
    row["seller_type"] = "dealer"  # Autolist only lists dealer cars
    return row


def main():
    # "python src/scrape_autolist.py test" turns on test mode
    test_mode = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_mode = True

    pages_per_model = config.AUTOLIST_PAGES_PER_MODEL
    if test_mode:
        pages_per_model = 1
        print("TEST MODE: only 1 page of the first model")

    all_rows = []
    seen_vins = set()

    for make in config.MAKES_AND_MODELS:
        for model in config.MAKES_AND_MODELS[make]:
            print("Scraping", make, model)

            for page_number in range(1, pages_per_model + 1):
                records = get_one_page(make, model, page_number)
                print("    page", page_number, "->", len(records), "cars")

                for record in records:
                    # skip cars we already have (same VIN)
                    vin = record.get("vin")
                    if vin in seen_vins:
                        continue
                    seen_vins.add(vin)
                    all_rows.append(clean_record(record, make, model))

                time.sleep(config.DELAY_SECONDS)

            # save after every model so we don't lose data if something breaks
            pd.DataFrame(all_rows).to_csv(OUTPUT_FILE, index=False)
            print("    saved", len(all_rows), "total rows so far")

            if test_mode:
                print("Done (test mode). Saved to", OUTPUT_FILE)
                return

    print("Done! Saved", len(all_rows), "listings to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
