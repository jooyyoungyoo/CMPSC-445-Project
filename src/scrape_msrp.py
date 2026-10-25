# Step 4: Collect the original MSRP for each make / model / model year.
#
# J.D. Power has a page for every model year, like
#     https://www.jdpower.com/cars/2015/toyota/camry
# and the page has the base MSRP for every trim stored in its data, like
#     "trimName":"Sedan 4D LE I4", ... "baseMSRP":22970, ... "modelYear":2015
#
# We save one row per trim: make, model, year, trim name, base MSRP.
# If the script is run again, it skips model years already in the CSV.
#
# Run from the project folder:
#     python src/scrape_msrp.py          (full scrape, about 20 minutes)
#     python src/scrape_msrp.py test     (just 2015 Camry)

import os
import sys
import time
import requests
import pandas as pd

import config

OUTPUT_FILE = os.path.join(config.RAW_DATA_FOLDER, "msrp_raw.csv")


def make_url(make, model, year):
    # "Silverado 1500" -> "silverado-1500", "CR-V" -> "cr-v"
    make_part = make.lower().replace(" ", "-")
    model_part = model.lower().replace(" ", "-")
    return "https://www.jdpower.com/cars/" + str(year) + "/" + make_part + "/" + model_part


def download_with_retry(url):
    # J.D. Power sometimes gives a random 403, so wait and try again
    for attempt in range(3):
        response = requests.get(url, headers=config.HEADERS, timeout=30)
        if response.status_code == 200:
            return response.text
        if response.status_code == 404:
            return None
        print("    status", response.status_code, "- waiting 30 seconds and trying again")
        time.sleep(30)
    return None


def read_number(text, start_position):
    # read the digits starting at start_position, like "22970,..." -> 22970
    digits = ""
    position = start_position
    while position < len(text) and text[position].isdigit():
        digits = digits + text[position]
        position = position + 1
    if digits == "":
        return None
    return int(digits)


def find_trims(page_text, year):
    # The data in the page has \" instead of ", so fix that first
    page_text = page_text.replace('\\"', '"')

    # Split the page at every trim name. Each piece then starts with
    # one trim's name and contains that trim's baseMSRP.
    pieces = page_text.split('"trimName":"')

    trims = {}
    for piece in pieces[1:]:
        trim_name = piece[:piece.find('"')]

        # make sure this trim is really from the year we asked for
        if '"modelYear":' + str(year) not in piece:
            continue

        position = piece.find('"baseMSRP":')
        if position == -1:
            continue
        msrp = read_number(piece, position + len('"baseMSRP":'))

        # the page repeats some data, so only keep each trim once
        if msrp is not None and trim_name not in trims:
            trims[trim_name] = msrp
    return trims


def load_existing_rows():
    if os.path.exists(OUTPUT_FILE):
        old_data = pd.read_csv(OUTPUT_FILE)
        print("Found", len(old_data), "trims already saved, will skip those years")
        return old_data.to_dict("records")
    return []


def main():
    # "python src/scrape_msrp.py test" turns on test mode
    test_mode = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_mode = True

    all_rows = load_existing_rows()
    already_done = set()
    for row in all_rows:
        already_done.add((row["make"], row["model"], row["year"]))

    missing = []

    for make in config.MAKES_AND_MODELS:
        for model in config.MAKES_AND_MODELS[make]:
            for year in range(config.MSRP_FIRST_YEAR, config.MSRP_LAST_YEAR + 1):
                # test mode only does the 2015 Camry
                if test_mode:
                    if model != "Camry" or year != 2015:
                        continue
                if (make, model, year) in already_done:
                    continue

                url = make_url(make, model, year)
                page_text = download_with_retry(url)
                time.sleep(config.DELAY_SECONDS)

                trims = {}
                if page_text is not None:
                    trims = find_trims(page_text, year)

                if len(trims) == 0:
                    print(year, make, model, "-> no MSRP found")
                    missing.append(str(year) + " " + make + " " + model)
                    continue

                for trim_name in trims:
                    all_rows.append({
                        "make": make,
                        "model": model,
                        "year": year,
                        "trim_name": trim_name,
                        "base_msrp": trims[trim_name],
                        "source": "jdpower",
                    })
                print(year, make, model, "->", len(trims), "trims")

            # save after every model
            pd.DataFrame(all_rows).to_csv(OUTPUT_FILE, index=False)

    pd.DataFrame(all_rows).to_csv(OUTPUT_FILE, index=False)
    print("Done! Saved", len(all_rows), "trim prices to", OUTPUT_FILE)
    print("Model years with no MSRP:", len(missing))
    for item in missing:
        print("   ", item)


if __name__ == "__main__":
    main()
