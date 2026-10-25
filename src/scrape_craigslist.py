# Step 3b: Scrape used car listings from Craigslist.
#
# For each city, model, and seller type (owner / dealer):
#   1. Load the search results page and grab the listing links
#   2. Open each listing page and read the car details + description
#
# The CSV is saved after every search, and if the script is run again it skips
# listings that are already in the CSV (so it can pick up where it left off).
#
# Run from the project folder:
#     python src/scrape_craigslist.py          (full scrape, about an hour)
#     python src/scrape_craigslist.py test     (1 city, 1 model, 2 listings each)

import os
import sys
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

import config

OUTPUT_FILE = os.path.join(config.RAW_DATA_FOLDER, "craigslist_raw.csv")


class BlockedError(Exception):
    # raised if Craigslist starts refusing our requests
    pass


def download_page(url, params=None):
    response = requests.get(url, params=params, headers=config.HEADERS, timeout=30)
    if response.status_code == 403 or response.status_code == 429:
        raise BlockedError("Got status " + str(response.status_code) + " for " + url)
    if response.status_code != 200:
        print("      status", response.status_code, "- skipping", url)
        return None
    return BeautifulSoup(response.text, "html.parser")


def get_search_results(region, seller_code, query):
    # Returns a list of (url, title, price) for one search page
    url = "https://" + region + ".craigslist.org/search/" + seller_code
    soup = download_page(url, params={"query": query})
    results = []
    if soup is None:
        return results

    for item in soup.select("li.cl-static-search-result"):
        link = item.find("a")
        title_tag = item.select_one(".title")
        price_tag = item.select_one(".price")
        if link is None:
            continue

        title = ""
        if title_tag is not None:
            title = title_tag.get_text(strip=True)
        price = ""
        if price_tag is not None:
            price = price_tag.get_text(strip=True)

        results.append((link.get("href"), title, price))
    return results


def parse_listing_page(soup):
    # Read all the details from one listing page into a dictionary
    row = {}

    title_tag = soup.select_one("#titletextonly")
    if title_tag is not None:
        row["title"] = title_tag.get_text(strip=True)

    price_tag = soup.select_one(".postingtitletext .price")
    if price_tag is not None:
        row["price"] = price_tag.get_text(strip=True)

    # year and "make model trim" text at the top of the details box
    year_tag = soup.select_one(".valu.year")
    if year_tag is not None:
        row["year"] = year_tag.get_text(strip=True)
    makemodel_tag = soup.select_one(".valu.makemodel")
    if makemodel_tag is not None:
        row["make_model_text"] = makemodel_tag.get_text(strip=True)

    # the rest of the details look like:  <span class="labl">odometer:</span> <span class="valu">86,058</span>
    for attr in soup.select(".attrgroup .attr"):
        label_tag = attr.select_one(".labl")
        value_tag = attr.select_one(".valu")
        if label_tag is None or value_tag is None:
            continue
        # turn "paint color:" into "paint_color"
        label = label_tag.get_text(strip=True).lower()
        label = label.replace(":", "").strip().replace(" ", "_")
        row[label] = value_tag.get_text(strip=True)

    # the seller's description
    body = soup.select_one("#postingbody")
    if body is not None:
        # remove the hidden "QR Code Link to This Post" box first
        for qr_box in body.select(".print-information"):
            qr_box.decompose()
        row["description"] = body.get_text(" ", strip=True)

    # when it was posted
    time_tag = soup.select_one(".postinginfos time")
    if time_tag is not None:
        row["posted_date"] = time_tag.get("datetime")

    # post id, like "post id: 7984935015"
    for info in soup.select(".postinginfo"):
        text = info.get_text(strip=True)
        if text.startswith("post id:"):
            row["post_id"] = text.replace("post id:", "").strip()

    return row


def load_existing_rows():
    # If we already scraped some listings, load them so we don't repeat them
    if os.path.exists(OUTPUT_FILE):
        old_data = pd.read_csv(OUTPUT_FILE)
        print("Found", len(old_data), "listings already saved, will skip those")
        return old_data.to_dict("records")
    return []


def main():
    # "python src/scrape_craigslist.py test" turns on test mode
    test_mode = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_mode = True

    regions = config.CRAIGSLIST_REGIONS
    max_per_search = config.CRAIGSLIST_MAX_PER_SEARCH
    if test_mode:
        regions = regions[:1]
        max_per_search = 2
        print("TEST MODE: 1 city, 1 model, 2 listings per seller type")

    all_rows = load_existing_rows()
    seen_urls = set()
    for row in all_rows:
        seen_urls.add(row["listing_url"])

    try:
        for region in regions:
            for make in config.MAKES_AND_MODELS:
                for model in config.MAKES_AND_MODELS[make]:
                    query = config.CRAIGSLIST_QUERIES[model]

                    for seller_code in config.CRAIGSLIST_SELLER_TYPES:
                        seller_type = config.CRAIGSLIST_SELLER_TYPES[seller_code]
                        print(region, "|", make, model, "|", seller_type)

                        results = get_search_results(region, seller_code, query)
                        time.sleep(config.DELAY_SECONDS)
                        print("    search found", len(results), "listings")

                        opened = 0
                        for result in results:
                            if opened >= max_per_search:
                                break
                            url = result[0]
                            if url in seen_urls:
                                continue
                            seen_urls.add(url)

                            soup = download_page(url)
                            time.sleep(config.DELAY_SECONDS)
                            opened = opened + 1
                            if soup is None:
                                continue

                            row = parse_listing_page(soup)
                            row["listing_url"] = url
                            row["search_title"] = result[1]
                            row["search_price"] = result[2]
                            row["region"] = region
                            row["seller_type"] = seller_type
                            row["search_make"] = make
                            row["search_model"] = model
                            row["source"] = "craigslist"
                            all_rows.append(row)
                            print("      got:", row.get("title", "(no title)"))

                        # save after every search
                        pd.DataFrame(all_rows).to_csv(OUTPUT_FILE, index=False)

                    if test_mode:
                        print("Done (test mode). Saved", len(all_rows), "rows to", OUTPUT_FILE)
                        return

            print("Finished", region, "- total listings so far:", len(all_rows))

    except BlockedError as error:
        # stop right away instead of hammering the site
        print("Craigslist is blocking us:", error)
        print("Stopping. Everything so far is saved; run again later to continue.")
    except KeyboardInterrupt:
        print("Stopped by user. Everything so far is saved.")

    pd.DataFrame(all_rows).to_csv(OUTPUT_FILE, index=False)
    print("Saved", len(all_rows), "listings to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
