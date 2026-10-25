# Shared settings for the scrapers.
# Both scrapers import from here so they always use the same makes/models.

import os

# The project folder is one level up from this src/ folder
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 4 makes, 3 models each
MAKES_AND_MODELS = {
    "Toyota": ["Camry", "Corolla", "RAV4"],
    "Honda": ["Civic", "Accord", "CR-V"],
    "Ford": ["F-150", "Escape", "Explorer"],
    "Chevrolet": ["Silverado 1500", "Equinox", "Malibu"],
}

# Pretend to be a normal browser, otherwise some sites reject the request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# Seconds to wait between requests so we don't hammer the servers
DELAY_SECONDS = 3

# Where the raw CSV files go
RAW_DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data", "raw")

# ---- Autolist settings ----
# 20 cars per page, so 4 pages = up to 80 cars per model
AUTOLIST_PAGES_PER_MODEL = 4

# ---- MSRP (J.D. Power) settings ----
# J.D. Power has trim-level MSRP back to the 2008 model year
MSRP_FIRST_YEAR = 2008
MSRP_LAST_YEAR = 2026

# ---- Craigslist settings ----
# Big cities so there are enough listings for each model
CRAIGSLIST_REGIONS = [
    "newyork", "losangeles", "chicago", "houston", "dallas",
    "phoenix", "philadelphia", "atlanta", "seattle", "denver",
]

# cto = for sale by owner, ctd = for sale by dealer
CRAIGSLIST_SELLER_TYPES = {"cto": "owner", "ctd": "dealer"}

# Max listing pages to open per (region, model, seller type) search
CRAIGSLIST_MAX_PER_SEARCH = 4

# People type model names in different ways, so each model gets its own
# search text. In Craigslist search, "|" means OR.
CRAIGSLIST_QUERIES = {
    "Camry": "camry",
    "Corolla": "corolla",
    "RAV4": "rav4|rav-4",
    "Civic": "civic",
    "Accord": "accord",
    "CR-V": "cr-v|crv",
    "F-150": "f-150|f150",
    "Escape": "ford escape",
    "Explorer": "ford explorer",
    "Silverado 1500": "silverado 1500",
    "Equinox": "equinox",
    "Malibu": "malibu",
}
