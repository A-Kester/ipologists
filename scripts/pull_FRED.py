"""
Fetches the macro data from the FRED using its API.
The FRED_API_KEY is obtained for free on their website
"""

import pandas as pd
from  fredapi import Fred
import os
from dotenv import load_dotenv

load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')

def fetch(start_date ="2000-01-01"):
    fred = Fred(api_key= FRED_API_KEY)

    series = { 'vix': 'VIXCLS',
    'nasdaq': 'NASDAQCOM',
    'fed_funds': 'FEDFUNDS',
    'treasury_10y':'DGS10',
    'cpi': 'CPIAUCSL',
    'unemployment':'UNRATE',
    'gdp':'GDPC1',
    'ipo_volume':'IPB50001N'}


    macro_data = pd.DataFrame()
    for name, id in series.items():
        macro_data[name] = fred.get_series(id, observation_start=start_date)

    macro_data.index= pd.to_datetime(macro_data.index)
    macro_data = macro_data.resample("ME").last().ffill()
    macro_data['market_return_1m'] = macro_data['nasdaq'].pct_change().shift(1)
    macro_data.index.name = 'date'

    macro_data.to_csv("../data/processed/macro.csv")
    return macro_data

if __name__ == "__main__":
    macro_data = fetch()
    print(macro_data.tail())

