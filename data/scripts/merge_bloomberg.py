"""
Imports the raw Bloomberg datasets and merges them together.
Cleans up the formatting / Adds a binary underpriced column, an offer size to market cap metric, and cleans up the bookrunner columns
Highlights who the lead bookrunner is and whether a bulge bracket was one of the underwritters.
"""

import pandas as pd

data_2000 = "../../data/raw/ipo_data_2000.csv"
data_2010 = "../../data/raw/ipo_data_2010.csv"

def loading_and_cleaning(path):
    df = pd.read_csv(path)

    df['Pricing Date'] = pd.to_datetime(df["Pricing Date"], format='mixed')
    df = df[df['Issuer Ticker'].str.endswith(' US', na=False)]
    df["ticker"] = df["Issuer Ticker"].str.replace(" US","").str.strip()
    df['Industry Sector'] = df['Industry Sector'].str.strip()

    df['underpriced'] = (df['Offer To 1st Close'] > 0).astype(int)
    df['offer_size_to_mktcap'] = df['Offer Size (M)'] / df['Market Cap at Offer (M)']
    df["offer_size_to_mktcap"] = df["offer_size_to_mktcap"].replace([float("inf"), float("-inf")], pd.NA)
    
    df['lead_bookrunner'] = df['Bookrunner'].str.split(",").str[0].str.strip()
    bulge_bracket = ["Goldman Sachs", "Morgan Stanley", "JP Morgan", "BofA Securities", "Citi", "Barclays"]
    df["has_bulge_bracket"] = df["Bookrunner"].apply(lambda x : int(any(bank in str(x) for bank in bulge_bracket)))

    df = df.drop(columns=["Bookrunner", "Issuer Ticker"])
    return df

def bloomberg_merge():
    df_2000 = loading_and_cleaning(data_2000)
    df_2010 = loading_and_cleaning(data_2010)

    ipo_data = pd.concat([df_2000, df_2010], ignore_index=True)
    ipo_data = ipo_data.sort_values('Pricing Date')
    
    ipo_data.to_csv("../../data/processed/ipo_clean.csv", index=False)

if __name__=="__main__":
    ipo_data = bloomberg_merge()