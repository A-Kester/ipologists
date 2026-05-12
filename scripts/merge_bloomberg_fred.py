"""
Merges the macroeconomic data from the FRED to the Bloomberg IPO dataset.
Uses month as the key for the left-join merge
"""
import pandas as pd

IPO_DATA = "../data/processed/ipo_clean.csv"
MACRO_DATA = "../data/processed/macro.csv"

def merge():
    ipo = pd.read_csv(IPO_DATA, parse_dates=["Pricing Date"], index_col=False)
    macro = pd.read_csv(MACRO_DATA, parse_dates=["date"], index_col=False)

    ipo["month_key"] = ipo ["Pricing Date"].dt.to_period("M").dt.to_timestamp('M')
    macro["month_key"]=macro["date"].dt.to_period("M").dt.to_timestamp("M")

    macro= macro.drop(columns=["date"])

    merged_data = ipo.merge(macro, on="month_key", how = "left")
    merged_data= merged_data.drop(columns=["month_key"])

    print(f"Shape: {merged_data.shape}")

    merged_data.to_csv("../data/processed/bloomberg_macro.csv", index=False)
    return merged_data

if __name__ =="__main__":
    df = merge()
    print(df.head())