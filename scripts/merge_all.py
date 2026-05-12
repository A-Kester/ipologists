""" 
Produces both of our datasets:
    - dataset_full.csv  has all of the IPOs, with no risk scores
    - dataset_with_risk.csv has the 857 IPOs that we have S-1 risk scores for
"""

import pandas as pd 

main_path  = "../data/processed/bloomberg_macro.csv"
risk_path  = "../data/processed/risk_scores.csv"
full_output   = "../data/final/dataset_full.csv"
risk_output   = "../data/final/dataset_with_risk.csv"

def run():
    df = pd.read_csv(main_path)
    risk = pd.read_csv(risk_path)

    print(f"main length: {len(df)}")
    print(f"risk length: {len(risk)}")

    df.to_csv(full_output, index =False)
    print("Saved full dataset")

    df_with_risk = df.merge(risk, on="ticker", how="inner")
    df_with_risk.to_csv(risk_output, index = False)
    print(f"Saved risk dataset with {len(df_with_risk)} rows")

    print(f"columns of risk dataset: {list(df_with_risk.columns)}")

if __name__ == "__main__":
    run()