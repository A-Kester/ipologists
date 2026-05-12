"""
Uses ChatGPT's gpt-4o-mini model to score the S-1 filings risk factors excepts on 
four categories: financial, competitive, regulatory, and overall
"""

import os 
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import json
load_dotenv()

TEXT_PATH = "../data/s1_text"
OUTPUT_PATH = "../data/processed/risk_scores.csv"
model = "gpt-4o-mini"

risk_categories = ["regulatory_risk", "competitive_risk", "financial_risk", "overall_risk"]

prompt= """You are a financial analyst scoring IPO risk from S-1 filings excerpts. 
the text you receive is a partial excerpt of the Risk Factors section,
so focus on what is mentioned rather than penalizing for what is absent.

Score each dimension from 1 (very low risk) to 10 (very high risk):

- regulatory_risk: exposure to government policy, legal challenges, regulatory change,
    or compliance burdens mentioned in the text
- competitive_risk: threats from competitors, market alternatives, pricing pressure,
    or loss of key customers/partners
- financial_risk: concerns about leverage, liquidity, capital requirements,
    cash burn, or financial covenant risk
- overall_risk: a single holistic score summarizing the severity of risks across
    all dimensions present in the excerpt

Return ONLY a valid JSON object with exactly these four keys and integer values 1-10:
Do NOT include explanation or markdown: Example output: {"regulatory_risk":6,"competitive_risk":4,"financial_risk":5,"overall_risk":5}

"""

client = OpenAI()

def score_text(ticker, text):
    for attempt in range(1,2+1):
        try:
            response = client.chat.completions.create(
                model = model,
                temperature=0,
                response_format={"type": "json_object"},
                messages = [
                    {"role": "system", "content":prompt},
                    {"role": "user", "content":f"S-1 Risk Factors excerpt for {ticker}:\n\n{text}"}])
            raw = response.choices[0].message.content
            scores=json.loads(raw)

            for cat in risk_categories:
                if cat not in scores:
                    raise ValueError("Missing a key")
                scores[cat]= int(scores[cat])
                if not (1<= scores[cat]<= 10):
                    raise ValueError("Incorrect scoring")
            return scores
        
        except Exception as e:
            print(f"Attempt Failed: {e}")
            if attempt< 2:
                time.sleep(2**attempt)
    return None


def run():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    files = [file for file in os.listdir(TEXT_PATH) if file.endswith(".txt")]

    results = []
    for i, filename in enumerate(files):
        ticker = filename.replace(".txt","")
        path = os.path.join(TEXT_PATH, filename)

        with open(path, "r") as f:
            text = f.read()
        scores = score_text(ticker, text)

        if scores:
            scores["ticker"] = ticker
            results.append(scores)
            print(f"{i+1}/{len(files)} {ticker} is good")
        else:
            print(f"{i+1}/{len(files)} {ticker} has failed")
        
        time.sleep(0.3)

    df = pd.DataFrame(results)[["ticker"] + risk_categories]
    df.to_csv(OUTPUT_PATH, index=False)
    print(df[risk_categories].describe().round(2))

if __name__ == "__main__":
    run()
