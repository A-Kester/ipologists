from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import pandas as pd
import os, re , time

load_dotenv()
EMAIL = os.getenv("EMAIL")
DATA_PATH = "../../data/processed/bloomberg_macro.csv"
S1_DIR = "../../data/s1_filings"
TEXT_DIR = "../../data/s1_text"

CHAR_LIMIT = 10000
os.makedirs(S1_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

def is_valid_ticker(ticker):
    ticker = str(ticker).strip()
    if re.search(r'\d', ticker):
        return False
    if ticker.endswith("Q"):
        return False
    return True

def find_filing(ticker):
    base = os.path.join(S1_DIR, "sec-edgar-filings", ticker, "S-1")
    for root, _, files in os.walk(base):
        for f in files:
            if f == 'full-submission.txt':
                return os.path.join(root, f)
    return None

def extract_text(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    return soup.get_text(separator=' ', strip=True)

def extract_risk_factors(text):
    pattern = re.compile(
        r'RISK\s+FACTORS(.*?)' r'(USE\s+OF\s+PROCEEDS|DILUTION|DIVIDEND\s+POLICY|CAPITALIZATION' r'|SELECTED\s+FINANCIAL|MANAGEMENT\'S\s+DISCUSSION' r'|BUSINESS\s+OVERVIEW|OUR\s+BUSINESS|INDUSTRY\s+BACKGROUND'r'|ITEM\s+2|FORWARD.LOOKING\s+STATEMENTS)',
        re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None
    longest = max(matches, key=lambda x: len(x[0]))
    text = longest[0].strip()
    keywords = ['adverse', 'risk', 'could', 'may not', 'unable', 'fail', 'harm', 'loss']
    if sum(1 for word in keywords if word in text.lower()) < 3:
        return None
    return text[:CHAR_LIMIT]

def run():
    df = pd.read_csv(DATA_PATH)

    tickers = df['ticker'].dropna().unique()
    valid = [t for t in tickers if is_valid_ticker(t)]
    already_done = set(f.replace('.txt', '') for f in os.listdir(TEXT_DIR))
    tickers_to_pull = [t for t in valid if t not in already_done]
    dl = Downloader('IPO Research', EMAIL, S1_DIR)

    
    for i, ticker in enumerate(tickers_to_pull):
        print(f"[{i+1}/{len(tickers_to_pull)}] {ticker}", end=' ')
        try:
            dl.get("S-1", ticker, limit=1)
        except Exception as e:
            print("download has failed")
            continue
        filepath = find_filing(ticker)
        if not filepath:
            continue
        try:
            text = extract_text(filepath)
            risk = extract_risk_factors(text)
            with open(os.path.join(TEXT_DIR, f'{ticker}.txt'), 'w', encoding='utf-8') as f:
                f.write(risk)
            print("saved risk profile")
        except Exception as e:
            print(f"failed: {e}")
        time.sleep(0.5)


if __name__ == "__main__":
    run()