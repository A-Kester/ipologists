"""
Checks the quality of extracted S-1 risk factor texts.
It deletes bad extractions so they can be retrived with improved regex
"""
import os

TEXT_PATH = "../../data/s1_text"

keywords = ["adverse", "risk", "could","may not","uncertainty", "fail", "loss", "unable","cannot",'harm',"litigation","debt","volatile","competition"]

files = os.listdir(TEXT_PATH)
accept=[]
reject=[]

for f in files:
    path = os.path.join(TEXT_PATH, f)
    with open(path,"r") as file:
        text = file.read().lower()
    
    matches = sum(1 for word in keywords if word in text)

    if matches >= 5:
        accept.append(f)
    else:
        reject.append(f)

for f in reject:
    os.remove(os.path.join(TEXT_PATH,f))
    

print(f"deleted: {len(reject)}")
print(f"accepted: {len(accept)}")

    