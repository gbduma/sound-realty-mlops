import json, time, requests, pandas as pd, sys
API = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080/predict/full"
df = pd.read_csv("data/future_unseen_examples.csv")
for _, row in df.head(5).iterrows():
    payload = row.drop(labels=["price","date","id"], errors="ignore").to_dict()
    r = requests.post(API, json=payload, timeout=30)
    print(r.status_code, r.json())
    time.sleep(0.2)
