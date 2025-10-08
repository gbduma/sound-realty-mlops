import json, time, requests, pandas as pd, sys

API = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080/predict/full"

df = pd.read_csv("data/future_unseen_examples.csv")

for i, (_, row) in enumerate(df.head(8).iterrows(), start=1):
    payload = row.drop(labels=["price", "date", "id"], errors="ignore").to_dict()
    try:
        r = requests.post(API, json=payload, timeout=30)
        r.raise_for_status()
        body = r.json()
        # print a single tidy line
        print(json.dumps({
            "idx": i,
            "prediction": body.get("prediction"),
            "latency_ms": body.get("latency_ms"),
            "model_version": body.get("model_version")
        }))
    except Exception as e:
        print(json.dumps({"idx": i, "error": str(e)}))
    time.sleep(0.1)