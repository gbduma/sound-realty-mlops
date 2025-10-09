# scripts/post_examples.py
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

def main():
    p = argparse.ArgumentParser()
    p.add_argument("url", help="Endpoint URL, e.g. http://localhost:8080/predict/full")
    p.add_argument("--csv", default="data/future_unseen_examples.csv", help="Path to examples CSV")
    p.add_argument("--limit", type=int, default=5, help="How many rows to send")
    p.add_argument("--model", default=None, help="Model selector, e.g. baseline_knn or enhanced_auto")
    p.add_argument("--show-aux", action="store_true", default=True, help="Print aux (comps/quality) info")
    p.add_argument("--no-aux", dest="show_aux", action="store_false")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    rows = df.to_dict(orient="records")[: args.limit]

    # allow ?model=... query param
    url = args.url
    if args.model:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}model={args.model}"

    for i, row in enumerate(rows, 1):
        try:
            t0 = time.time()
            r = requests.post(url, json=row, timeout=15)
            elapsed = (time.time() - t0) * 1000
            r.raise_for_status()
            obj = r.json()

            pred = obj.get("prediction")
            mv   = obj.get("model_version")
            lat  = obj.get("latency_ms", round(elapsed, 2))
            print(f"{r.status_code} idx={i} pred={pred:.2f} model={mv} latency_ms={lat}")

            if args.show_aux and "aux" in obj:
                aux = obj["aux"]
                comps = aux.get("comps", {})
                quality = aux.get("quality", {})
                # Keep it concise but useful
                cc = comps.get("comp_count")
                ce = comps.get("comps_estimate")
                band = comps.get("comps_band")
                ood = quality.get("ood_score")
                conf = quality.get("confidence")
                msgs = quality.get("messages", [])
                print("  aux.comps:", json.dumps({"comp_count": cc, "comps_estimate": ce, "comps_band": band}, indent=2))
                print("  aux.quality:", json.dumps({"ood_score": ood, "confidence": conf, "messages": msgs}, indent=2))

        except requests.HTTPError as e:
            print(json.dumps({"idx": i, "error": str(e)}))
        except Exception as e:
            print(json.dumps({"idx": i, "error": f"unexpected: {e}"}))

if __name__ == "__main__":
    main()
