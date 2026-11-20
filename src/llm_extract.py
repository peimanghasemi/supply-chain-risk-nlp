# Prompt-based extraction of risk events with an LLM.
# Works with any OpenAI-compatible API: set LLM_API_KEY (and LLM_BASE_URL / LLM_MODEL if needed).
import argparse
import json
import os

import pandas as pd
from openai import OpenAI

SCHEMA = {
    "supplier": "company name or null",
    "event_type": "insolvency | cyber_attack | fire | natural_disaster | labor_strike | no_risk",
    "location": "country or city, or null",
    "severity": "low | medium | high",
    "expected_impact": "one short sentence on the likely effect on customers' supply",
}

PROMPT = """You are a supply chain risk analyst.
Read the news text and return ONLY a JSON object with these fields:
{schema}
If the text describes no supply risk, use event_type "no_risk" and severity "low".

News text:
\"\"\"{text}\"\"\""""


def extract(client: OpenAI, model: str, text: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": PROMPT.format(schema=json.dumps(SCHEMA, indent=2), text=text)}],
    )
    return json.loads(resp.choices[0].message.content)


def main(data: str, n: int, out: str) -> None:
    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("LLM_BASE_URL"))
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    df = pd.read_csv(data).head(n)
    records = []
    for text, label in zip(df["text"], df["label"]):
        try:
            rec = extract(client, model, text)
        except Exception as e:  # don't stop the whole batch on one bad response
            rec = {"error": str(e)}
        rec.update({"text": text, "gold_label": label})
        records.append(rec)
    res = pd.DataFrame(records)
    res.to_csv(out, index=False)
    if "event_type" in res:
        acc = (res["event_type"] == res["gold_label"]).mean()
        print(f"Event-type agreement with gold labels: {acc:.3f} on {len(res)} texts")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sample_news.csv")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--out", default="llm_extractions.csv")
    a = p.parse_args()
    main(a.data, a.n, a.out)
