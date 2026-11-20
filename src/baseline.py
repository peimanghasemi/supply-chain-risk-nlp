# Baseline: TF-IDF + logistic regression for risk event classification.
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def main(data: str, out: str) -> None:
    df = pd.read_csv(data)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["label"], test_size=0.25, stratify=df["label"], random_state=42
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(classification_report(y_te, pred, digits=3))

    Path(out).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(out) / "baseline.joblib")
    metrics = {"macro_f1": round(f1_score(y_te, pred, average="macro"), 4)}
    (Path(out) / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sample_news.csv")
    p.add_argument("--out", default="models")
    a = p.parse_args()
    main(a.data, a.out)
