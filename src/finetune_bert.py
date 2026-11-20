# Fine-tune a Transformer (DistilBERT by default) on the risk event labels.
# CPU is ok for the sample data, use a GPU for anything bigger.
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


def main(data: str, model_name: str, out: str, epochs: int) -> None:
    df = pd.read_csv(data)
    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    df["labels"] = df["label"].map(label2id)

    train_df, test_df = train_test_split(df, test_size=0.25, stratify=df["labels"], random_state=42)
    tok = AutoTokenizer.from_pretrained(model_name)

    def encode(batch):
        return tok(batch["text"], truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(train_df[["text", "labels"]], preserve_index=False).map(encode, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "labels"]], preserve_index=False).map(encode, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    def compute_metrics(eval_pred):
        logits, y = eval_pred
        return {"macro_f1": f1_score(y, np.argmax(logits, axis=-1), average="macro")}

    args = TrainingArguments(
        output_dir=out,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=10,
        report_to="none",
        seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
    trainer.train()

    pred = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
    print(classification_report(test_ds["labels"], pred, target_names=labels, digits=3))

    Path(out).mkdir(parents=True, exist_ok=True)
    trainer.save_model(Path(out) / "best")
    tok.save_pretrained(Path(out) / "best")
    metrics = {"macro_f1": round(f1_score(test_ds["labels"], pred, average="macro"), 4), "model": model_name}
    (Path(out) / "bert_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sample_news.csv")
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--out", default="models/bert")
    p.add_argument("--epochs", type=int, default=3)
    a = p.parse_args()
    torch.manual_seed(42)
    main(a.data, a.model, a.out, a.epochs)
