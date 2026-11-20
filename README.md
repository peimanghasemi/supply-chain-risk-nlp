# supply-chain-risk-nlp

Small project for detecting supply chain risk events (supplier insolvency, cyber attacks, fires, natural disasters, strikes) in news text, and ranking suppliers by risk.

I built it to compare three ways of doing the same NLP task, which is a question I ran into at work: a classic baseline, a fine-tuned Transformer, and an LLM with a prompt.

## What's inside

- `src/baseline.py` - TF-IDF + logistic regression. Fast, cheap, easy to explain.
- `src/finetune_bert.py` - fine-tunes DistilBERT (or any Hugging Face model) for event classification.
- `src/llm_extract.py` - prompt-based extraction with an LLM. Returns JSON with supplier, event type, location, severity and expected impact.
- `spark/RiskAggregation.scala` - Spark job that aggregates events per supplier, weights them by severity and ranks suppliers.
- `data/make_sample_data.py` - builds the sample dataset.

Classes: `insolvency`, `cyber_attack`, `fire`, `natural_disaster`, `labor_strike`, `no_risk`.

## Running it

```bash
pip install -r requirements.txt
python data/make_sample_data.py              # builds data/sample_news.csv

python src/baseline.py
python src/finetune_bert.py --epochs 3        # use a GPU if you have one (Colab works fine)

export LLM_API_KEY=...                         # any OpenAI-compatible endpoint
python src/llm_extract.py --n 30
```

Spark part:

```bash
cd spark
sbt package
spark-submit --class RiskAggregation target/scala-2.12/risk-aggregation_2.12-0.1.jar events.csv out/
```

## About the data

The sample data in `data/sample_news.csv` is synthetic (template headlines about made-up companies), so everything runs without downloading anything. The models score almost perfectly on it, which obviously doesn't mean much. To get real numbers, swap in labelled news with the columns `text,label`.

## Notes

In a real system I wouldn't run the LLM on every article. A cheap classifier can filter the stream first, and the LLM then only extracts details from texts flagged as risks. That keeps cost and latency down. The models also need monitoring once deployed: new sources and new kinds of events show up all the time, so a small human-checked sample each week is useful for spotting drift and for retraining.

## License

MIT
