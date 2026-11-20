# Builds a small synthetic dataset of risk headlines so the pipeline can run end to end.
import csv
import random

random.seed(7)

suppliers = ["Brenner Metallwerk", "Kova Plastics", "Nordia Semiconductors", "Sato Precision",
             "Alpina Chemicals", "Riverton Steel", "Lumen Electronics", "Delta Pharma Ingredients",
             "Vistula Castings", "Mekong Textiles", "Orion Logistics", "Helvar Components"]
places = ["Germany", "Poland", "Taiwan", "Vietnam", "Italy", "Mexico", "Austria", "India",
          "Thailand", "Czech Republic", "Japan", "Turkey"]

templates = {
    "insolvency": [
        "{s} files for insolvency after months of losses",
        "Court opens insolvency proceedings against {s} in {l}",
        "{s} misses debt payments, creditors push for restructuring",
        "Supplier {s} enters administration, deliveries at risk",
        "{s} warns it may run out of cash by next quarter",
    ],
    "cyber_attack": [
        "Ransomware attack halts production at {s}",
        "{s} shuts IT systems after cyber attack in {l}",
        "Hackers breach {s}, order processing disrupted",
        "{s} confirms data breach, plants running at reduced capacity",
        "Cyber incident forces {s} to stop shipments for several days",
    ],
    "fire": [
        "Fire breaks out at {s} plant in {l}",
        "Blaze destroys warehouse of {s}",
        "{s} factory evacuated after explosion and fire",
        "Production stopped at {s} after fire in main hall",
        "Firefighters battle fire at {s} site in {l}",
    ],
    "natural_disaster": [
        "Floods in {l} force {s} to close plant",
        "Earthquake damages {s} facility in {l}",
        "Typhoon disrupts operations at {s}",
        "Heavy rain floods {s} warehouse, stock damaged",
        "{s} halts output as storm hits {l}",
    ],
    "labor_strike": [
        "Workers at {s} go on strike over wages",
        "Union announces week-long strike at {s} in {l}",
        "Strike at {s} port terminal delays shipments",
        "{s} staff walk out, production slowed",
        "Labour dispute at {s} enters second week",
    ],
    "no_risk": [
        "{s} opens new research centre in {l}",
        "{s} reports record revenue for the year",
        "{s} appoints new chief executive",
        "{s} launches sustainable product line",
        "{s} expands production capacity in {l}",
        "{s} wins industry award for quality",
    ],
}

rows = []
for label, ts in templates.items():
    for _ in range(60):
        text = random.choice(ts).format(s=random.choice(suppliers), l=random.choice(places))
        rows.append((text, label))
random.shuffle(rows)

with open("data/sample_news.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["text", "label"])
    w.writerows(rows)

print(f"wrote {len(rows)} rows")
