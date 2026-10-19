"""Download the UCI Heart Disease (Cleveland) data and write the cleaned CSV."""
import os

from heart.data import clean, download, load_raw

raw = download("data/raw/processed.cleveland.data")
os.makedirs("data/processed", exist_ok=True)
df = clean(load_raw(raw))
df.to_csv("data/processed/heart_clean.csv", index=False)
print(f"wrote data/processed/heart_clean.csv: {df.shape[0]} rows, {df.shape[1]} cols")
