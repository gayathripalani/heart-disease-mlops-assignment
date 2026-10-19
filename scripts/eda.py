"""EDA figures for the cleaned Heart Disease data -> docs/screenshots/eda/."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heart.data import CATEGORICAL, NUMERIC, TARGET  # noqa: E402

OUT = "docs/screenshots/eda"
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv("data/processed/heart_clean.csv")
sns.set_theme(style="whitegrid")


def save(name):
  plt.tight_layout()
  plt.savefig(f"{OUT}/{name}.png", dpi=140)
  plt.close()


print("shape:", df.shape)
print("missing:\n", df.isna().sum()[df.isna().sum() > 0])
print("class balance:\n", df[TARGET].value_counts(normalize=True).round(3))
print(df[NUMERIC].describe().round(2))

ax = df[TARGET].map({0: "No disease", 1: "Disease"}).value_counts().plot.bar(rot=0)
ax.set_title("Class balance")
save("class_balance")

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, col in zip(axes.flat, NUMERIC):
  sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=ax, element="step")
axes.flat[-1].axis("off")
save("numeric_histograms")

plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation heatmap")
save("correlation_heatmap")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["thalach", "oldpeak", "age"]):
  sns.boxplot(data=df, x=TARGET, y=col, ax=ax)
save("boxplots")

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.flat, CATEGORICAL):
  sns.countplot(data=df, x=col, hue=TARGET, ax=ax)
save("categorical_counts")

miss = df.isna().sum()
miss[miss > 0].plot.bar(rot=0, title="Missing values per column")
save("missing_values")
print("figures written to", OUT)
