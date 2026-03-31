from datasets import load_dataset
import pandas as pd

dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

train = dataset["train"].to_pandas()
val = dataset["validation"].to_pandas()
test = dataset["test"].to_pandas()

train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)

print("Done! Shape:", train.shape)
print(train.head())
