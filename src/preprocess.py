import pandas as pd
from transformers import BertTokenizerFast
from utils import load_config

def main():
    cfg = load_config()
    df = pd.read_csv(cfg["data_csv"])
    tokenizer = BertTokenizerFast.from_pretrained(cfg["model_name"])
    enc = tokenizer(
        df["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=cfg["max_length"]
    )
    out = pd.DataFrame({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "label": df["category"].astype("category").cat.codes
    })
    out.to_parquet("data/dataset.parquet", index=False)
    print("Saved preprocessed data to data/dataset.parquet")

if __name__ == "__main__":
    main()
