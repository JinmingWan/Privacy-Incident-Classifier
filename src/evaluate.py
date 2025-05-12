import torch
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from utils import load_config, set_seed, TicketDataset

def evaluate():
    cfg = load_config()
    set_seed(cfg["seed"])

    ds = TicketDataset("data/dataset.parquet")
    loader = DataLoader(ds, batch_size=cfg["batch_size"])
    model = BertForSequenceClassification.from_pretrained(cfg["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k,v in batch.items()}
            logits = model(**batch).logits
            preds += logits.argmax(dim=-1).cpu().tolist()
            labels += batch["labels"].cpu().tolist()

    print(classification_report(labels, preds,
          target_names=["data_leak","unauthorized_access","other"]))

if __name__ == "__main__":
    evaluate()
