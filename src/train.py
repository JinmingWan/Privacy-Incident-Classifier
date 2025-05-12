import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AdamW
from utils import load_config, set_seed

class TicketDataset(Dataset):
    def __init__(self, path):
        df = pd.read_parquet(path)
        self.input_ids = df["input_ids"].tolist()
        self.attention_mask = df["attention_mask"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "labels": torch.tensor(self.labels[idx])
        }

def train():
    cfg = load_config()
    set_seed(cfg["seed"])

    # Datasets & loaders
    train_ds = TicketDataset("data/dataset.parquet")
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    # Model & optimizer
    model = BertForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=cfg["num_labels"]
    )
    optim = AdamW(model.parameters(), lr=cfg["learning_rate"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    model.train()
    for epoch in range(cfg["epochs"]):
        total_loss = 0
        for batch in loader:
            batch = {k: v.to(device) for k,v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{cfg['epochs']} — loss: {total_loss/len(loader):.4f}")

    # Save
    model.save_pretrained(cfg["output_dir"])
    print(f"Model saved to {cfg['output_dir']}")

if __name__ == "__main__":
    train()

