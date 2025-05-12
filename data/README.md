# privacy-incident-classifier/data

This folder holds the ticket dataset used to train the incident classifier.

## Download

Using Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download tobiasbueck/multilingual-customer-support-tickets \
  --unzip -p privacy-incident-classifier/data/
