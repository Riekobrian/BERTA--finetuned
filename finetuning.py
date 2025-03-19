from transformers import BertTokenizer, BertForSequenceClassification

# Example: Downloading a BERT model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)  # Example: binary classification
