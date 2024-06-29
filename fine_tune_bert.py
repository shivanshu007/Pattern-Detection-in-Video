import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd

# Load dataset from CSV
data = pd.read_csv("data.csv")
dataset = Dataset.from_pandas(data)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split dataset
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Load model and move to CPU
device = torch.device("cpu")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train model
trainer.train()

# Save model
model.save_pretrained('fine_tuned_bert')
tokenizer.save_pretrained('fine_tuned_bert')

# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
#
# class NumberDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length=128):
#         self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_length)
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item
#
#
# # Example texts and labels (replace with your actual data)
# texts = ["1234567890", "hello world", "0987654321", "no digits here"]
# labels = [1, 0, 1, 0]
#
# # Split data into train and test sets
# train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# train_dataset = NumberDataset(train_texts, train_labels, tokenizer)
# val_dataset = NumberDataset(val_texts, val_labels, tokenizer)
#
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=2)
#
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
#
# optimizer = AdamW(model.parameters(), lr=5e-5)
# num_epochs = 3
# num_training_steps = num_epochs * len(train_loader)
# lr_scheduler = get_scheduler(
#     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
# )
#
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)
#
# model.train()
# for epoch in range(num_epochs):
#     for batch in train_loader:
#         inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
#         labels = batch['labels'].to(device)
#
#         outputs = model(**inputs, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#
# model.eval()
# predictions = []
# true_labels = []
# for batch in val_loader:
#     inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
#     labels = batch['labels']
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#     logits = outputs.logits
#     predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
#     true_labels.extend(labels)
#
# accuracy = accuracy_score(true_labels, predictions)
# print(f"Validation Accuracy: {accuracy}")
#
# # Save the trained model
# torch.save(model.state_dict(), 'fine_tuned_bert.pth')
