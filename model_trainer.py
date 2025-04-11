from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import pandas as pd

# 1. Load and prepare dataset
true = pd.read_csv('true.csv')
fake = pd.read_csv('fake.csv')

true['label'] = 1  # REAL news
fake['label'] = 0  # FAKE news

df = pd.concat([true, fake])
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle

# Use only the "text" column (can also add 'title' if needed)
texts = df['text'].astype(str)
labels = df['label']

# 2. Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 3. Custom Dataset
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512)
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# 4. Split data into train and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

train_dataset = NewsDataset(train_texts, train_labels)
val_dataset = NewsDataset(val_texts, val_labels)

# 5. Load BERT model for classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 6. Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    save_total_limit=1
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 8. Train the model
trainer.train()

# 9. Save the trained model and tokenizer
model.save_pretrained('./bert_model')
tokenizer.save_pretrained('./bert_model')



