from transformers import AlbertTokenizer, AlbertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

class LLMModel:
    def __init__(self, model_name='albert-base-v2', num_labels=2):
        self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        self.model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def tokenize_data(self, data, max_length=512):
        return self.tokenizer(data, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

    def prepare_dataset(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        train_encodings = self.tokenize_data(X_train)
        test_encodings = self.tokenize_data(X_test)
        return train_encodings, y_train, test_encodings, y_test

    def train_model(self, train_encodings, train_labels, output_dir='./model_output'):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=True
        )
        train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )
        trainer.train()

    def predict(self, inputs):
        inputs = self.tokenize_data(inputs)
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.cpu().numpy()

    def save_model(self, path='./fine_tuned_model'):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)