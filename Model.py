from transformers import AlbertTokenizer, AlbertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np

class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class LLMModel:
    def __init__(self, model_name='albert-base-v2', num_labels=2, class_weights=None):
        self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        self.model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, 
                                                                     problem_type="single_label_classification",
                                                                     hidden_dropout_prob=0.1)
        if class_weights is not None:
            self.model.loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

    def tokenize_data(self, data, max_length=512):
        return self.tokenizer(data, truncation=True, return_tensors='pt')

    def prepare_dataset(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        train_encodings = self.tokenize_data(X_train)
        test_encodings = self.tokenize_data(X_test)
        return train_encodings, y_train, test_encodings, y_test

    def train_model(self, train_encodings, train_labels, val_encodings=None, val_labels=None, output_dir='./model_output'):
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.tensor(train_labels)
        )
        
        val_dataset = None
        if val_encodings and val_labels:
            val_dataset = torch.utils.data.TensorDataset(
                val_encodings['input_ids'],
                val_encodings['attention_mask'],
                torch.tensor(val_labels)
            )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,  # Increased epochs
            per_device_train_batch_size=16,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir='./logs',
            logging_steps=100,
            learning_rate=2e-5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = CustomTrainer(
            class_weights=torch.tensor(self.class_weights, dtype=torch.float) if self.class_weights else None,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
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