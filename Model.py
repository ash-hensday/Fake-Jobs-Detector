from transformers import AlbertTokenizer, AlbertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")  # Assumes inputs are dictionaries
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(model.device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class LLMModel:
    def __init__(self, model_name='albert-base-v2', num_labels=2, class_weights=None):
        self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        self.model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, 
                                                                     problem_type="single_label_classification",
                                                                     hidden_dropout_prob=0.1)
        self.class_weights = class_weights

    def tokenize_data(self, data, max_length=512):
        return self.tokenizer(data, truncation=True, padding=False, return_tensors='pt', max_length=max_length)
    
    def prepare_dataset(self, X, y, test_size=0.2, val_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, stratify=y_train, random_state=42)
        class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        train_encodings = self.tokenize_data(X_train)
        test_encodings = self.tokenize_data(X_test)
        val_encodings = self.tokenize_data(X_val)
        return train_encodings, y_train, test_encodings, y_test, val_encodings, y_val
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        return {"accuracy": accuracy, "f1": f1}

    def train_model(self, train_encodings, train_labels, val_encodings, val_labels, output_dir='./model_output'):
        
        train_dataset = torch.utils.data.TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.tensor(train_labels, dtype=torch.long)
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
            num_train_epochs=5,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            fp16=True,
            device='cuda',
            evaluation_strategy="epoch" if val_encodings and val_labels else "no",
            save_strategy="epoch",
            logging_dir='./logs',
            logging_steps=100,
            learning_rate=2e-5,
            weight_decay=0.01,
            load_best_model_at_end=bool(val_encodings and val_labels),
            metric_for_best_model="f1" if val_encodings and val_labels else None
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = CustomTrainer(
            class_weights=self.class_weights,
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