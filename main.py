from Data_Preprocessing import Preprocessor
from Model import LLMModel

def main():
    # Define dataset file paths
    postings_path = "/content/postings.csv"
    fake_postings_path = "/content/Fake Postings.csv"
    industries_path = '/content/industries.csv'
    job_industries_path = '/content/job_industries.csv'
    
    # Initialize preprocessor and prepare data
    preprocessor = Preprocessor(postings_path, fake_postings_path, industries_path, job_industries_path)
    X, y = preprocessor.prepare_data_for_model()
    
    # Initialize model
    model = LLMModel(model_name='albert-base-v2', num_labels=2)
    
    # Prepare dataset for training
    train_encodings, train_labels, test_encodings, test_labels, val_encodings, val_labels = model.prepare_dataset(X, y)
    
    # Train the model
    model.train_model(train_encodings, train_labels, val_encodings, val_labels)
    
    # Save the fine-tuned model