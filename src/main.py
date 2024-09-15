# In main.py

import os
import pandas as pd
from sklearn.metrics import f1_score
from config import DATASET_FOLDER, OUTPUT_FILE
from batch_processor import process_data_parallel

def calculate_f1_score(true_values, predictions):
    return f1_score(true_values, predictions, average='weighted')

def main():
    # Process training data
    train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    train_predictions = process_data_parallel(train_df)
    
    # Calculate and print F1 score for training data
    train_f1 = calculate_f1_score(train_df['entity_value'], train_predictions)
    print(f"Training F1 Score: {train_f1}")
    
    # Process test data
    test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    test_predictions = process_data_parallel(test_df)
    
    # Create submission file
    submission = pd.DataFrame({
        'index': test_df['index'],
        'prediction': test_predictions
    })
    submission.to_csv(OUTPUT_FILE, index=False)
    
    # Run sanity check
    os.system(f'python sanity.py --test_filename {os.path.join(DATASET_FOLDER, "test.csv")} --output_filename {OUTPUT_FILE}')

if __name__ == "__main__":
    main()