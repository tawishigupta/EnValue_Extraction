# In batch_processor.py

import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from utils import download_images
from image_processing import preprocess_image, extract_text
from entity_extraction import extract_entity_value
from config import IMAGE_DIR, BATCH_SIZE

def process_image(row, image_dir):
    image_path = os.path.join(image_dir, os.path.basename(row['image_link']))
    image = preprocess_image(image_path)
    text = extract_text(image)
    prediction = extract_entity_value(text, row['entity_name'])
    return prediction

def process_batch(batch_df, image_dir):
    download_images(batch_df['image_link'], image_dir)
    
    predictions = []
    for _, row in batch_df.iterrows():
        prediction = process_image(row, image_dir)
        predictions.append(prediction)
    
    return predictions

def process_data_parallel(df):
    batches = [df[i:i+BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
    results = []
    
    with ProcessPoolExecutor() as executor:
        future_to_batch = {executor.submit(process_batch, batch, IMAGE_DIR): i 
                           for i, batch in enumerate(batches)}
        
        for future in tqdm(future_to_batch):
            batch_index = future_to_batch[future]
            try:
                predictions = future.result()
                results.append((batch_index, predictions))
            except Exception as e:
                print(f'Batch {batch_index} generated an exception: {e}')
    
    return [pred for _, preds in sorted(results, key=lambda x: x[0]) for pred in preds]