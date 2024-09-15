import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from constants import entity_unit_map
from utils import download_images

class ProductImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.group_ids = {g: i for i, g in enumerate(self.data['group_id'].unique())}
        self.entity_names = {e: i for i, e in enumerate(self.data['entity_name'].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, os.path.basename(self.data.iloc[idx, 1]))
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error reading image {img_name}: {e}")
            image = Image.new('RGB', (224, 224))
        
        group_id = self.group_ids[self.data.iloc[idx, 2]]
        entity_name = self.entity_names[self.data.iloc[idx, 3]]
        
        if 'entity_value' in self.data.columns:
            entity_value = self.data.iloc[idx, 4]
            value, unit = self.parse_entity_value(entity_value)
        else:
            value, unit = 0, ''  # For test set
        
        if self.transform:
            image = self.transform(image)
        
        return image, group_id, entity_name, value, unit

    @staticmethod
    def parse_entity_value(entity_value):
        if pd.isna(entity_value) or entity_value == '':
            return 0, ''
        parts = entity_value.split()
        if len(parts) != 2:
            return 0, ''
        try:
            value = float(parts[0])
            unit = parts[1]
            return value, unit
        except ValueError:
            return 0, ''

class EntityExtractor(nn.Module):
    def __init__(self, num_group_ids, num_entity_names):
        super(EntityExtractor, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the last fully connected layer
        
        self.group_embedding = nn.Embedding(num_group_ids, 50)
        self.entity_embedding = nn.Embedding(num_entity_names, 50)
        
        self.fc = nn.Sequential(
            nn.Linear(2048 + 100, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.unit_classifier = nn.Linear(2048 + 100, len(entity_unit_map))

    def forward(self, image, group_id, entity_name):
        img_features = self.cnn(image)
        group_emb = self.group_embedding(group_id)
        entity_emb = self.entity_embedding(entity_name)
        
        combined = torch.cat((img_features, group_emb, entity_emb), dim=1)
        value_output = self.fc(combined)
        unit_output = self.unit_classifier(combined)
        return value_output, unit_output

def train(model, train_loader, criterion, optimizer, device, patience=3):
    model.train()
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            # Training code
            pass
        
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print("Early stopping triggered")
            break


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            images, group_ids, entity_names, values, units = [b.to(device) for b in batch]
            
            value_outputs, unit_outputs = model(images, group_ids, entity_names)
            
            value_loss = criterion(value_outputs.squeeze(), values.float())
            unit_loss = nn.CrossEntropyLoss()(unit_outputs, units)
            loss = value_loss + unit_loss
            
            total_loss += loss.item()
    return total_loss / len(val_loader)

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            images, group_ids, entity_names = [b.to(device) for b in batch[:3]]
            
            value_outputs, unit_outputs = model(images, group_ids, entity_names)
            value_preds = value_outputs.squeeze().cpu().numpy()
            unit_preds = torch.argmax(unit_outputs, dim=1).cpu().numpy()
            
            predictions.extend(list(zip(value_preds, unit_preds)))
    return predictions

def post_process(predictions, entity_names):
    processed = []
    for (value, unit_idx), entity_name in zip(predictions, entity_names):
        if value <= 0:
            processed.append("")
        else:
            units = list(entity_unit_map[entity_name])
            unit = units[unit_idx % len(units)]
            processed.append(f"{value:.2f} {unit}")
    return processed

def process_batch(batch_df, image_dir, batch_size=32):
    # Download images for this batch
    download_images(batch_df['image_link'], image_dir)
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ProductImageDataset(batch_df, image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Split data
    train_data, val_data = train_test_split(batch_df, test_size=0.2)
    
    # Initialize and train model
    model = EntityExtractor(num_group_ids=len(batch_df['group_id'].unique()), 
                            num_entity_names=len(batch_df['entity_name'].unique()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Train
    for epoch in range(10):  # Adjust number of epochs as needed
        train(model, dataloader, criterion, optimizer, device)
    
    # Predict
    val_loader = DataLoader(ProductImageDataset(val_data, image_dir, transform=transform), 
                            batch_size=batch_size)
    predictions = predict(model, val_loader, device)
    
    # Post-process predictions
    processed_predictions = post_process(predictions, val_data['entity_name'])
    
    # Calculate F1 score
    f1 = f1_score(val_data['entity_value'], processed_predictions, average='weighted')
    
    return model, f1

def main():
    DATASET_FOLDER = '../dataset/'
    IMAGE_DIR = '../images/'
    BATCH_SIZE = 1000
    
    train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    
    # Split data into batches
    batches = [train_df[i:i+BATCH_SIZE] for i in range(0, len(train_df), BATCH_SIZE)]
    
    results = []
    best_model = None
    best_f1 = 0
    
    # Process batches in parallel
    with ProcessPoolExecutor() as executor:
        future_to_batch = {executor.submit(process_batch, batch, IMAGE_DIR): i 
                           for i, batch in enumerate(batches)}
        
        for future in tqdm(future_to_batch):
            batch_index = future_to_batch[future]
            try:
                model, f1 = future.result()
                results.append((batch_index, f1))
                if f1 > best_f1:
                    best_model = model
                    best_f1 = f1
            except Exception as e:
                print(f'Batch {batch_index} generated an exception: {e}')
    
    # Sort results by F1 score
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("Batch processing complete. Best F1 score:", best_f1)
    
    # Use the best model to make predictions on the test set
    test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = ProductImageDataset(test_df, IMAGE_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    test_predictions = predict(best_model, test_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    processed_test_predictions = post_process(test_predictions, test_df['entity_name'])
    
    # Create submission file
    submission = pd.DataFrame({
        'index': test_df['index'],
        'prediction': processed_test_predictions
    })
    submission.to_csv('../submission.csv', index=False)
    
    # Run sanity check
    os.system('python sanity.py --test_filename ../dataset/test.csv --output_filename ../submission.csv')

if __name__ == "__main__":
    main()