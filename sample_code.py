# import os
# import random
# import pandas as pd

# def predictor(image_link, category_id, entity_name):
#     '''
#     Call your model/approach here
#     '''
#     
#     return "" if random.random() > 0.5 else "10 inch"

# if __name__ == "__main__":
#     DATASET_FOLDER = '../dataset/'
    
#     test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
#     test['prediction'] = test.apply(
#         lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
#     output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
#     test[['index', 'prediction']].to_csv(output_filename, index=False)



    import os
    import pandas as pd
    import torch
    from torchvision import transforms
    from PIL import Image
    from tqdm import tqdm
    from src.model import create_model
    from src.utils import download_images
    from src.constants import entity_unit_map

    def process_batch(batch, model, device, download_folder):
        download_images(batch['image_link'], download_folder, allow_multiprocessing=True)
        
        predictions = []
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        for _, row in batch.iterrows():
            img_path = os.path.join(download_folder, os.path.basename(row['image_link']))
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(img_tensor)
                    prediction = outputs[row['entity_name']].argmax().item()
                    prediction_str = convert_prediction_to_string(prediction, row['entity_name'])
                predictions.append(prediction_str)
            else:
                predictions.append("")
        
        return pd.DataFrame({'index': batch.index, 'prediction': predictions})

    def convert_prediction_to_string(prediction, entity_name):
        # This is a placeholder implementation. You need to implement this based on your encoding scheme.
        unit = next(iter(entity_unit_map[entity_name]))  # Get the first unit for this entity
        return f"{prediction} {unit}"

    def main():
        DATASET_FOLDER = 'dataset'
        DOWNLOAD_FOLDER = 'temp_images'
        BATCH_SIZE = 32
        
        if not os.path.exists(DOWNLOAD_FOLDER):
            os.makedirs(DOWNLOAD_FOLDER)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define your num_classes_dict based on your dataset
        num_classes_dict = {
            'width': 100,
            'depth': 100,
            'height': 100,
            'item_weight': 200,
            'maximum_weight_recommendation': 50,
            'voltage': 30,
            'wattage': 50,
            'item_volume': 150
        }
        
        model = create_model(num_classes_dict).to(device)
        # Load your trained model weights here
        # model.load_state_dict(torch.load('path_to_your_model_weights.pth'))
        model.eval()
        
        test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
        
        results = []
        for i in tqdm(range(0, len(test), BATCH_SIZE)):
            batch = test.iloc[i:i+BATCH_SIZE]
            batch_results = process_batch(batch, model, device, DOWNLOAD_FOLDER)
            results.append(batch_results)
            
            # Clear temporary images
            for file in os.listdir(DOWNLOAD_FOLDER):
                os.remove(os.path.join(DOWNLOAD_FOLDER, file))
        
        final_results = pd.concat(results)
        output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
        final_results.to_csv(output_filename, index=False)
        
        # Run sanity check
        os.system(f"python src/sanity.py --test_filename {os.path.join(DATASET_FOLDER, 'test.csv')} --output_filename {output_filename}")

    if __name__ == "__main__":
        main()