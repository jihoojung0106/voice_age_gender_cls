import torch
from torch.utils.data import DataLoader
import numpy as np
from train_age_gender import KoreanDataset, collate_fn
from model import ECAPA_gender_age
# Inference function for age and gender prediction
def inference(model, dataloader):
    model.eval()  # Set model to evaluation mode
    age_predictions = []
    gender_predictions = []
    
    with torch.no_grad():  # Disable gradient computation for inference
        for batch_idx, (wav, age_target, gender_target) in enumerate(dataloader):
            wav = wav.cuda()  # Move data to GPU if available
            
            # Predict age and gender
            age_output, gender_output = model(wav)
            
            # Post-process age and gender outputs
            predicted_ages = age_output.cpu().numpy() * val_dataset.std_age + val_dataset.mean_age
            predicted_genders = (torch.sigmoid(gender_output).cpu().numpy() > 0.5).astype(int)
            
            # Collect predictions
            age_predictions.extend(predicted_ages)
            gender_predictions.extend(predicted_genders)

            # Print predictions for this batch
            print(f"Batch {batch_idx+1}:")
            print("Ages:", predicted_ages)
            print("Genders:", predicted_genders)
    
    return age_predictions, gender_predictions

# Example usage of inference function
if __name__ == "__main__":
    # Load model and weights
    model = ECAPA_gender_age().cuda()
    model.load_state_dict(torch.load("checkpoints/multi-task_BiEncoder/epoch=20-step=2582.ckpt"))
    # Set up validation dataset and dataloader for inference
    val_dataset = KoreanDataset(
        wav_folder="sample_korean_data",
        is_train=False
    )
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # Run inference
    age_predictions, gender_predictions = inference(model, val_dataloader)

    # Print aggregated results
    print("Age Predictions:", age_predictions)
    print("Gender Predictions:", gender_predictions)
