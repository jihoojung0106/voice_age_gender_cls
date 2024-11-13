import torch
from torch.utils.data import DataLoader
import numpy as np
from train_age_gender import KoreanDataset, collate_fn,val_collate_fn,TimitDataset
from model import ECAPA_gender_age
import csv
from collections import defaultdict
import os
# Inference function for age and gender prediction
def inference(model, dataloader):
    model.eval()  # Set model to evaluation mode
    age_predictions = []
    gender_predictions = []
    
    total_age_error = 0.0
    correct_gender_predictions = 0
    total_samples = 0
    results_by_id = defaultdict(list)

    with torch.no_grad():  # Disable gradient computation for inference
        for batch_idx, (wav, age_target, gender_target, file_name) in enumerate(dataloader):
            wav = wav.cuda()  # Move data to GPU if available
            
            # Predict age and gender
            age_output, gender_output = model(wav)
            
            # Post-process age and gender outputs
            predicted_ages = age_output.cpu().numpy() * val_dataset.std_age + val_dataset.mean_age
            predicted_genders = (torch.sigmoid(gender_output).cpu().numpy() > 0.5).astype(int)
            predicted_genders_logit = gender_output.cpu().numpy()
            
            # 실제 타겟 값도 CPU로 이동 및 변환
            age_target = (age_target.cpu().numpy() * val_dataset.std_age) + val_dataset.mean_age
            gender_target = gender_target.cpu().numpy()
            
            # 각 파일별로 예측 및 실제 값 기록
            for i in range(len(file_name)):
                # ID 추출
                if "timit" in file_name[i]:
                    id_ = os.path.basename(file_name[i]).split("_")[0][1:]
                elif "남여" in file_name[i]:
                    id_ = os.path.basename(file_name[i]).split("_")[3]
                else:
                    id_ = os.path.basename(file_name[i])  # Default to full basename

                # 계산 결과
                age_error = abs(predicted_ages[i] - age_target[i])
                gender_correct = int(predicted_genders[i] == gender_target[i])

                # ID별로 데이터를 추가
                results_by_id[id_].append({
                    "file_name": file_name[i],
                    "predicted_age": predicted_ages[i],
                    "true_age": age_target[i],
                    "predicted_gender": predicted_genders[i],
                    "gender_logit": predicted_genders_logit[i],
                    "true_gender": gender_target[i],
                    "age_error": age_error,
                    "gender_correct": gender_correct
                })

    with open("predictions_grouped_by_id.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # CSV 파일의 헤더 작성
        csv_writer.writerow([
            "ID", "File Name", "Predicted Age", "True Age", "Predicted Gender", 
            "Gender Logit", "True Gender", "Age Error", "Gender Correct", 
            "Average Predicted Age", "Average Gender Logit", "Average Age Error"
        ])

        for id_, records in results_by_id.items():
            # 평균 계산
            avg_predicted_age = sum(r["predicted_age"] for r in records) / len(records)
            avg_gender_logit = sum(r["gender_logit"] for r in records) / len(records)
            avg_age_error = sum(r["age_error"] for r in records) / len(records)
            
            # ID별 각 데이터와 평균값 기록
            file_names = [r["file_name"] for r in records]
            predicted_ages = [f"{r['predicted_age']:.2f}" for r in records]
            true_ages = [f"{r['true_age']:.2f}" for r in records]
            predicted_genders = [r["predicted_gender"] for r in records]
            gender_logits = [f"{r['gender_logit']:.2f}" for r in records]
            true_genders = [r["true_gender"] for r in records]
            age_errors = [f"{r['age_error']:.2f}" for r in records]
            gender_corrects = [r["gender_correct"] for r in records]

            # ID별 데이터와 평균값 기록
            csv_writer.writerow([
                id_,
                ", ".join(file_names),
                ", ".join(predicted_ages),
                ", ".join(true_ages),
                ", ".join(map(str, predicted_genders)),
                ", ".join(gender_logits),
                ", ".join(map(str, true_genders)),
                ", ".join(age_errors),
                ", ".join(map(str, gender_corrects)),
                f"{avg_predicted_age:.2f}",  # 평균 예측 나이
                f"{avg_gender_logit:.2f}",   # 평균 성별 로그값
                f"{avg_age_error:.2f}"      # 평균 나이 오차
            ])

    # CSV 파일에 전체 성능 기록
    with open("predictions_per_file.csv", "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([])
        csv_writer.writerow(["Overall Performance"])
        csv_writer.writerow(["Average Age Error", f"{avg_age_error:.2f}"])
        #csv_writer.writerow(["Gender Accuracy (%)", f"{gender_accuracy:.2f}"])
            
    return age_predictions, gender_predictions
def load_resume_path(ckpt_path):
    state_dict = torch.load(ckpt_path)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # "module." 제거
        new_state_dict[name] = v
    return new_state_dict
# Example usage of inference function
if __name__ == "__main__":
    # Load model and weights
    model = ECAPA_gender_age().cuda()
    new_state_dict=load_resume_path("/home/jungji/speaker_attribute/spnet/jungji/log/spnet/ver1/model_epoch_10_.pth")
    model.load_state_dict(new_state_dict)
    # Set up validation dataset and dataloader for inference
    val_dataset = KoreanDataset(
        wav_folder=["/mnt/bear2/zipped_datasets/korean_conversation_age/001.자유대화 음성(일반남녀)/Validation","/mnt/bear2/zipped_datasets/korean_conversation_age/003.자유대화(소아남여,_유아_등_혼합)/01.데이터/2.Validation","/mnt/bear2/zipped_datasets/korean_conversation_age/002.자유대화(노인남여)/01.데이터/2.Validation"],
        is_train=False
    )
    #val_dataset=TimitDataset()
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=val_collate_fn)
    
    # Run inference
    age_predictions, gender_predictions = inference(model, val_dataloader)

    # Print aggregated results
    print("Age Predictions:", age_predictions)
    print("Gender Predictions:", gender_predictions)
