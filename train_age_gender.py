from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np
from torchaudio.functional import resample
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from model import ECAPA_gender_age
import torchaudio
import wavencoder
import random
import glob
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
def collate_fn(batch):
    wavs, ages, genders = zip(*batch)

    # wavs는 각 오디오 텐서가 (1, mel_bins, frames) 형태로 되어있음
    # frames 축을 기준으로 가장 긴 샘플에 맞춰 패딩
    max_frames = max(wav.shape[2] for wav in wavs)  # frames 길이
    wavs_padded = []
    
    for wav in wavs:
        pad_size = max_frames - wav.shape[2]
        wav_padded = torch.nn.functional.pad(wav, (0, pad_size), value=0.0)  # 오른쪽으로 패딩
        wavs_padded.append(wav_padded)
    
    # 텐서를 쌓아 최종 배치 (batch_size, 1, mel_bins, max_frames)
    wavs = torch.stack(wavs_padded).squeeze(1)
    
    # 나이와 성별은 텐서로 변환
    ages = torch.stack(ages)
    genders = torch.stack(genders)
    
    return wavs, ages, genders

def get_ext_files(wav_folder, is_train=False,ext='wav'):
    validation_files = []
    if is_train:
        for root, dirs, files in os.walk(wav_folder):
            for file in files:
                if file.endswith(f'.{ext}') and "VALIDATION" not in os.path.join(root, file):
                    validation_files.append(os.path.join(root, file))
    else:
        for root, dirs, files in os.walk(wav_folder):
            for file in files:
                if file.endswith(f'.{ext}') and "VALIDATION" in os.path.join(root, file):
                    validation_files.append(os.path.join(root, file))
    return validation_files

def crop_audio(audio, sample_rate=16000, crop_length=1.0):
    # Crop audio to the specified length (1 second)
    num_samples = int(sample_rate * crop_length)
    if audio.size(1) > num_samples:
        start = random.randint(0, audio.size(1) - num_samples)
        return audio[:, start:start + num_samples]
    else:
        return audio

class KoreanDataset(Dataset):
    def __init__(self,
    wav_folder="/home/jungji/speaker_attribute/speaker_age_estimation_ssl_study/New_Sample-2",
    is_train=True,
    ):
        self.wav_files=sorted(get_ext_files(wav_folder,is_train))
        self.ages=[float(os.path.basename(filename).split('_')[4]) for filename in self.wav_files]
        self.mean_age = np.mean(self.ages)
        self.std_age = np.std(self.ages)
        self.gender_dict = {'M' : 0.0, 'F' : 1.0}
        
    def get_attribute_from_filename(self, filename):
        age=float(os.path.basename(filename).split('_')[4])
        gender=self.gender_dict[os.path.basename(filename).split('_')[2]]
        return age,gender
    def __len__(self):
        return len(self.wav_files)
    def load_audio(self, path : str) -> torch.Tensor:
        audio, sr = torchaudio.load(path)
        if sr != 16000:
            audio = resample(audio, sr, 16000)
        return audio
    def logtorchfbank(self, x: torch.Tensor) -> torch.Tensor:
        # Preemphasis
        flipped_filter = torch.FloatTensor([-0.97, 1.]).unsqueeze(0).unsqueeze(0)
        x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), 'reflect')
        x = F.conv1d(x, flipped_filter).squeeze(1)

        # Melspectrogram
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
            f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80
        )
        mel_spectrogram = mel_spectrogram  # MelSpectrogram 객체가 GPU에서 동작하도록 수정
        x = mel_spectrogram(x) + 1e-6

        # Log and normalize
        x = x.log()
        x = x - torch.mean(x, dim=-1, keepdim=True)
        return x
    def __getitem__(self, idx):
        file = self.wav_files[idx]
        age,gender=self.get_attribute_from_filename(file)
        wav = crop_audio(self.load_audio(file))
        wav = self.logtorchfbank(wav) #(1,80,frames)
        age = (age - self.mean_age)/self.std_age
        
        return wav,torch.tensor(age),torch.tensor(gender)
def return_model_dict(model,weights_path="checkpoints/gender_classifier.model"):
    state_dict = torch.load(weights_path)
    filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc7.')}
    model_dict = model.state_dict()
    model_dict.update(filtered_dict)
    return model_dict
def save_model(model, epoch, save_dir="/home/jungji/speaker_attribute/speaker_age_estimation_ssl_study/jungji/log/spnet"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
   
def loss_fn(age_output, age_target, gender_output, gender_target):
    # 나이 예측에 대해 Mean Absolute Error 사용
    age_loss = F.l1_loss(age_output, age_target)

    # 성별 예측에 대해 Soft Margin Loss 사용
    # 성별 타겟이 0 (남성) 또는 1 (여성)인 경우를 -1, +1로 변환하여 사용
    gender_target_transformed = (gender_target * 2) - 1  # 0, 1 -> -1, +1
    gender_loss = F.soft_margin_loss(gender_output, gender_target_transformed)
    
    return age_loss,gender_loss
# 학습 루프 정의
def train(model, dataloader, val_dataloader, optimizer, num_epochs=10):
    model.train()  # 학습 모드 설정
    
    # TensorBoard SummaryWriter 초기화
    writer = SummaryWriter(log_dir="logs/train")

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        total_loss = 0.0
        for batch_idx, (wav, age_target, gender_target) in enumerate(dataloader):
            
            age_output, gender_output = model(wav)
            
            age_loss, gender_loss = loss_fn(age_output, age_target, gender_output, gender_target)
            loss = age_loss + gender_loss  # 총 손실
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
            
            # TensorBoard에 손실 기록 (매 배치마다)
            writer.add_scalar("Loss/Total", loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar("Loss/Age", age_loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar("Loss/Gender", gender_loss.item(), epoch * len(dataloader) + batch_idx)
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}, Age Loss: {age_loss.item():.4f}, Gender Loss: {gender_loss.item():.4f}")
        
        # 에포크당 평균 학습 손실 출력 및 TensorBoard에 기록
        avg_train_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Loss/Average_Train", avg_train_loss, epoch)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (wav, age_target, gender_target) in enumerate(val_dataloader):
                wav, age_target, gender_target = wav.cuda(), age_target.cuda(), gender_target.cuda()
                age_output, gender_output = model(wav)
                age_loss, gender_loss = loss_fn(age_output, age_target, gender_output, gender_target)
                loss = age_loss + gender_loss
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Validation Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/Average_Validation", avg_val_loss, epoch)

        # 모델 저장
        save_model(model, epoch + 1)
    
    writer.close()  # Tens
from torch.utils.data import DataLoader
if __name__=="__main__":    
    accelerator = Accelerator()
    train_dataset = KoreanDataset(
        wav_folder="sample_korean_data",
        is_train=True
    )
    val_dataset = KoreanDataset(
        wav_folder="sample_korean_data",
        is_train=False
    )

    # DataLoader 설정
    model=ECAPA_gender_age()
    model_dict=return_model_dict(model)
    model.load_state_dict(model_dict)

    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=4, collate_fn=collate_fn)
    #model=model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    train_dataloader, val_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, val_dataloader, model, optimizer
    )
    model.train()  # Set model to training mode

    train(model, train_dataloader,val_dataloader, optimizer, num_epochs=10)
    