from torch.utils.data import Dataset
import os
from torch.utils.data.sampler import WeightedRandomSampler

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
import pandas as pd


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # GPU ID를 지정하세요. 예: "0,1"은 첫 번째와 두 번째 GPU 사용

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
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
def val_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    wavs, ages, genders,file_names = zip(*batch)

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
    
    return wavs, ages, genders,file_names

def get_ext_files(wav_folder, is_train=False,ext='wav'):
    if type(wav_folder) == list:
        validation_files = []
        for folder in wav_folder:
            cnt=0
            if is_train:
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        if random.random() < 0.9995:
                            continue
                        if file.endswith(f'.{ext}') and "Validation" not in os.path.join(root, file):
                            validation_files.append(os.path.join(root, file))
                            cnt+=1
                            if cnt >= NUM:
                                 break  # Inner loop break
                    if cnt >= NUM:
                        break
            else:
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        if random.random()< 0.999:
                            continue
                        if file.endswith(f'.{ext}') and "Validation" in os.path.join(root, file):
                            validation_files.append(os.path.join(root, file))
                            cnt+=1
                            if cnt >= 3000:
                                 break  # Inner loop break
                    if cnt >= 3000:
                        break  # Outer loop break
        
    else:
        validation_files = []
        if is_train:
            for root, dirs, files in os.walk(wav_folder):
                for file in files:
                    if file.endswith(f'.{ext}') and "Validation" not in os.path.join(root, file) and "수도" in os.path.join(root, file):
                        validation_files.append(os.path.join(root, file))
        else:
            for root, dirs, files in os.walk(wav_folder):
                for file in files:
                    if file.endswith(f'.{ext}') and "Validation" in os.path.join(root, file):
                        validation_files.append(os.path.join(root, file))
                        if len(validation_files) >= 10000:
                            break
    if is_train:
        filename="Training_file_list.txt"
    else:
        filename="Validation_file_list.txt"
    with open(filename, "w") as f:
        for file in files:
            f.write(file + "\n")                    
    return validation_files

def get_files(wav_folder,is_train=False,file_name=None):
    # if is_train:
    #     NUM=500000
    # else:
    #     NUM=3000
    selected_files = []
    for root, dirs, files in os.walk(wav_folder):
        for file in files:
            if file.endswith(f'.wav'):
                selected_files.append(os.path.join(root, file))
        #     if len(selected_files) >= NUM:
        #         break
        # if len(selected_files) >= NUM:
        #         break
            
    with open(file_name, "w") as f:
        for file in selected_files:
            f.write(file + "\n") 
    #print(f"Number of files: {len(selected_files)}, {file_name}")                   
    return selected_files




def crop_audio(audio, sample_rate=16000, crop_length=1.0):
    # Crop audio to the specified length (1 second)
    num_samples = int(sample_rate * crop_length)
    if audio.size(1) > num_samples:
        start = random.randint(0, audio.size(1) - num_samples)
        return audio[:, start:start + num_samples]
    else:
        padded_audio = F.pad(audio, (0, num_samples - audio.size(1)))
        return padded_audio
def read_lines_to_list(file_path):
    lines = []
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]
def select_random_samples(data_list, num_samples=100000):
    return random.sample(data_list, min(num_samples, len(data_list)))
def read_wav_files(folder_path):
    wav_files = [
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if f.endswith(".WAV") or f.endswith(".wav")
    ]
    return wav_files
def create_balanced_dataloader(is_train, batch_size=16, num_workers=4):
    dataset = MozillaDataset(is_train=is_train)
    sampler = create_balanced_sampler(dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn  # None 제거
    )
    return dataloader

def create_balanced_sampler(dataset):
    # gender_age별 샘플 개수 계산
    class_counts = dataset.data_label['gender_age'].value_counts().to_dict()
    class_weights = {label: 1.0 / count for label, count in class_counts.items()}
    
    # 각 샘플의 가중치 계산
    sample_weights = dataset.data_label['gender_age'].map(class_weights).values
    
    # WeightedRandomSampler 생성
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    return sampler
class TimitDataset(Dataset):
    def __init__(self,
    wav_folder="/home/jungji/speaker_attribute/speaker_age_estimation_ssl_study/New_Sample-2",
    is_train=False,
    ):  
        #self.wav_files=[]
        self.wav_files = read_wav_files("/mnt/lynx2/datasets/timit/wav_data/TRAIN")
        data = pd.read_csv("/mnt/lynx2/datasets/timit/data_info_height_age.csv", usecols=["ID", "Sex", "age"])
        self.data_dict = data.set_index("ID").T.to_dict()
        if not os.path.exists("age_statistics.txt"):
            self.ages=[float(os.path.basename(filename).split('_')[4]) for filename in self.wav_files]
            self.mean_age = np.mean(self.ages)
            self.std_age = np.std(self.ages)
            with open("age_statistics.txt", "w") as f:
                f.write(f"mean_age: {self.mean_age}\n")
                f.write(f"std_age: {self.std_age}\n")
        else:
            with open("age_statistics.txt", "r") as f:
                lines = f.readlines()
                self.mean_age = float(lines[0].split(": ")[1].strip())
                self.std_age = float(lines[1].split(": ")[1].strip())
        
        self.gender_dict = {'M' : 0.0, 'F' : 1.0}
        self.is_train=is_train  
    def get_attribute_from_filename(self, filename):
        id_=os.path.basename(filename).split("_")[0][1:]
        age=float(self.data_dict[id_]["age"])
        gender=self.gender_dict[self.data_dict[id_]["Sex"]]
        return age,gender
    def __len__(self):
        return len(self.wav_files)
    def load_audio(self, path: str) -> torch.Tensor:
        try:
            audio, sr = torchaudio.load(path)
            if sr != 16000:
                audio = resample(audio, sr, 16000)
            return audio
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None  # 오류가 있는 파일은 None을 반환하여 건너뜁니다.
    def logtorchfbank(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == 0:
            return None

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
        
        try:
            wav = self.load_audio(file)
            if wav is None:
                return None
            wav = crop_audio(wav)
            wav = self.logtorchfbank(wav)  # (1, 80, frames)
        except ValueError as e:
            # 길이가 0인 샘플을 건너뛰기 위해 None을 반환하거나 다른 처리를 할 수 있습니다.
            print(f"Skipping sample {file} due to error: {e}")
            return None  # None을 반환하여 나중에 DataLoader에서 이 샘플을 무시할 수 있도록 합니다.

        age = (age - self.mean_age)/self.std_age
        if self.is_train:
            return wav,torch.tensor(age),torch.tensor(gender)
        else:
            return wav,torch.tensor(age),torch.tensor(gender),file


class MozillaDataset(Dataset):
    def __init__(self,
    is_train=False,
    ):  
        #self.wav_files=[]
        
        ##self.wav_files = read_wav_files("/mnt/lynx1/datasets/CommonVoice_16k/cv_16k")
        if is_train:
            self.data = pd.read_csv("mozilla_csv/train_filtered.csv", usecols=["path", "gender", "age"])
            path_list = self.data["path"].tolist()
            if os.path.exists("mozilla_train_files.txt"):
                self.wav_files=read_lines_to_list("mozilla_train_files.txt")
            else:
                self.wav_files = [f"/mnt/lynx1/datasets/CommonVoice_16k/cv_16k/{path.replace('.mp3','.wav')}" for path in path_list if os.path.exists(f"/mnt/lynx1/datasets/CommonVoice_16k/cv_16k/{path.replace('.mp3','.wav')}")]
        else:
            self.data = pd.read_csv("mozilla_csv/test_filtered.csv", usecols=["path", "gender", "age"])
            path_list = self.data["path"].tolist()
            self.wav_files = [f"/mnt/lynx1/datasets/CommonVoice_16k/cv_16k/{path.replace('.mp3','.wav')}" for path in path_list if os.path.exists(f"/mnt/lynx1/datasets/CommonVoice_16k/cv_16k/{path.replace('.mp3','.wav')}")]
        self.data_dict = self.data.set_index("path").T.to_dict()
        print(f"Mozilla Number of files: {len(self.wav_files)}")
        with open("age_statistics.txt", "r") as f:
            lines = f.readlines()
            self.mean_age = float(lines[0].split(": ")[1].strip())
            self.std_age = float(lines[1].split(": ")[1].strip())
        
        self.gender_dict = {'male' : 0.0, 'female' : 1.0}
        self.age_dict={
            'nineties':random.randint(90,99),
            'eighties':random.randint(80,89),
            'seventies':random.randint(70,79),
            'sixties':random.randint(60,69),
            'fifties':random.randint(50,59),
            'fourties':random.randint(40,49),
            'thirties':random.randint(30,39),
            'twenties':random.randint(20,29),
            'teens':random.randint(10,19),
        }
        self.gender_age_dict = {
            "twenties_male":179113,
            "thirties_male":97950,
            "fourties_male":80935,
            "twenties_female":72160,
            "teens_male":34266,
            "fifties_male":33172,
            "sixties_female":31342,
            "sixties_male":25838,
            "fourties_female":19666,
            "teens_female":19110,
            "thirties_female":19075,
            "fifties_female":16016,
            "seventies_male":3201,
            "seventies_female":2212,
            "eighties_male":868,
            "eighties_female":123,
            "nineties_male":53,
        }
        self.is_train=is_train  
        if is_train:
            total_samples = sum(self.gender_age_dict.values())
            self.weights = []
            for file in self.wav_files:
                id_ = os.path.basename(file).replace(".wav", ".mp3")
                gender = self.data_dict[id_]["gender"]
                age = self.data_dict[id_]["age"]
                key = f"{age}_{gender}"
                weight = total_samples / self.gender_age_dict[key]
                self.weights.append(weight)
            
            self.sampler = WeightedRandomSampler(self.weights, len(self.wav_files))

    
    def get_attribute_from_filename(self, filename):
        id_=os.path.basename(filename).replace(".wav",".mp3")
        age=float(self.age_dict[self.data_dict[id_]["age"]])
        gender=self.gender_dict[self.data_dict[id_]["gender"]]
        return age,gender
    def __len__(self):
        return len(self.wav_files)
    def load_audio(self, path: str) -> torch.Tensor:
        try:
            audio, sr = torchaudio.load(path)
            if sr != 16000:
                audio = resample(audio, sr, 16000)
            return audio
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None  # 오류가 있는 파일은 None을 반환하여 건너뜁니다.
    def logtorchfbank(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == 0:
            return None

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
        
        try:
            wav = self.load_audio(file)
            if wav is None:
                return None
            wav = crop_audio(wav)
            wav = self.logtorchfbank(wav)  # (1, 80, frames)
        except ValueError as e:
            # 길이가 0인 샘플을 건너뛰기 위해 None을 반환하거나 다른 처리를 할 수 있습니다.
            print(f"Skipping sample {file} due to error: {e}")
            return None  # None을 반환하여 나중에 DataLoader에서 이 샘플을 무시할 수 있도록 합니다.

        age = (age - self.mean_age)/self.std_age
        if self.is_train:
            return wav,torch.tensor(age),torch.tensor(gender)
        else:
            return wav,torch.tensor(age),torch.tensor(gender),file












class KoreanDataset(Dataset):
    def __init__(self,
    wav_folder="/home/jungji/speaker_attribute/speaker_age_estimation_ssl_study/New_Sample-2",
    is_train=True,
    ):  
        #self.wav_files=[]
        if is_train:
            file_0=read_lines_to_list("/home/jungji/speaker_attribute/spnet/file_list_0_train.txt")
            file_0=select_random_samples(file_0)
            file_1=read_lines_to_list("/home/jungji/speaker_attribute/spnet/file_list_1_train.txt")
            file_1=select_random_samples(file_1)
            file_2=read_lines_to_list("/home/jungji/speaker_attribute/spnet/file_list_2_train.txt")
            file_2=select_random_samples(file_2)
        else:
            file_0=read_lines_to_list("/home/jungji/speaker_attribute/spnet/file_list_0_val.txt")
            file_0=select_random_samples(file_0,1000)
            file_1=read_lines_to_list("/home/jungji/speaker_attribute/spnet/file_list_1_val.txt")
            file_1=select_random_samples(file_1,1000)
            file_2=read_lines_to_list("/home/jungji/speaker_attribute/spnet/file_list_2_val.txt")
            file_2=select_random_samples(file_2,1000)
        # for i in range(len(wav_folder)):
        #     wav_files=get_files(wav_folder[i],is_train=is_train,file_name=f"file_list_{i}_{'train' if is_train else 'val'}.txt")
        #     self.wav_files+=wav_files
        self.wav_files=file_0+file_1+file_2
        print("Korean Number of files: ",len(self.wav_files))
        if not os.path.exists("age_statistics.txt"):
            self.ages=[float(os.path.basename(filename).split('_')[4]) for filename in self.wav_files]
            self.mean_age = np.mean(self.ages)
            self.std_age = np.std(self.ages)
            with open("age_statistics.txt", "w") as f:
                f.write(f"mean_age: {self.mean_age}\n")
                f.write(f"std_age: {self.std_age}\n")
        else:
            with open("age_statistics.txt", "r") as f:
                lines = f.readlines()
                self.mean_age = float(lines[0].split(": ")[1].strip())
                self.std_age = float(lines[1].split(": ")[1].strip())
        
        self.gender_dict = {'M' : 0.0, 'F' : 1.0}
        self.is_train=is_train  
    def get_attribute_from_filename(self, filename):
        age=float(os.path.basename(filename).split('_')[4])
        gender=self.gender_dict[os.path.basename(filename).split('_')[2]]
        return age,gender
    def __len__(self):
        return len(self.wav_files)
    def load_audio(self, path: str) -> torch.Tensor:
        try:
            audio, sr = torchaudio.load(path)
            if sr != 16000:
                audio = resample(audio, sr, 16000)
            return audio
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None  # 오류가 있는 파일은 None을 반환하여 건너뜁니다.
    def logtorchfbank(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == 0:
            return None

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
        
        try:
            wav = self.load_audio(file)
            if wav is None:
                return None
            wav = crop_audio(wav)
            wav = self.logtorchfbank(wav)  # (1, 80, frames)
        except ValueError as e:
            # 길이가 0인 샘플을 건너뛰기 위해 None을 반환하거나 다른 처리를 할 수 있습니다.
            print(f"Skipping sample {file} due to error: {e}")
            return None  # None을 반환하여 나중에 DataLoader에서 이 샘플을 무시할 수 있도록 합니다.

        age = (age - self.mean_age)/self.std_age
        if self.is_train:
            return wav,torch.tensor(age),torch.tensor(gender)
        else:
            return wav,torch.tensor(age),torch.tensor(gender),file
def return_model_dict(model,weights_path="checkpoints/gender_classifier.model"):
    state_dict = torch.load(weights_path)
    filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc7.')}
    model_dict = model.state_dict()
    model_dict.update(filtered_dict)
    return model_dict
def load_resume_path(ckpt_path):
    state_dict = torch.load(ckpt_path)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # "module." 제거
        new_state_dict[name] = v
    return new_state_dict
def save_model(model, epoch, save_dir="jungji/log/spnet"):
    os.makedirs(save_dir, exist_ok=True)
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
def train(model, dataloader, val_dataloader, en_train_dataloader,mozilla_train_dataloader,optimizer, num_epochs=10,accelerator=None):
    model.train()  # 학습 모드 설정
    
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir="logs/train")

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        total_loss = 0.0
        for batch_idx, (wav, age_target, gender_target) in enumerate(mozilla_train_dataloader):
            if wav.ndim==4:
                wav=wav.squeeze(1)
            age_output, gender_output = model(wav)
            
            age_loss, gender_loss = loss_fn(age_output, age_target, gender_output, gender_target)
            loss = age_loss + gender_loss  # 총 손실
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
            
            if accelerator.is_main_process:
                writer.add_scalar("Loss/Total_En", loss.item(), epoch * len(mozilla_train_dataloader) + batch_idx)
                writer.add_scalar("Loss/Age_En", age_loss.item(), epoch * len(mozilla_train_dataloader) + batch_idx)
                writer.add_scalar("Loss/Gender_En", gender_loss.item(), epoch * len(mozilla_train_dataloader) + batch_idx)
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(mozilla_train_dataloader)}], "
                        f"Loss: {loss.item():.4f}, Age Loss: {age_loss.item():.4f}, Gender Loss: {gender_loss.item():.4f}")
        
        for batch_idx, (wav, age_target, gender_target) in enumerate(en_train_dataloader):
            if wav.ndim==4:
                wav=wav.squeeze(1)
            age_output, gender_output = model(wav)
            
            age_loss, gender_loss = loss_fn(age_output, age_target, gender_output, gender_target)
            loss = age_loss + gender_loss  # 총 손실
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
            
            if accelerator.is_main_process:
                writer.add_scalar("Loss/Total_En", loss.item(), epoch * len(en_train_dataloader) + batch_idx)
                writer.add_scalar("Loss/Age_En", age_loss.item(), epoch * len(en_train_dataloader) + batch_idx)
                writer.add_scalar("Loss/Gender_En", gender_loss.item(), epoch * len(en_train_dataloader) + batch_idx)
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(en_train_dataloader)}], "
                        f"Loss: {loss.item():.4f}, Age Loss: {age_loss.item():.4f}, Gender Loss: {gender_loss.item():.4f}")
        
        for batch_idx, (wav, age_target, gender_target) in enumerate(dataloader):
            if random.random()<0.5:
                continue
            if wav.ndim==4:
                wav=wav.squeeze(1)
            age_output, gender_output = model(wav)
            
            age_loss, gender_loss = loss_fn(age_output, age_target, gender_output, gender_target)
            loss = age_loss + gender_loss  # 총 손실
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
            
            if accelerator.is_main_process:
                writer.add_scalar("Loss/Total", loss.item(), epoch * len(dataloader) + batch_idx)
                writer.add_scalar("Loss/Age", age_loss.item(), epoch * len(dataloader) + batch_idx)
                writer.add_scalar("Loss/Gender", gender_loss.item(), epoch * len(dataloader) + batch_idx)
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], "
                        f"Loss: {loss.item():.4f}, Age Loss: {age_loss.item():.4f}, Gender Loss: {gender_loss.item():.4f}")
        
        
        avg_train_loss = total_loss / (len(dataloader) + len(en_train_dataloader))
        if accelerator.is_main_process:
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}")
            writer.add_scalar("Loss/Average_Train", avg_train_loss, epoch)

        # Validation loop
        model.eval()
        val_loss = 0.0
        if val_dataloader is not None:
            with torch.no_grad():
                for batch_idx, (wav, age_target, gender_target,filename) in enumerate(val_dataloader):
                    wav, age_target, gender_target = wav.cuda(), age_target.cuda(), gender_target.cuda()
                    age_output, gender_output = model(wav)
                    age_loss, gender_loss = loss_fn(age_output, age_target, gender_output, gender_target)
                    loss = age_loss + gender_loss
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            if accelerator.is_main_process:
                print(f"Epoch [{epoch+1}/{num_epochs}], Average Validation Loss: {avg_val_loss:.4f}")
                writer.add_scalar("Loss/Average_Validation", avg_val_loss, epoch)
        # 모델 저장
        if accelerator.is_main_process:
            save_model(model, epoch + 1)
    if accelerator.is_main_process:
        writer.close()  # Tens
from torch.utils.data import DataLoader
if __name__=="__main__":  
    
      
    accelerator = Accelerator()
    mozilla_train_dataset=MozillaDataset(is_train=True)
    train_dataset = KoreanDataset(
        wav_folder=[
            "/mnt/bear2/zipped_datasets/korean_conversation_age/001.자유대화 음성(일반남녀)/Training",
            "/mnt/bear2/zipped_datasets/korean_conversation_age/002.자유대화(노인남여)/01.데이터/1.Training",
            "/mnt/bear2/zipped_datasets/korean_conversation_age/003.자유대화(소아남여,_유아_등_혼합)/01.데이터/1.Training"
            ],
        is_train=True
    )
    val_dataset = KoreanDataset(
        wav_folder=["/mnt/bear2/zipped_datasets/korean_conversation_age/001.자유대화 음성(일반남녀)/Validation",
                    "/mnt/bear2/zipped_datasets/korean_conversation_age/003.자유대화(소아남여,_유아_등_혼합)/01.데이터/2.Validation",
                    "/mnt/bear2/zipped_datasets/korean_conversation_age/002.자유대화(노인남여)/01.데이터/2.Validation"],
        is_train=False
    )
    en_train_dataset=TimitDataset(is_train=True)
    

    # DataLoader 설정
    model=ECAPA_gender_age()
    resume_path="jungji/log/spnet/ver2/model_epoch_1.pth"
    # model_dict=return_model_dict(model,resume_path=resume_path)
    # model.load_state_dict(model_dict)
    new_state_dict=load_resume_path(resume_path)
    model.load_state_dict(new_state_dict)
    sampler = mozilla_train_dataset.sampler
    mozilla_train_dataloader = DataLoader(mozilla_train_dataset, sampler=sampler, batch_size=256, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, collate_fn=collate_fn)
    en_train_dataloader=DataLoader(en_train_dataset, batch_size=256, shuffle=True, num_workers=4, collate_fn=collate_fn)
    #val_dataloader=None
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=4, collate_fn=val_collate_fn)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    
    if val_dataloader is not None:
        train_dataloader, val_dataloader, en_train_dataloader,mozilla_train_dataloader,model, optimizer = accelerator.prepare(
            train_dataloader, val_dataloader, en_train_dataloader,mozilla_train_dataloader,model, optimizer
        )
    else:
        train_dataloader, model, optimizer = accelerator.prepare(
            train_dataloader, model, optimizer
        )
    model.train()  # Set model to training mode

    train(model, train_dataloader,val_dataloader, en_train_dataloader,mozilla_train_dataloader,optimizer, num_epochs=30,accelerator=accelerator)
    