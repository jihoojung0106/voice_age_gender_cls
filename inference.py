from config import TIMITConfig

from argparse import ArgumentParser
from multiprocessing import Pool
import os
import torchaudio
from TIMIT.dataset import TIMITDataset
from TIMIT.lightning_model_uncertainty_loss import LightningModel

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import pytorch_lightning as pl

import torch
import torch.utils.data as data

from tqdm import tqdm 
import pandas as pd
import numpy as np

import torch.nn.utils.rnn as rnn_utils
def collate_fn(batch):
    (seq, age, gender) = zip(*batch)
    seql = [x.reshape(-1,) for x in seq]
    seq_length = [x.shape[0] for x in seql]
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    return data, age, gender, seq_length

if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--data_path', type=str, default="/mnt/lynx2/datasets/timit/wav_data")
    parser.add_argument('--speaker_csv_path', type=str, default="Dataset/data_info_height_age.csv")
    parser.add_argument('--batch_size', type=int, default=TIMITConfig.batch_size)
    parser.add_argument('--epochs', type=int, default=TIMITConfig.epochs)
    parser.add_argument('--num_layers', type=int, default=TIMITConfig.num_layers)
    parser.add_argument('--hidden_state', type=int, default=TIMITConfig.hidden_state)
    parser.add_argument('--feature_dim', type=int, default=TIMITConfig.feature_dim)
    parser.add_argument('--lr', type=float, default=TIMITConfig.lr)
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--n_workers', type=int, default=TIMITConfig.n_workers)
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--model_checkpoint', type=str, default="checkpoints/multi-task_BiEncoder/epoch=20-step=2582.ckpt")
    parser.add_argument('--upstream_model', type=str, default=TIMITConfig.upstream_model)
    parser.add_argument('--model_type', type=str, default=TIMITConfig.model_type)
    
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    # Check device
    if not torch.cuda.is_available():
        device = 'cpu'
        hparams.gpu = 0
    else:        
        device = 'cuda'
        #print(f'Training Model with {hparams.state_number} on TIMIT Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')
    
    # Testing Dataset
    test_set = TIMITDataset(
        wav_folder = os.path.join(hparams.data_path, 'TEST'),
        hparams = hparams,
        is_train=False
    )

    ## Testing Dataloader
    testloader = data.DataLoader(
        test_set, 
        batch_size=1, 
        shuffle=False, 
        num_workers=hparams.n_workers,
        collate_fn = collate_fn,
    )

    csv_path = hparams.speaker_csv_path
    df = pd.read_csv(csv_path)
    a_mean = df[df['Use'] == 'TRN']['age'].mean()
    a_std = df[df['Use'] == 'TRN']['age'].std()

    #Testing the Model
    checkpoint_dir = 'checkpoints/multi-task_BiEncoder'

    # ckpt 파일 이름 가져오기
    ckpt_files = [os.path.join(checkpoint_dir,f) for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]

    if hparams.model_checkpoint:
        model = LightningModel.load_from_checkpoint(ckpt_files[0], HPARAMS=vars(hparams))
        model.to(device)
        model.eval()
        age_pred = []
        age_true = []
        gender_pred = []
        gender_true = []
        wav_path="/mnt/datasets/lip_reading/lrs3/trainval/1JxxCB9GiGU/50001.wav"
        wav, _ = torchaudio.load(wav_path)
        
        if(wav.shape[0] != 1):
            wav = torch.mean(wav, dim=0)
        
        x = wav.to(device)
        x_len=[wav.shape[1]]
        
        y_hat_a, y_hat_g = model(x, x_len)
        y_hat_a = y_hat_a.to('cpu')
        y_hat_g = y_hat_g.to('cpu')
        age_pred=y_hat_a*a_std+a_mean
        gender_pred=y_hat_g>0.5
        
        print(wav_path, age_pred, gender_pred)
        