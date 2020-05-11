import soundfile
import pandas as pd
import os
import librosa
import json
import numpy as np
from tqdm import tqdm
import argparse


time_coef = 160 * 4
src = '/home/shalgynov/mnt/sms_finance/'
dst = '/home/shalgynov/speaker_diarization/slice_data/'
df_path = '/home/shalgynov/speaker_diarization/sms_finance/diar_labels.csv'


def slice_audio(df, speaker_id, min_length, n_spk=1):
    try:
        os.makedirs(dst + speaker_id)
    except FileExistsError:
        print('You have already created data for this speaker.')
        return
    df = df[df['speaker_id'] == speaker_id]
    for i in range(df.shape[0]):
        print(df.iloc[i]['filename'])
        y, sr = librosa.core.load(src + df.iloc[i]['filename'], sr=None, mono=False)
        y_ch = y[0, :]
        v = df.iloc[i]['diarization_labels']
        v = np.array(list(map(int, list(v))))

        frames_y = np.where(v == n_spk)[0]
        if frames_y.shape[0] == 0:
            continue
        start = frames_y[0]
        for j in range(1, frames_y.shape[0] - 1):
            end = frames_y[j]
            if frames_y[j] + 3 < frames_y[j + 1]:
                if (end + 1 - start) * time_coef / sr >= min_length:
                    start = start * time_coef
                    end = (end + 1) * time_coef
                    wav_name = df.iloc[i]['filename'].replace('.wav', '')
                    l = [wav_name, speaker_id, str(start / sr), str(end / sr), 'wav']
                    wav_name = dst + '/' + speaker_id + '/' + '.'.join(l)
                    soundfile.write(wav_name, y_ch[start:end], sr, subtype='PCM_16', format='WAV')
                start = frames_y[j + 1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str)
    parser.add_argument('-d', '--dst', type=str)
    parser.add_argument('-df', '--dataframe', type=str)
    parser.add_argument('-l', '--length', type=float)
    args = parser.parse_args()
    src = args.src
    dst = args.dst
    df_path = args.df
    min_length = args.length

    data = pd.read_csv(df_path)
    speakers = list(set(data['speaker_id'].values))
    for speaker in tqdm(speakers):
        slice_audio(data, speaker, min_length)
