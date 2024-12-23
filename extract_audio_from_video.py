import torchvision
import torchaudio
from  torchaudio.transforms import Resample as Ta
import torch
import pdb

def extract(path, st, et):
    video = torchvision.io.read_video(path, pts_unit='sec', start_pts=st, end_pts=et)
    silent_video = video[0]
    audio = video[1].mean(axis=0)

    return silent_video, audio, video[2]['audio_fps']

def main(path):
    for i in range(0, 600, 3):
        _, audio, sr = extract(path, int(i), int(i+3))
        name = str(int(i/3)+1)
        resampler = Ta(sr, 16000, dtype=audio.dtype)
        torchaudio.save(name+"_from_vis.wav", resampler(audio).unsqueeze(0), 16000)

if __name__=="__main__":
    path = #PATH TO VIDEO
    main(path)