#%%
import re
import numpy as np
import random
import librosa
from datasets import load_dataset
import torch
import torchaudio
from jiwer import wer
import argparse

#%%

#%%
parser = argparse.ArgumentParser(description ='input')
parser.add_argument('--accent', type = str,  help ='query set')
parser.add_argument('--home_dir', type = str,  help ='home dir')

args = parser.parse_args()
accent = args.accent
home_dir = args.home_dir

#%%

#%%
data_files = {
    "test": f"{home_dir}/{accent}/manifests/test.csv",
}
dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )

#%%

#%%
print(dataset)

#%%

#%%
import pandas as pd

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    print(df)

#show_random_elements(dataset['test'].remove_columns(['name', 'path', 'mp3path']))

#%%

#%%
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\)]'

def convert_unicodes(batch):
    batch['text'] = batch['text'].encode('ascii', 'ignore')
    return batch

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower() + " "
    return batch

def upper_case(batch):
    batch["text"] = batch['text'].upper() 
    return batch

dataset = dataset.map(convert_unicodes)
dataset = dataset.map(remove_special_characters)
dataset = dataset.map(upper_case)

#show_random_elements(dataset['test'].remove_columns(['name', 'path', 'mp3path']))

#%%

#%%
def add_audio_array(batch):
    waveform, sample_rate = librosa.load(batch["path"], sr=16000)
    batch["audio"] = {
            'array': waveform,
            'sampling_rate': sample_rate
            }
    return batch

dataset = dataset.map(add_audio_array)

#%%


#%%
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-100h").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-100h")
processorLM = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")

#%%

#%%
import pdb
def map_to_pred(batch):
    audio = batch['audio']

    input_values = processor(audio["array"], return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values.to("cuda")).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        transcriptionLM = processorLM.batch_decode(logits.cpu().numpy()).text[0]
        #pdb.set_trace()
        batch["transcription"] = transcription
        batch["transcriptionLM"] = transcriptionLM
        return batch

#result = mcv['test'].map(map_to_pred, batched=True, batch_size=1, remove_columns=["audio"])
result = dataset['test'].map(map_to_pred, remove_columns=["audio"])

print("WER:", wer(result["text"], result["transcription"]))
print("WER with LM:", wer(result["text"], result["transcriptionLM"]))

#%%




