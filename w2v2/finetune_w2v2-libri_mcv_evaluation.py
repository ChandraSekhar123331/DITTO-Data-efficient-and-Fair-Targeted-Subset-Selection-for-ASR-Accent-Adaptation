#%%
import argparse
parser = argparse.ArgumentParser(description ='input')
parser.add_argument('--accent', type = str,  help ='query set')
parser.add_argument('--fxn', type = str,  help ='function')
parser.add_argument('--cp', type = int,  help ='checkpoint')

args = parser.parse_args()
accent = args.accent
fxn = args.fxn
cp = args.cp

#%%

#%%
from datasets import load_dataset, load_metric

data_files = {
    "test": f"../mz-isca/expts/{accent}/manifests/test.csv",
}
mcv = load_dataset("csv", data_files=data_files, delimiter="\t", )

#%%

#%%
print(mcv)  

#%%

#%%
from datasets import ClassLabel
import random
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

show_random_elements(mcv["test"].remove_columns(["name", "path", "mp3path"]), num_examples=10)
#%%

#%%
import re
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def convert_unicodes(batch):
    batch['text'] = batch['text'].encode('ascii', 'ignore')
    return batch

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower() + " "
    return batch

def upper_case(batch):
    batch["text"] = batch['text'].upper() 
    return batch

mcv = mcv.map(convert_unicodes)
mcv = mcv.map(remove_special_characters)
mcv = mcv.map(upper_case)

show_random_elements(mcv["test"].remove_columns(["name", "path", "mp3path"]))
#%%


### Load Wav2Vec2 Feature Extractor

#%%
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-100h")

#%%

### Preprocess Data

#%%
import numpy as np
import random
import librosa
import torch
import torchaudio

def add_audio_array(batch):
    waveform, sample_rate = librosa.load(batch["mp3path"], sr=16000)
    batch["audio"] = {
            'array': waveform,
            'sampling_rate': sample_rate
            }
    return batch

mcv = mcv.map(add_audio_array)

#%%

#%%
def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

mcv = mcv.map(prepare_dataset, remove_columns=mcv.column_names["test"], num_proc=4)

#%%

### Evaluate

#%%
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM

model = Wav2Vec2ForCTC.from_pretrained(f"mcv/{accent}/libri-100h/{fxn}/checkpoint-{cp}").cuda()
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-100h")
processorLM = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")

#%%

#%%
import pdb

wer_metric = load_metric("wer")

def map_to_result(batch):
    #pdb.set_trace()

    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["pred_str_LM"] = processorLM.batch_decode(logits.cpu().numpy()).text[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)

    return batch

results = mcv["test"].map(map_to_result, remove_columns=mcv["test"].column_names)

print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))
print("Test WER w LM: {:.3f}".format(wer_metric.compute(predictions=results["pred_str_LM"], references=results["text"])))

#%%

