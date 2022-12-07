import torch, os, sys, json, librosa, pickle
from IPython.core.display import display, HTML
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from itertools import islice

import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2Processor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import librosa
import IPython.display as ipd
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    hidden_rep: Optional[Tuple[torch.FloatTensor]] = None


from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x0 = self.dropout(x)
#         print('----------------------------')
#         print(x0[:,-10:])
#         print(x0.shape)
#         print('----------------------------')
        x1 = self.out_proj(x0)
        return x0, x1

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception("The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        hidden_rep, logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (hidden_rep + logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            hidden_rep=hidden_rep
        )

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# model_name_or_path = "/mnt/data/aman/mayank/MTP/mount_points/jan_19/Error-Driven-ASR-Personalization/MCV_accent/data/dristi_accent-recognition/checkpoint-6400/"
model_name_or_path = "/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/MCV_accent/data/dristi_accent-recognition/checkpoint-6400/"
config = AutoConfig.from_pretrained(model_name_or_path)
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
sampling_rate = processor.feature_extractor.sampling_rate
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)

def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)
    attention_mask = None

    with torch.no_grad():
        op = model(input_values, attention_mask=attention_mask)
        logits = op.logits
        hidden_rep = op.hidden_rep
        
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Accent": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
    return outputs, hidden_rep

def prediction(df_row):
    if 'path' in df_row: path = df_row["path"]
    else: path = df_row["audio_filepath"]
#     print(path)
    speech, sr = torchaudio.load(path)
    speech = speech[0].numpy().squeeze()
    speech = librosa.resample(np.asarray(speech), sr, sampling_rate)
    outputs, hidden_rep = predict(path, sampling_rate)
#     print(hidden_rep[:,-10:])
    return hidden_rep

def extract_features(file_list, file_dir):
    with open(file_dir.replace('.json', '_w2v2.file'), 'wb') as f:
        for file in tqdm(file_list):
            w2v2_features = prediction(file).cpu().detach().numpy()
            pickle.dump(w2v2_features, f)

accent_map = {"ABA":"arabic","SKA":"arabic","YBAA":"arabic","ZHAA":"arabic",
              "BWC":"chinese","LXC":"chinese","NCC":"chinese","TXHC":"chinese",
              "ASI":"hindi","RRBI":"hindi","SVBI":"hindi","TNI":"hindi",
              "HJK":"korean","HKK":"korean","YDCK":"korean","YKWK":"korean",
              "EBVS":"spanish","ERMS":"spanish","MBMPS":"spanish","NJS":"spanish",
              "HQTV":"vietnamese","PNV":"vietnamese","THV":"vietnamese","TLV":"vietnamese"
              }
speakers = [speaker for speaker, accent in accent_map.items()]
base_dir = './'

for _dir in tqdm(speakers[1:]):
    manifests_path = base_dir + _dir + '/manifests/'
    print('_'*20)
    print(_dir)

    seed_file_dir = manifests_path + 'seed.json'
    seed_file = open(seed_file_dir)
    seed_list = [json.loads(line.strip()) for line in seed_file]

    selection_file_dir = manifests_path + 'selection.json'
    selection_file = open(selection_file_dir)
    selection_list = [json.loads(line.strip()) for line in selection_file]

    test_file_dir = manifests_path + 'test.json'
    test_file = open(test_file_dir)
    test_list = [json.loads(line.strip()) for line in test_file]

    print('seed_file_starting')
    print(seed_file_dir)
    extract_features(seed_list, seed_file_dir)
    print(len(seed_list))
    print('seed_file_ending ...\n')
    
    print('selection_file_starting')
    extract_features(selection_list, selection_file_dir)
    print(len(selection_list))
    print('selection_file_ending ...\n\n')
    
    print('test_file_starting')
    extract_features(test_list, test_file_dir)
    print(len(test_list))
    print('test_file_ending ...\n\n')