
#import os
#print(os.environ['CUDA_VISIBLE_DEVICES'])
#os.environ['CUDA_VISIBLE_DEVICES']='1'
#print(os.environ['CUDA_VISIBLE_DEVICES'])
#

#%%
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split

import os, sys, json, random
import librosa

from transformers import AutoConfig, Wav2Vec2Processor

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

import librosa
import torch
import torchaudio
from sklearn.metrics import classification_report
from datasets import load_dataset, load_metric

from transformers import AutoConfig, Wav2Vec2Processor

#%%

#%%
@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None   
    h1: Optional[Tuple[torch.FloatTensor]] = None
    h2: Optional[Tuple[torch.FloatTensor]] = None

#%%

#%%
class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, 300)
        self.dense2 = nn.Linear(300, 100)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(100, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense1(x)
        x1 = torch.tanh(x)
        x2 = self.dropout(x1)
        x2 = self.dense2(x2)
        x2 = torch.tanh(x2)
        x3 = self.dropout(x2)        
        x3 = self.out_proj(x3)
        return x1, x2, x3


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
        for module in self.wav2vec2.encoder.layers[:10]:
            for param in module.parameters():
                param.requires_grad = False

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
        h1, h2, logits = self.classifier(hidden_states)

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
            output = (h1 + h2 + logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            h1=h1,
            h2=h2
        )

#%%

# ## Evaluation

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

#%%

#%%
model_name_or_path = "/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/classifier-data/training_data/inspect_old_numbers/checkpoint-6400/"
config = AutoConfig.from_pretrained(model_name_or_path)
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)

#%%

#%%
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["mp3path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, processor.feature_extractor.sampling_rate)

    batch["speech"] = speech_array
    return batch

def predict(batch):
    features = processor(batch["speech"], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = None

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits 

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch

#%%

#%%
test_dataset = load_dataset("csv", data_files={"test": "/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/classifier-data/training_data/final/test.csv"}, delimiter="\t")["test"]
print(test_dataset)
test_dataset = test_dataset.map(speech_file_to_array_fn)

#%%

#%%
result = test_dataset.map(predict, batched=True, batch_size=8)

label_names = [config.id2label[i] for i in range(config.num_labels)]
print(label_names)

y_true = [config.label2id[name] for name in result["accent"]]
y_pred = result["predicted"]

print(y_true[:5])
print(y_pred[:5])

#%%

#%%
print(classification_report(y_true, y_pred, target_names=label_names))

#%%
