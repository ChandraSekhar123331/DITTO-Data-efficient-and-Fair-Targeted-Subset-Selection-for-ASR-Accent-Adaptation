import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2Processor
import librosa
import IPython.display as ipd
import numpy as np
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

@dataclass
class SpeechClassifierOutput(ModelOutput):
        loss: Optional[torch.FloatTensor] = None
        logits: torch.FloatTensor = None
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        attentions: Optional[Tuple[torch.FloatTensor]] = None
        h1: Optional[Tuple[torch.FloatTensor]] = None
        h2: Optional[Tuple[torch.FloatTensor]] = None

from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2PreTrainedModel,
            Wav2Vec2Model
)

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = "/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/classifier-data/training_data/8acc_10freeze_final/checkpoint-6400/"
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
    #     attention_mask = features.attention_mask.to(device)
    #### just trying ####
    attention_mask = None
    #### just trying ####
    with torch.no_grad():
            op = model(input_values, attention_mask=attention_mask)
            h1, h2, logits = op.h1, op.h2, op.logits
            scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            outputs = [{"Accent": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
            return outputs, h1, h2, logits

def prediction(df_row):
    if 'path' in df_row: path = df_row['path']
    else: path = df_row['audio_filepath']
    speech, sr = torchaudio.load(path)
    speech = speech[0].numpy().squeeze()
    speech = librosa.resample(np.asarray(speech), sr, sampling_rate)
    outputs, h1, h2, logits = predict(path, sampling_rate)
#    print('{:20s} {}'.format('Accent', 'Score'))
#    for op in outputs:
#        print(f"{op['Accent']:20s} {op['Score']}")
    return outputs
    

test = pd.read_csv("~/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/classifier-data/training_data/final/test.csv", sep="\t")
test.head()

test.iloc[3]
prediction(test.iloc[3])

list(config.id2label.values())

import json
import pandas as pd
accents = list(config.id2label.values())
df = pd.DataFrame(columns=accents+['audio_filepath', 'text', 'duration', 'dir'])
dir_path = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/expts'
iloc = 0
for accent in tqdm(accents):
    manifests_path = dir_path + f'/{accent}/manifests/' 
    selection_file_dir = manifests_path + 'selection.json'
    selection_file = open(selection_file_dir)
    selection_list = [json.loads(line.strip()) for line in selection_file]
    for sel_file in tqdm(selection_list):
        output = prediction(sel_file)
        row = [float(op['Score'].replace('%', '')) for op in output] + [sel_file['audio_filepath'], sel_file['text'], sel_file['duration'], accent]
        df.loc[iloc] = row
        iloc += 1

df.head()

len(df)

df.to_csv("~/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/expts/classifier_pred_on_selection.csv", index=False)
