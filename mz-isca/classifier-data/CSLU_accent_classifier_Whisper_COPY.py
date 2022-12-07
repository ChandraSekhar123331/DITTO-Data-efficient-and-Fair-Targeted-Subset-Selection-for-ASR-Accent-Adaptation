#%%
import numpy as np
import pandas as pd
import pdb

from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split

import os
import sys

from datasets import load_dataset, load_metric

data_files = {
    "train": "CSLU_training_data/init/train.csv", 
    "validation": "CSLU_training_data/init/dev.csv",
}

#%%

#%%
dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

print(train_dataset)
print(eval_dataset)

#%%

#%%
input_column = "path"
output_column = "accent"

# we need to distinguish the unique labels in our SER dataset
label_list = train_dataset.unique(output_column)
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)
print(f"A classification problem with {num_labels} classes: {label_list}")

#%%

#%%
from transformers import AutoConfig
from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperModel

model_name_or_path = "openai/whisper-base"
pooling_mode = "mean"

# config
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="whisper_clf",
)
setattr(config, 'pooling_mode', pooling_mode)

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
target_sampling_rate = processor.feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")

#%%


"""# Preprocess Data"""

#%%
def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]

    result = processor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)

    return result

#%%

#%%
column_names = train_dataset.column_names
train_dataset = train_dataset.map(
    preprocess_function,
    batch_size=100,
    batched=True,
    num_proc=4,
    remove_columns=column_names
)
eval_dataset = eval_dataset.map(
    preprocess_function,
    batch_size=100,
    batched=True,
    num_proc=4,
    remove_columns=column_names
)

print(train_dataset)

#idx = 2
#print(f"Training input_values: {train_dataset[idx]['input_values']}")
#print(f"Training attention_mask: {train_dataset[idx]['attention_mask']}")
#print(f"Training labels: {train_dataset[idx]['labels']} - {train_dataset[idx]['accent']}")

#%%

#%%
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput

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
import pdb
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.whisper.modeling_whisper import (
    WhisperPreTrainedModel,
)

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, 300)
        self.dense2 = nn.Linear(300, 100)
        self.dropout = nn.Dropout(.3)
        self.out_proj = nn.Linear(100, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense1(x)
        x1 = torch.tanh(x)
#         x2 = self.dropout(x1)
        x2 = self.dense2(x1)
        x2 = torch.tanh(x2)
#         x3 = self.dropout(x2)        
        x3 = self.out_proj(x2)
        return x1, x2, x3

class WhisperForSpeechClassification(WhisperPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.whisper = WhisperModel(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        pass

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)  # b x t x 768 --> b x 768
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_features,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=None,
            labels=None,
    ):
        with torch.no_grad():
            outputs = self.whisper(
                input_features,
                output_hidden_states=output_hidden_states,
                return_dict=True
            )

        pdb.set_trace()

        hidden_states = outputs.last_hidden_state
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        h1, h2, logits = self.classifier(hidden_states)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

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

#%%
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch

import transformers

@dataclass
class DataCollatorCTCWithPadding:

    feature_extractor: WhisperFeatureExtractor
    processor: WhisperProcessor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch

data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor, processor=processor, padding=True)
#%%

#%%
is_regression = False

import numpy as np
from transformers import EvalPrediction


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

model = WhisperForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
)

model.freeze_feature_extractor()

#%%

#%%
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="CSLU_checkpoints/whisper",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    eval_accumulation_steps=4,
    evaluation_strategy="steps",
    num_train_epochs=200,
    fp16=True,
    save_steps=500,
    eval_steps=100,
    logging_steps=800,
    learning_rate=1e-4,
    save_total_limit=10,
)

#%%

#%%
from typing import Any, Dict, Union
import torch
from packaging import version
from torch import nn
from transformers import (
    Trainer,
    is_apex_available,
)

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()

        inputs = self._prepare_inputs(inputs)

###############################################
        #import pdb
        #pdb.set_trace()
###############################################

#        if self.use_amp:
#            with autocast():
#                loss = self.compute_loss(model, inputs)
#        else:
#            loss = self.compute_loss(model, inputs)

        loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

###############################################
        loss = loss.mean()
###############################################

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

"""Now, all instances can be passed to Trainer and we are ready to start training!"""

trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=feature_extractor,
)

#%%

### Training

#%%
trainer.train()

#%%
