# Commented out IPython magic to ensure Python compatibility.
# %env LC_ALL=C.UTF-8
# %env LANG=C.UTF-8
# %env TRANSFORMERS_CACHE=/content/cache
# %env HF_DATASETS_CACHE=/content/cache
# %env CUDA_LAUNCH_BLOCKING=1

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split

import os
import sys

from datasets import load_dataset, load_metric

data_files = {
    "train": "training_data/8acc/dristi_train.csv", 
    "validation": "training_data/8acc/dristi_dev.csv",
}

dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

print(train_dataset)
print(eval_dataset)

input_column = "mp3path"
output_column = "accent"

# we need to distinguish the unique labels in our SER dataset
label_list = train_dataset.unique(output_column)
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)
print(f"A classification problem with {num_labels} classes: {label_list}")

from transformers import AutoConfig, Wav2Vec2Processor

#model_name_or_path = "facebook/wav2vec2-large-lv60" #"facebook/wav2vec2-base-960h" #"facebook/wav2vec2-large-960h" #"facebook/wav2vec2-base"
model_name_or_path = "facebook/wav2vec2-base"
pooling_mode = "mean"

# config
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
)
setattr(config, 'pooling_mode', pooling_mode)

processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)
target_sampling_rate = processor.feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")

"""# Preprocess Data"""

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

train_dataset = train_dataset.map(
    preprocess_function,
    batch_size=100,
    batched=True,
    num_proc=4
)
eval_dataset = eval_dataset.map(
    preprocess_function,
    batch_size=100,
    batched=True,
    num_proc=4
)

idx = 2
#print(f"Training input_values: {train_dataset[idx]['input_values']}")
#print(f"Training attention_mask: {train_dataset[idx]['attention_mask']}")
print(f"Training labels: {train_dataset[idx]['labels']} - {train_dataset[idx]['accent']}")


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

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

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
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

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

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
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
        logits = self.classifier(hidden_states)

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
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch

import transformers
from transformers import Wav2Vec2Processor


@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

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

model = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
)

model.freeze_feature_extractor()

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="training_data/8acc",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=10,
    fp16=True,
    save_steps=800,
    eval_steps=800,
    logging_steps=800,
    learning_rate=1e-4,
    save_total_limit=2,
)


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

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
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
    tokenizer=processor.feature_extractor,
)

### Training

trainer.train()

##############################################################################################
#
### Evaluation
#"""
#
#import librosa
#from sklearn.metrics import classification_report
#
#test_dataset = load_dataset("csv", data_files={"test": "/content/data/test.csv"}, delimiter="\t")["test"]
#test_dataset
#
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Device: {device}")
#
#model_name_or_path = "m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition"
#config = AutoConfig.from_pretrained(model_name_or_path)
#processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
#model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)
#
#def speech_file_to_array_fn(batch):
#    speech_array, sampling_rate = torchaudio.load(batch["path"])
#    speech_array = speech_array.squeeze().numpy()
#    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, processor.feature_extractor.sampling_rate)
#
#    batch["speech"] = speech_array
#    return batch
#
#
#def predict(batch):
#    features = processor(batch["speech"], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
#
#    input_values = features.input_values.to(device)
#    attention_mask = features.attention_mask.to(device)
#
#    with torch.no_grad():
#        logits = model(input_values, attention_mask=attention_mask).logits 
#
#    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
#    batch["predicted"] = pred_ids
#    return batch
#
#test_dataset = test_dataset.map(speech_file_to_array_fn)
#
#result = test_dataset.map(predict, batched=True, batch_size=8)
#
#label_names = [config.id2label[i] for i in range(config.num_labels)]
#label_names
#
#y_true = [config.label2id[name] for name in result["emotion"]]
#y_pred = result["predicted"]
#
#print(y_true[:5])
#print(y_pred[:5])
#
#print(classification_report(y_true, y_pred, target_names=label_names))
#
#"""# Prediction"""
#
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torchaudio
#from transformers import AutoConfig, Wav2Vec2Processor
#
#import librosa
#import IPython.display as ipd
#import numpy as np
#import pandas as pd
#
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model_name_or_path = "m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition"
#config = AutoConfig.from_pretrained(model_name_or_path)
#processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
#sampling_rate = processor.feature_extractor.sampling_rate
#model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)
#
#def speech_file_to_array_fn(path, sampling_rate):
#    speech_array, _sampling_rate = torchaudio.load(path)
#    resampler = torchaudio.transforms.Resample(_sampling_rate)
#    speech = resampler(speech_array).squeeze().numpy()
#    return speech
#
#
#def predict(path, sampling_rate):
#    speech = speech_file_to_array_fn(path, sampling_rate)
#    features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
#
#    input_values = features.input_values.to(device)
#    attention_mask = features.attention_mask.to(device)
#
#    with torch.no_grad():
#        logits = model(input_values, attention_mask=attention_mask).logits
#
#    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
#    outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
#    return outputs
#
#
#STYLES = """
#<style>
#div.display_data {
#    margin: 0 auto;
#    max-width: 500px;
#}
#table.xxx {
#    margin: 50px !important;
#    float: right !important;
#    clear: both !important;
#}
#table.xxx td {
#    min-width: 300px !important;
#    text-align: center !important;
#}
#</style>
#""".strip()
#
#def prediction(df_row):
#    path, emotion = df_row["path"], df_row["emotion"]
#    df = pd.DataFrame([{"Emotion": emotion, "Sentence": "    "}])
#    setup = {
#        'border': 2,
#        'show_dimensions': True,
#        'justify': 'center',
#        'classes': 'xxx',
#        'escape': False,
#    }
#    ipd.display(ipd.HTML(STYLES + df.to_html(**setup) + "<br />"))
#    speech, sr = torchaudio.load(path)
#    speech = speech[0].numpy().squeeze()
#    speech = librosa.resample(np.asarray(speech), sr, sampling_rate)
#    ipd.display(ipd.Audio(data=np.asarray(speech), autoplay=True, rate=sampling_rate))
#
#    outputs = predict(path, sampling_rate)
#    r = pd.DataFrame(outputs)
#    ipd.display(ipd.HTML(STYLES + r.to_html(**setup) + "<br />"))
#
#test = pd.read_csv("/content/data/test.csv", sep="\t")
#test.head()
#
#prediction(test.iloc[0])
#
#prediction(test.iloc[1])
#
#prediction(test.iloc[2])
