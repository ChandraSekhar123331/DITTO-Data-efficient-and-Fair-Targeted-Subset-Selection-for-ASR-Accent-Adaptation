#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# In[9]:


from platform import python_version
print(python_version())

# In[ ]:




# In[5]:


!unset CUDA_VISIBLE_DEVICES
# !export CUDA_VISIBLE_DEVICES=1

# In[6]:


!nvidia-smi
!echo $CUDA_VISIBLE_DEVICES

# In[5]:


# %%capture

# !pip install git+https://github.com/huggingface/datasets.git
# !pip install git+https://github.com/huggingface/transformers.git
# !pip install jiwer
# !pip install torchaudio
# !pip install librosa

# # Monitor the training process
# # !pip install wandb

# In[3]:


%env LC_ALL=C.UTF-8
%env LANG=C.UTF-8
%env TRANSFORMERS_CACHE=content/cache
%env HF_DATASETS_CACHE=content/cache
%env CUDA_LAUNCH_BLOCKING=1
%env CUDA_DEVICE_ORDER=PCI_BUS_ID
%env CUDA_VISIBLE_DEVICES=0

# In[8]:


import os
print(os.environ['CUDA_VISIBLE_DEVICES'])
os.environ['CUDA_VISIBLE_DEVICES']='1'
print(os.environ['CUDA_VISIBLE_DEVICES'])

# In[1]:

#%%
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split

import os, sys, json, random
import librosa

#%%

# In[2]:


# data = []
# json_folder = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/MCV_accent/jsons/'
# jsons = [f.name for f in os.scandir(json_folder) if 'json' in f.name and f.name.split('.')[0] not in ['unlabelled', 'other']]

# print(jsons)
# for accent in jsons:
#     print("loading", accent)
#     json_path = json_folder + accent
#     json_file = open(json_path)
#     json_item_list = [line for line in json_file]
#     json_item_list = json_item_list[:4000]
#     json_item_list = [json.loads(line.strip()) for line in json_item_list]
#     for sample in tqdm(json_item_list):
#         try:
#             path = sample['audio_filepath']
#             name = str(path).split('/')[-1].split('.')[0]
#             label = sample['accent']
#             duration = librosa.get_duration(filename=path)
#             if duration > 30:
#                 continue
#             data.append({
#                 "name": name,
#                 "path": path,
#                 "accent": label
#             })
#         except Exception as e:
#             print(str(path), e)
#             pass

# In[ ]:




# ## inval

# In[39]:

#%%
from multiprocessing import Pool, Manager

def process_jsons(accent):
    print("loading", accent)
    json_path = json_folder + accent +'.json'
    json_file = open(json_path)
    json_item_list = [line for line in json_file]
    random.shuffle(json_item_list)
    json_item_list = json_item_list[:2500]
    json_item_list = [json.loads(line.strip()) for line in json_item_list]
    for sample in tqdm(json_item_list):
        try:
            path = sample['audio_filepath'].replace('.wav', '.mp3').replace('wav', 'clips')
            name = str(path).split('/')[-1].split('.')[0]
            label = sample['accent']
            duration = librosa.get_duration(filename=path)
            if duration > 10:
                continue
            L.append({
                "name": name,
                "path": path,
                "accent": label
            })
        except Exception as e:
            print(str(path), e)
            pass
#         break

# json_folder = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/classifier-data/val/'
#json_folder = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/classifier-data/inval/'
json_folder = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/validated/accent-trans/classifier-train/'


jsons = ['indian', 'us', 'scotland', 'philippines', 'african', 'hongkong', 'ireland', 'england']
# p = Pool(16)
# with Manager() as manager:
#     L = manager.list()
#     with p:
#         p.map(process_jsons, jsons)
#     print(L)


manager = Manager()
L = manager.list()
pool = Pool(16)
pool.map(process_jsons, jsons)
# tqdm(pool.imap(process_jsons, jsons), len(jsons))
pool.close()
pool.join()
print(len(L))


#%%
# In[40]:

#%%
data = list(L)
print(len(data))
print(data[:3])

#%%

# In[41]:

#%%
df = pd.DataFrame(data)
# df = df.sample(frac=1).reset_index(drop=True)
df.head()

# In[42]:


df['mp3path'] = df['path'].str.replace('.wav', '.mp3').replace('wav', 'clips')
df.head()

#%%

# In[43]:


## Filter broken and non-existed paths

#%%
print(f"Step 0: {len(df)}")

df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
df = df.dropna(subset=["path"])
df = df.drop("status", 1)
print(f"Step 1: {len(df)}")

# df = df.sample(frac=1)
# df = df.reset_index(drop=True)
df.head()

#%%
# In[44]:

#%%
print("Labels: ", df["accent"].unique())
print()
df.groupby("accent").count()[["path"]]

#%%
# In[ ]:


df1 = pd.concat(g.sample(100) for idx, g in df.groupby('accent'))

# In[46]:


df2 = df.drop(df1.index)

# In[47]:


print("Labels: ", df["accent"].unique())
print()
df.groupby("accent").count()[["path"]]

# In[48]:


print("Labels: ", df1["accent"].unique())
print()
df1.groupby("accent").count()[["path"]]

# In[49]:


print("Labels: ", df2["accent"].unique())
print()
df2.groupby("accent").count()[["path"]]

# In[50]:


pd.merge(df1, df2, on=['name', 'path', 'accent', 'mp3path'], how='inner')

# In[ ]:




# ## val

# In[51]:


from multiprocessing import Pool, Manager

def process_jsons(accent):
    print("loading", accent)
    json_path = json_folder + accent +'.json'
    json_file = open(json_path)
    json_item_list = [line for line in json_file]
#     random.shuffle(json_item_list)
#     json_item_list = json_item_list[:100]
    json_item_list = [json.loads(line.strip()) for line in json_item_list]
    for sample in tqdm(json_item_list):
        try:
            path = sample['audio_filepath'].replace('.wav', '.mp3').replace('wav', 'clips')
            name = str(path).split('/')[-1].split('.')[0]
            label = sample['accent']
            duration = librosa.get_duration(filename=path)
            if duration > 20:
                continue
            L_val.append({
                "name": name,
                "path": path,
                "accent": label
            })
        except Exception as e:
            print(str(path), e)
            pass
#         break

json_folder = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/classifier-data/val/'

jsons = ['indian', 'us', 'scotland', 'philippines', 'african', 'hongkong', 'ireland', 'england']
# p = Pool(16)
# with Manager() as manager:
#     L = manager.list()
#     with p:
#         p.map(process_jsons, jsons)
#     print(L)


manager = Manager()
L_val = manager.list()
pool = Pool(16)
pool.map(process_jsons, jsons)
# tqdm(pool.imap(process_jsons, jsons), len(jsons))
pool.close()
pool.join()
print(len(L_val))

# In[52]:


data_val = list(L_val)
print(len(data_val))
print(data_val[:3])

# In[53]:


df_val = pd.DataFrame(data_val)
# df = df.sample(frac=1).reset_index(drop=True)
df_val.head()

# In[54]:


df_val['mp3path'] = df_val['path'].str.replace('.wav', '.mp3').replace('wav', 'clips')
df_val.head()

# In[55]:


## Filter broken and non-existed paths

print(f"Step 0: {len(df_val)}")

df_val["status"] = df_val["path"].apply(lambda path: True if os.path.exists(path) else None)
df_val = df_val.dropna(subset=["path"])
df_val = df_val.drop("status", 1)
print(f"Step 1: {len(df_val)}")

df_val.head()

# In[56]:


print("Labels: ", df_val["accent"].unique())
print()
df_val.groupby("accent").count()[["path"]]

# In[ ]:




# In[ ]:




# In[60]:


x1 = df_val.groupby("accent").count()[["path"]]
x2 = df2.groupby("accent").count()[["path"]]
x1+x2

# In[61]:


temp = df_val.copy(deep=True)

# In[ ]:


# result1 = temp.drop(temp.groupby('accent').tail(n).index, axis=0)

# In[62]:


t1 = pd.concat(g.sample(100) for idx, g in temp.groupby('accent'))

# In[63]:


for idx, g in temp.groupby('accent'):
    print(idx)

# In[74]:


acc2co = {
    'african':2000,
    'hongkong':1200,
    'ireland':2000,
    'philippines':2000,
    'scotland':1600,
    'indian':1606,
    'england':1103,
    'us':1102
}

# In[75]:


t1 = pd.concat(g.sample(acc2co[idx]) for idx, g in temp.groupby('accent'))

# In[78]:


x3 = t1.groupby("accent").count()[["path"]] 
x3

# In[79]:


x2+x3

# In[ ]:




# In[83]:


val_inval = pd.concat([df2, t1])
val_inval.groupby("accent").count()[["path"]]

# In[ ]:




# In[ ]:




# In[9]:


import torchaudio
import librosa
import IPython.display as ipd
import numpy as np

idx = np.random.randint(0, len(df))
sample = df.iloc[idx]
path = sample["mp3path"]
label = sample["accent"]


print(f"ID Location: {idx}")
print(f"      Label: {label}")
print()

speech, sr = torchaudio.load(path)

print(sr)
# speech = speech[0].numpy().squeeze()
# speech = librosa.resample(np.asarray(speech), sr, 16_000)
# ipd.Audio(data=np.asarray(speech), autoplay=True, rate=16000)

# In[10]:


!ls

# In[66]:


!mkdir -p training_data/final

# For training purposes, we need to split data into train test sets; in this specific example, we break with a `20%` rate for the test set.

# In[18]:


#%%
save_path = "training_data/final_w0_overlap"

train_df, rem_df = train_test_split(df, train_size=0.8, random_state=101, stratify=df["accent"])
dev_df, test_df = train_test_split(rem_df, test_size=0.5, random_state=101, stratify=rem_df['accent'])

train_df = train_df.reset_index(drop=True)
dev_df = dev_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
dev_df.to_csv(f"{save_path}/dev.csv", sep="\t", encoding="utf-8", index=False)
test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)

print(train_df.shape)
print(dev_df.shape)
print(test_df.shape)

#%%


# In[84]:


save_path = "training_data/final/"

train_df, dev_df = train_test_split(val_inval, train_size=0.9, random_state=101, stratify=val_inval["accent"])

train_df = train_df.reset_index(drop=True)
dev_df = dev_df.reset_index(drop=True)
test_from_inval = df1.reset_index(drop=True)

train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
dev_df.to_csv(f"{save_path}/dev.csv", sep="\t", encoding="utf-8", index=False)
test_from_inval.to_csv(f"{save_path}/test_from_inval.csv", sep="\t", encoding="utf-8", index=False)

print(train_df.shape)
print(dev_df.shape)
print(test_from_inval.shape)

# ## Prepare Data for Training

# In[37]:


# Loading the created dataset using datasets
from datasets import load_dataset, load_metric

data_files = {
    "train": "training_data/8acc/train.csv", 
    "validation": "training_data/8acc/test.csv",
}

dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

print(train_dataset)
print(eval_dataset)

# In[2]:


# We need to specify the input and output column
input_column = "path"
output_column = "accent"

# In[3]:


# we need to distinguish the unique labels in our SER dataset
label_list = train_dataset.unique(output_column)
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)
print(f"A classification problem with {num_labels} classes: {label_list}")

# In[5]:


from transformers import AutoConfig, Wav2Vec2Processor

# In[6]:


# model_name_or_path = "facebook/wav2vec2-large-lv60" #"facebook/wav2vec2-base-960h" #"facebook/wav2vec2-large-960h" #"facebook/wav2vec2-base"
model_name_or_path = "facebook/wav2vec2-base" #"facebook/wav2vec2-base-960h" #"facebook/wav2vec2-large-960h" #"facebook/wav2vec2-base"
pooling_mode = "mean"

# In[7]:


# config
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
)
setattr(config, 'pooling_mode', pooling_mode)

# In[8]:


processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)
target_sampling_rate = processor.feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")

# # Preprocess Data

# In[ ]:


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

# In[27]:


train_dataset = train_dataset.map(
    preprocess_function,
    batch_size=100,
    batched=True,
    num_proc=16
)
eval_dataset = eval_dataset.map(
    preprocess_function,
    batch_size=100,
    batched=True,
    num_proc=16
)

# In[ ]:


idx = 0
print(f"Training input_values: {train_dataset[idx]['input_values']}")
print(f"Training attention_mask: {train_dataset[idx]['attention_mask']}")
print(f"Training labels: {train_dataset[idx]['labels']} - {train_dataset[idx]['accent']}")

# ## Model
# 
# 

# In[2]:


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

# In[3]:


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

# In[4]:


from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch

import transformers
from transformers import Wav2Vec2Processor

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

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

# In[5]:


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# Next, the evaluation metric is defined. There are many pre-defined metrics for classification/regression problems, but in this case, we would continue with just **Accuracy** for classification and **MSE** for regression. You can define other metrics on your own.

# In[15]:


is_regression = False

# In[16]:


import numpy as np
from transformers import EvalPrediction


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

# Now, we can load the pretrained XLSR-Wav2Vec2 checkpoint into our classification model with a pooling strategy.

# In[22]:


model = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
)
# this will give a warning which is totally fine

# The first component of XLSR-Wav2Vec2 consists of a stack of CNN layers that are used to extract acoustically meaningful - but contextually independent - features from the raw speech signal. This part of the model has already been sufficiently trained during pretraining and as stated in the [paper](https://arxiv.org/pdf/2006.13979.pdf) does not need to be fine-tuned anymore. 
# Thus, we can set the `requires_grad` to `False` for all parameters of the *feature extraction* part.

# In[23]:


model.freeze_feature_extractor()

# In[28]:


# for module in model._modules['wav2vec2'].encoder.layers[:5]:
#     for param in module.parameters():
#         if param.requires_grad:
#             print(param)

# In[36]:




# In[ ]:


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="data/lite_8acc",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=1.0,
    fp16=True,
    save_steps=800,
    eval_steps=800,
    logging_steps=800,
    learning_rate=1e-4,
    save_total_limit=2,
)

# In[ ]:




# For future use we can create our training script, we do it in a simple way. You can add more on you own.

# In[ ]:


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
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)
        
#         import pdb
#         pdb.set_trace()

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

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

# Now, all instances can be passed to Trainer and we are ready to start training!

# In[ ]:


trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
)

# ### Training

# Training will take between 10 and 60 minutes depending on the GPU allocated to this notebook. 
# 
# In case you want to use this google colab to fine-tune your model, you should make sure that your training doesn't stop due to inactivity. A simple hack to prevent this is to paste the following code into the console of this tab (right mouse click -> inspect -> Console tab and insert code).

# ```javascript
# function ConnectButton(){
#     console.log("Connect pushed"); 
#     document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
# }
# setInterval(ConnectButton,60000);
# ```

# In[ ]:


trainer.train()

# In[ ]:




# ## Evaluation

# In[5]:


# need to save the model first
import librosa
import torch
import torchaudio
from sklearn.metrics import classification_report
from datasets import load_dataset, load_metric

from transformers import AutoConfig, Wav2Vec2Processor

# In[6]:


test_dataset = load_dataset("csv", data_files={"test": "training_data/final/test.csv"}, delimiter="\t")["test"]
test_dataset

# In[7]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# In[8]:


model_name_or_path = "/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/classifier-data/training_data/8acc_10freeze_final/checkpoint-6400/"
config = AutoConfig.from_pretrained(model_name_or_path)
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)

# In[9]:


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

# In[10]:


import numpy as np
test_dataset = test_dataset.map(speech_file_to_array_fn)

# In[11]:


result = test_dataset.map(predict, batched=True, batch_size=8)

# In[27]:


label_names = [config.id2label[i] for i in range(config.num_labels)]
label_names

# In[28]:


y_true = [config.label2id[name] for name in result["accent"]]
y_pred = result["predicted"]

print(y_true[:5])
print(y_pred[:5])

# In[30]:


print(classification_report(y_true, y_pred, target_names=label_names))

# In[ ]:




# ## testing file generation

# In[15]:


from multiprocessing import Pool, Manager
import random
import json
from tqdm import tqdm
import librosa
import pandas as pd
import os

def process_jsons(accent):
    print("loading", accent)
    json_path = json_folder + accent +'/manifests/test.json'
    json_file = open(json_path)
    json_item_list = [line for line in json_file]
    random.shuffle(json_item_list)
    json_item_list = json_item_list[:100]
    json_item_list = [json.loads(line.strip()) for line in json_item_list]
    for sample in tqdm(json_item_list):
        try:
            path = sample['audio_filepath'].replace('.wav', '.mp3').replace('wav', 'clips')
            name = str(path).split('/')[-1].split('.')[0]
            label = sample['accent']
            duration = librosa.get_duration(filename=path)
            if duration > 20:
                continue
            L.append({
                "name": name,
                "path": path,
                "accent": label
            })
        except Exception as e:
            print(str(path), e)
            pass
#         break

json_folder = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/expts/'

jsons = ['indian', 'us', 'scotland', 'philippines', 'african', 'hongkong', 'ireland', 'england']

manager = Manager()
L = manager.list()
pool = Pool(16)
pool.map(process_jsons, jsons)
# tqdm(pool.imap(process_jsons, jsons), len(jsons))
pool.close()
pool.join()
print(len(L))

# In[16]:


data = list(L)
print(len(data))
print(data[:3])

# In[17]:


df = pd.DataFrame(data)
# df = df.sample(frac=1).reset_index(drop=True)
df.head()

# In[18]:


df['mp3path'] = df['path'].str.replace('.wav', '.mp3').replace('wav', 'clips')
df.head()

# In[21]:


## Filter broken and non-existed paths
import os 

print(f"Step 0: {len(df)}")

df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
df = df.dropna(subset=["path"])
df = df.drop("status", 1)
print(f"Step 1: {len(df)}")

# df = df.sample(frac=1)
# df = df.reset_index(drop=True)
df.head()

# In[22]:


print("Labels: ", df["accent"].unique())
print()
df.groupby("accent").count()[["path"]]

# In[23]:


save_path = "training_data/final/"

df.to_csv(f"{save_path}/test_from_val.csv", sep="\t", encoding="utf-8", index=False)

print(df.shape)

# In[33]:


import pandas as pd
df1 = pd.read_csv('training_data/final/test_from_val.csv', sep='\t', encoding='utf-8')
df2 = pd.read_csv('training_data/final/test_from_inval.csv', sep='\t', encoding='utf-8')

print(df1.shape, df2.shape)

# In[34]:


df1.head()

# In[35]:


df3 = pd.concat([df1, df2])
df3.shape

# In[36]:


save_path = "training_data/final/"

df3.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)

print(df3.shape)

# In[ ]:




# In[1]:


!gpustat

# In[ ]:




# In[ ]:




# # Prediction

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2Processor

import librosa
import IPython.display as ipd
import numpy as np
import pandas as pd

# In[ ]:


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

# In[4]:


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
        print('----------------------------')
        print(x)
        print(x.shape)
        print('----------------------------')
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

# In[6]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name_or_path = "/mnt/data/aman/mayank/MTP/mount_points/jan_19/Error-Driven-ASR-Personalization/MCV_accent/data/dristi_accent-recognition/checkpoint-6400/"
model_name_or_path = "/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/MCV_accent/data/w2v2-accent-recognition-lite/checkpoint-6400/"
config = AutoConfig.from_pretrained(model_name_or_path)
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
sampling_rate = processor.feature_extractor.sampling_rate
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)

# In[62]:


print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# In[64]:


for param in model.base.parameters():
    print('ho')

# In[65]:


from pprint import pprint
pprint(vars(model))

# In[67]:


model.__dict__

# In[80]:


for i in model._modules['wav2vec2'].encoder.layers:
    print('ho')

# In[88]:


for param in model._modules['wav2vec2'].encoder.layers[:5]:
#     print(param)
    param.requires_grad = True


# In[91]:


for module in model._modules['wav2vec2'].encoder.layers[:5]:
    for param in module.parameters():
        param.requires_grad = False

# In[ ]:




# In[ ]:




# In[ ]:




# In[8]:


model.classifier.dense

# In[ ]:




# In[13]:


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
        logits = model(input_values, attention_mask=attention_mask).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Accent": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
    return outputs


STYLES = """
<style>
div.display_data {
    margin: 0 auto;
    max-width: 500px;
}
table.xxx {
    margin: 50px !important;
    float: right !important;
    clear: both !important;
}
table.xxx td {
    min-width: 300px !important;
    text-align: center !important;
}
</style>
""".strip()

def prediction(df_row):
    path, accent = df_row["path"], df_row["accent"]
    df = pd.DataFrame([{"Accent": accent, "Sentence": "    "}])
    setup = {
        'border': 2,
        'show_dimensions': True,
        'justify': 'center',
        'classes': 'xxx',
        'escape': False,
    }
    ipd.display(ipd.HTML(STYLES + df.to_html(**setup) + "<br />"))
    speech, sr = torchaudio.load(path)
    speech = speech[0].numpy().squeeze()
    speech = librosa.resample(np.asarray(speech), sr, sampling_rate)
    ipd.display(ipd.Audio(data=np.asarray(speech), autoplay=True, rate=sampling_rate))

    outputs = predict(path, sampling_rate)
    r = pd.DataFrame(outputs)
    ipd.display(ipd.HTML(STYLES + r.to_html(**setup) + "<br />"))

# In[14]:


test = pd.read_csv("data/with_all_1k/test.csv", sep="\t")
test.head()

# In[15]:


test.iloc[3]

# In[18]:


prediction(test.iloc[3])

# In[19]:


print(test.iloc[1])
prediction(test.iloc[1])

# In[ ]:


prediction(test.iloc[2])

# In[ ]:




# In[ ]:




# In[24]:


test.iloc[1]['path']
f1 = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mozilla/cv-corpus-7.0-2021-07-21/en/clips/common_voice_en_19046599.mp3'

# In[25]:


import numpy as np
import json
from tqdm import tqdm
import pickle
import librosa
import torch
import torchaudio
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model

device = torch.device('cuda')

to = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
mo = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h").to(device)
layer_num = 1

# waveform, sample_rate = torchaudio.load(test.iloc[3]['path'])
waveform, sample_rate = librosa.load(f1, sr=48000)
# print(sample_rate)
waveform = torch.tensor(waveform)
speech_inp = to(waveform, return_tensors='pt', padding='longest').input_values.to(device)
w2v2_features = mo(speech_inp, output_hidden_states=True).hidden_states[layer_num - 1].mean(1).cpu().detach().numpy()

# In[26]:


w2v2_features.shape

# In[28]:


print(w2v2_features)

# In[ ]:




# In[94]:


!gpustat

# In[31]:


!ps -u

# In[33]:


!nvidia-smi

# In[ ]:




# In[ ]:



