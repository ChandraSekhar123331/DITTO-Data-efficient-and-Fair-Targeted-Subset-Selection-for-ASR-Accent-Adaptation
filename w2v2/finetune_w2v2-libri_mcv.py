#%%
from datasets import load_dataset, load_metric

accent = 'scotland'
fxn = 'LogDMI'
data_files = {
    "train": f"../mz-isca/expts/{accent}/manifests/TSS_output/all/budget_200/target_20/{fxn}/eta_1.0/euclidean/wv10_100/run_1/train/train.csv", 
    #"train": f"../mz-isca/expts/{accent}/manifests/TSS_output/all/budget_200/classifier/run_1/train/train.csv", 
    #"train": f"../mz-isca/expts/{accent}/manifests/TSS_output/all/budget_200/target_20/random/run_1/train/train.csv", 
    "dev": f"../mz-isca/expts/{accent}/manifests/dev.csv",
    "test": f"../mz-isca/expts/{accent}/manifests/test.csv",
}
mcv = load_dataset("csv", data_files=data_files, delimiter="\t", )

#%%

#%%
mcv

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

show_random_elements(mcv["train"].remove_columns(["name", "path", "mp3path"]), num_examples=10)
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

show_random_elements(mcv["train"].remove_columns(["name", "path", "mp3path"]))
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

print(mcv["train"][0])

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
rand_int = random.randint(0, len(mcv["train"]))

print("Target text:", mcv["train"][rand_int]["text"])
print("Input array shape:", np.asarray(mcv["train"][rand_int]["audio"]["array"]).shape)
print("Sampling rate:", mcv["train"][rand_int]["audio"]["sampling_rate"])

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

mcv = mcv.map(prepare_dataset, remove_columns=mcv.column_names["train"], num_proc=4)

#%%

#%%
max_input_length_in_sec = 4.0
mcv["train"] = mcv["train"].filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])

#%%

## Training & Evaluation

#%%
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

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
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

#%%

#%%
import pdb

wer_metric = load_metric("wer")

def compute_metrics(pred):
#    pdb.set_trace()

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)

    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

#%%


#%%
from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-100h",
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
)

model.freeze_feature_encoder()

#%%

#%%
from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir=f'mcv/{accent}/libri-100h',
  group_by_length=True,
  per_device_train_batch_size=8,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  gradient_checkpointing=True,
  save_steps=10,
  eval_steps=10,
  logging_steps=10,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=2,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=mcv["train"],
    eval_dataset=mcv["dev"],
    tokenizer=processor.feature_extractor,
)

trainer.train()

#%%

"""
${}^1$ To allow models to become independent of the speaker rate, in CTC, consecutive tokens that are identical are simply grouped as a single token. However, the encoded labels should not be grouped when decoding since they don't correspond to the predicted tokens of the model, which is why the `group_tokens=False` parameter has to be passed. If we wouldn't pass this parameter a word like `"hello"` would incorrectly be encoded, and decoded as `"helo"`.

${}^2$ The blank token allows the model to predict a word, such as `"hello"` by forcing it to insert the blank token between the two l's. A CTC-conform prediction of `"hello"` of our model would be `[PAD] [PAD] "h" "e" "e" "l" "l" [PAD] "l" "o" "o" [PAD]`.

"""


### Evaluate

#%%
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM

model = Wav2Vec2ForCTC.from_pretrained(f"mcv/{accent}/libri-100h/checkpoint-90").cuda()
processorLM = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")

#%%

#%%
import pdb

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

#%%
show_random_elements(results)

#%%

#%%
results[1]

#%%

#%%
model.to("cuda")

with torch.no_grad():
  logits = model(torch.tensor(mcv["test"][:1]["input_values"], device="cuda")).logits

pred_ids = torch.argmax(logits, dim=-1)

# convert ids to tokens
print(" ".join(processor.tokenizer.convert_ids_to_tokens(pred_ids[0].tolist())))

#%%


