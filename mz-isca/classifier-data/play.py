import pdb
import torch
import inspect
from datasets import load_dataset
from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperModel
model_name_or_path = "openai/whisper-base"

model = WhisperModel.from_pretrained(model_name_or_path)

signature = inspect.signature(model.forward)

signature_columns = list(signature.parameters.keys())
print(signature_columns)

#%%
processor_Whisper = WhisperProcessor.from_pretrained(model_name_or_path)

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

#inputs_Whisper = processor_Whisper(
#        [d["array"] for d in dataset[:1]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
#)
inputs_Whisper = processor_Whisper(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
pdb.set_trace()
print(inputs_Whisper.shape)

#with torch.no_grad():
#    outputs = model(**inputs_Whisper)
#print(outputs.last_hidden_state.shape)


#from transformers import Wav2Vec2Processor
#processor_Wav2Vec2 = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
#
#inputs_Wav2Vec2 = processor_Wav2Vec2(
#    [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate)
#
#
#print(inputs_Whisper.keys())
#print(inputs_Wav2Vec2.keys())
#%%
