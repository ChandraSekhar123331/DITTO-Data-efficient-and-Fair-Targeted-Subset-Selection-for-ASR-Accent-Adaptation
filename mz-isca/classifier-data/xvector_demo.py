#%%
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForXVector
from datasets import load_dataset
import torch

#%%

#%%
dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv")

#%%

# audio file is decoded on the fly
#%%
inputs = feature_extractor(
    [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
)
with torch.no_grad():
    embeddings = model(**inputs).embeddings

print(embeddings.shape)

embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

#%%

# the resulting embeddings can be used for cosine similarity-based retrieval
#%%
cosine_sim = torch.nn.CosineSimilarity(dim=-1)
similarity = cosine_sim(embeddings[0], embeddings[1])
threshold = 0.7  # the optimal threshold is dataset-dependent
if similarity < threshold:
    print("Speakers are not the same!")
round(similarity.item(), 2)

#%%

#%%
print(inputs.keys(), embeddings[0].shape)
print(inputs)

#%%


#%%
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')

results = processor(
    [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate)

#%%

#%%
print(results.keys())
print(results)

#%%


#%%
from transformers import AutoConfig, Wav2Vec2Model

model_name_or_path = 'anton-l/wav2vec2-base-superb-sv'

config = AutoConfig.from_pretrained(
            model_name_or_path,   
            finetuning_task="wav2vec2_clf",
        )

model = Wav2Vec2ForXVector.from_pretrained(model_name_or_path)
model2 = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')

print(config)

#%%

#%%
with open('xvector_model.info', 'w') as f:
    print(model.__dict__, file=f)

#%%

#%%
print(type(model.__dict__['_modules']))
print(model.__dict__['_modules'].keys())
#for module in model.feature_extractor:
#    print(module)

#%%

#%%
model2.__dict__['_modules']['feature_extractor']

#%%

#%%
model.feature_extractor.__freeze_parameters()

#%%
