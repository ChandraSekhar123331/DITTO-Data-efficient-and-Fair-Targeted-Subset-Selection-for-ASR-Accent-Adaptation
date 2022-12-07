from datasets import load_dataset

dataset = load_dataset("europarl_bilingual", lang1="en", lang2="sv", split="train")

chars_to_ignore_regex = '[,?.!\-\;\:"“%‘”?—’…–]' 

import re

def extract_text(batch):
    text = batch["translation"][target_lang]
    batch["text"] = re.sub(chars_to_ignore_regex, "", text.lower())
    return batch

dataset = dataset.map(extract_text, remove_columns=dataset.column_names)

with open("text.txt", "w") as file:
      file.write(" ".join(dataset["text"]))


with open("5gram.arpa", "r") as read_file, open("5gram_correct.arpa", "w") as write_file:
    has_added_eos = False
    for line in read_file:
        if not has_added_eos and "ngram 1=" in line:
            count=line.strip().split("=")[-1]
            write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
        elif not has_added_eos and "<s>" in line:
            write_file.write(line)
            write_file.write(line.replace("<s>", "</s>"))
            has_added_eos = True
        else:
            write_file.write(line)

                                                                        
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("hf-test/xls-r-300m-sv")

vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}


from pyctcdecode import build_ctcdecoder

decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()),
        kenlm_model_path="5gram_correct.arpa",
)

from transformers import Wav2Vec2ProcessorWithLM

processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)






      



