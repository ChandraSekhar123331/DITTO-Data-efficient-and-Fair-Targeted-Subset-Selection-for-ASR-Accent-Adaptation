import librosa, glob, sys, json, pickle, os
from tqdm import tqdm
import pandas as pd, numpy as np
from pydub import AudioSegment
from glob import glob

json_folder = './data/'
base_dir = '../../mozilla/cv-corpus-7.0-2021-07-21/en/'
jsons = [_ for _ in glob(f"{json_folder}/*/all.json")]
# jsons = [f"{json_folder}/indian/all.json", f"{json_folder}/african/all.json"] 
print(jsons)
# jsons = ['us.json']

def path_proc(pth):
    if './' in pth:
        return pth.replace('./', '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mozilla/')
    return pth if '~/' not in pth else pth.replace('~/', '/home/mayank/') 


def convert(line_list, json_path):
        for json_item in tqdm(line_list):
            mp3name = json_item['audio_filepath'].split('/en/wav/')[1].replace('wav', 'mp3')
            wav_file = base_dir + 'wav/' + mp3name.replace('.mp3', '.wav')
            # print(json_item)
            try:
                os.remove(wav_file)
            except:
                continue
        

for json_name in tqdm(jsons):
    json_path = json_name
    print('_'*20)
    print(json_name)

    json_file = open(json_path)
    item_list = [line for line in json_file]
#     json_item_list = json_item_list[:100]
    # json_item_list = [json.loads(line.strip()) for line in item_list]
    json_item_list = []
    for line in item_list:
        try:
            json_item_list.append(json.loads(line.strip()))
        except:
            print(line)
    print('file_starting')
    print(json_path)
    convert(json_item_list, json_path)
    print(len(json_item_list))
    print('file_ending ...\n')