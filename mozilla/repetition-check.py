import librosa, glob, sys, json, pickle, os
from tqdm import tqdm
import pandas as pd, numpy as np
from pydub import AudioSegment

wav, mp3, cnt = 0, 0, 0

def path_proc(pth):
    if './' in pth:
        return pth.replace('./', '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mozilla/')
    return pth if '~/' not in pth else pth.replace('~/', '/home/mayank/') 

def check(line_list):
    global cnt, wav, mp3
    cnt = 0
    #     if wav in 
    for json_item in tqdm(line_list):
        pth = json_item['audio_filepath']
        if os.path.exists(pth):
            cnt+=1
        if 'wav' in pth: wav += 1
        elif 'mp3' in pth: mp3 += 1
#         print(pth)
#         with open('validated.json') as mother:
#             if mother.read().find(pth)!=-1:
#                 print("{} seen already".format(mp3name))
#                 print("STOP!")
#                 sys.exit()
#     print("counted repeats: {}".format(cnt))

json_file = open('non-validated.json')
# json_file = open('validated.json')
json_item_list = [line for line in json_file]
# json_item_list = json_item_list[:5]
json_item_list = [json.loads(line.strip()) for line in json_item_list]
print('file_starting')
print('non-validated.json')
# print('validated.json')
check(json_item_list)
print(len(json_item_list))
print("wav: {}, mp3: {}".format(wav, mp3))
print("% wav exists {:.2f}".format(100*cnt/len(json_item_list)))
print('file_ending ...\n')