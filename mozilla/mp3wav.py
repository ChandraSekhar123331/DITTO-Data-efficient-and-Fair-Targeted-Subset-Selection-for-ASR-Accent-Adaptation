import librosa, glob, sys, json, pickle, os
from tqdm import tqdm
import pandas as pd, numpy as np
from pydub import AudioSegment
import argparse

# json_folder = './accent/'
# base_dir = './cv-corpus-7.0-2021-07-21/en/'
# jsons = [f.name for f in os.scandir(json_folder) if 'json' in f.name]
# jsons = ['us.json']

# def path_proc(pth):
#     if './' in pth:
#         return pth.replace('./', '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mozilla/')
#     return pth if '~/' not in pth else pth.replace('~/', '/home/mayank/') 

# def processed_name(json_name):
#     json_name = json_name.split('/')[-1]
#     return './processed_jsons/'+json_name
cnt = 0   
def convert(line_list, json_path):
    global cnt
    for item in tqdm(line_list):
        filepath = item['audio_filepath'].replace('.wav', '.mp3').replace('/wav/', '/clips/')
        try:
            sound = AudioSegment.from_file(filepath, read_ahead_limit=0.01)
        except:
            print(item['audio_filepath'])
            print("Custom ***TSS*** Error: Unable to decode file. Exiting now. Consider increasing the read_ahead_limit or ignore this mp3 file altogether")
            sys.exit()
        sound.export(item['audio_filepath'], format="wav")
#         if os.path.isfile(item['audio_filepath']):
#             pass
#         else:
#             mp3path = item['audio_filepath'].replace('.wav', '.mp3').replace('/wav/', '/clips/')
#             sound = AudioSegment.from_mp3(mp3path)
#             sound.export(item['audio_filepath'], format="wav")
#     with open(json_path, 'w') as new_json_file:
#         for json_item in tqdm(line_list):
#             mp3name = json_item['audio_filepath'].split('/en/wav/')[1].replace('wav', 'mp3')
#             path_to_mp3 = base_dir+'clips/'+mp3name
#             json_item['audio_filepath'] = path_proc(json_item['audio_filepath'])
# #             wav_file = base_dir + 'wav/' + mp3name.replace('.mp3', '.wav')
# #             sound = AudioSegment.from_mp3(path_to_mp3)
# #             sound.export(wav_file, format="wav")
# #             duration = librosa.get_duration(filename=wav_file)
#             duration = librosa.get_duration(filename=path_to_mp3)
#             json_item['duration'] = duration
#             json.dump(json_item, new_json_file)
# #             os.unlink(wav_file)
#             new_json_file.write('\n')

# def convert(line_list, json_path):
#     global cnt
#     with open(json_path, 'w') as new_json_file:
#         for json_item in tqdm(line_list):
#             duration = json_item['duration']
#             if duration<20: 
#                 json.dump(json_item, new_json_file)
#                 new_json_file.write('\n')
#             else: 
#                 cnt+=1
#                 print(json_item)


parser = argparse.ArgumentParser()
parser.add_argument("--accent", type=str, required=True)

args = parser.parse_args()
accents = [args.accent]
# accents = ['canada', 'england', 'australia', 'us', 'philippines', 'indian', 'african']

# base_dir = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-expts/'
base_dir = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/expts/'
print(accents)


for accent in accents:
    
#     json_folder = base_dir+accent+'/manifests/'
    json_folder = base_dir+accent + "/"
    print('_'*20)
    print(accent)
    print("\n")
    
    jsons = [f.name for f in os.scandir(json_folder) if '.json' in f.name]
    
    for file in jsons:
        
        file_path = json_folder+file
        json_file = open(file_path)
        json_item_list = [line for line in json_file]
        json_item_list = [json.loads(line.strip()) for line in json_item_list]
        
        print('{} starting'.format(file))
        convert(json_item_list, file_path)
        print(len(json_item_list))
        print('file_ending ...\n')
#         break
#     break

print("total {} files of >20sec".format(cnt))