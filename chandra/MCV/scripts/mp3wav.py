import librosa, glob, sys, json, pickle, os
from tqdm import tqdm
import pandas as pd, numpy as np
from pydub import AudioSegment

# json_folder = 'validated/accent-trans/'
base_dir = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mozilla/cv-corpus-7.0-2021-07-21/en/'
# jsons = [f.name for f in os.scandir(json_folder) if 'json' in f.name]


def _write(path, json_list):
    with open(path, 'w') as file:
        for json_entry in json_list:
            json.dump(json_entry, file)
            file.write("\n")


# def convert(line_list, json_path):
#     global cnt
#     for item in tqdm(line_list):
#         filepath = item['audio_filepath'].replace('.wav', '.mp3').replace('/wav/', '/clips/')
#         try:
#             sound = AudioSegment.from_file(filepath, read_ahead_limit=0.01)
#         except:
#             print("Custom ***TSS*** Error: Unable to decode file. Exiting now. Consider increasing the read_ahead_limit or ignore this mp3 file altogether")
#             sys.exit()
#         sound.export(item['audio_filepath'], format="wav")
#         except:
#             print(item['audio_filepath'])
#             sys.exit()
#         if os.path.isfile(item['audio_filepath']):
#             pass
#         else:
#             mp3path = item['audio_filepath'].replace('.wav', '.mp3').replace('/wav/', '/clips/')
#             sound = AudioSegment.from_mp3(mp3path)
#             sound.export(item['audio_filepath'], format="wav")

cnt = 0
def convert(line_list, json_path):
    global cnt
    new_json_lst = []
    with open(json_path, 'w') as new_json_file:
        for json_item in tqdm(line_list):
            mp3name = json_item['audio_filepath'].split('/en/wav/')[1].replace('wav', 'mp3')
            path_to_mp3 = base_dir+'clips/'+mp3name
            duration = librosa.get_duration(filename=path_to_mp3)
            if duration>20:
                cnt += 1
                print(json_item, "has length > 20sec")
                continue
            if not(os.path.isfile(json_item['audio_filepath'])):
                sound = AudioSegment.from_file(path_to_mp3, read_ahead_limit=0.01)
                sound.export(json_item['audio_filepath'], format="wav")
            new_json_item = json_item
            new_json_item['duration'] = duration
            new_json_lst.append(new_json_item)
            # json.dump(json_item, new_json_file)
#             os.unlink(wav_file)
            # new_json_file.write('\n')

    return new_json_lst

# def filter_durations(line_list, json_path):
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


accents = ['african', 'indian', 'hongkong', 'philippines', 'england', 'scotland', 'ireland', 'australia', 'canada', 'us']
low_resource = ['bermuda', 'southatlandtic', 'wales', 'malaysia']

json_base_dir = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/chandra/MCV/data/'
print(accents+low_resource)

for accent in accents+low_resource:

    json_folder = json_base_dir+accent + "/"
    print('_'*20)
    print(accent)
    print("\n")
    
    
    file = 'all.json'
    file_path = json_folder+file
    json_file = open(file_path)
    json_item_list = [line for line in json_file]
    json_item_list = [json.loads(line.strip()) for line in json_item_list]
        
    print('{} starting'.format(accent))
    new_json_lst = convert(json_item_list, file_path)

    _write(file_path, new_json_lst)
    print(len(json_item_list))
    print('file_ending ...\n')
    print('_'*20)

print("total {} files of >20sec".format(cnt))
