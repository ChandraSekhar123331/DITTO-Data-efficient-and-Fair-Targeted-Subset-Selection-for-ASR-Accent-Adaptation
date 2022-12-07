import librosa, glob, sys, json, pickle, os
from tqdm import tqdm
import pandas as pd, numpy as np
from pydub import AudioSegment

base_dir = './cv-corpus-7.0-2021-07-21/en/'
accent_files, filenames_in_accent = {}, {}


total_processed, already_seen_files, no_upvotes, no_accent, no_transcript, no_file, other, valid = [0]*8

def process_mcv(row):
    global total_processed, already_seen_files, no_upvotes, no_accent, no_transcript, no_file, other, valid
    try:
        mp3name, transcript, up, down = row['path'], row['sentence'], int(row['up_votes']), int(row['down_votes'])
        gender, accent, age = row['gender'], row['accent'], row['age']
        client_id = row['client_id']
#         json_base = 'accent-trans'
        json_base = 'accent'
    except:
        other += 1
#         return 'false'
    try:
        transcript = transcript.strip()
    except:
        transcript = None
    
    path_to_mp3 = base_dir+'clips/'+mp3name
    total_processed += 1
    if 'mp3' not in mp3name:
        other += 1
        return 'false'
    if accent!=accent:
        accent = 'unlabelled'
#         json_base = 'trans'
#     if accent=='unlabelled' and transcript!=transcript:
#         other += 1
#         return false
    if type(up)!=int or type(down)!=int:
        other += 1
        return 'false'
#     if up<=down:
#         return 'false'
    if not(up>0):
        no_upvotes += 1
        return 'false'
    if not(os.path.isfile(path_to_mp3)):
        print(path_to_mp3)
        no_file += 1
        return 'false'
    json_name = json_base + '/'+accent+'.json'
#     json_name = 'non-validated.json'
    json_file = open(json_name, 'a+')
    if json_file.read().find(mp3name.replace('mp3', ''))!=-1:
        print("{} seen already".format(mp3name))
        already_seen_files += 1
        return 'false'
    wav_file = base_dir + 'wav/' + mp3name.replace('.mp3', '.wav')
    json_item = {'text': transcript, 'audio_filepath': wav_file, 'up':up, 'down':down, 'gender':gender,
                'client_id':client_id, 'accent':accent}
    json.dump(json_item, json_file)
    json_file.write('\n')
    json_file.close()
    if transcript!=transcript:
        no_transcript += 1
    elif accent == 'unlabelled':
        no_accent += 1
    else:
        valid += 1
    return 'true'

total_processed = 0
tsvs = [f.name for f in os.scandir(base_dir) if 'tsv' in f.name and f.name.split('.')[0] not in ['validated', 'reported']]
# tsvs = ['validated.tsv']
# print(tsvs)
tqdm.pandas()
# sys.exit()
for tsv_file in tsvs:
    print("started file", tsv_file)
    total_processed, already_seen_files, no_upvotes, no_accent, no_transcript, no_file, other, valid = [0]*8
    tsv_file = base_dir+tsv_file
    interviews_df = pd.read_csv(tsv_file, sep='\t', dtype='unicode',
                                usecols = ['path','sentence','up_votes',
                                           'down_votes', 'gender', 'accent', 'client_id', 'age'])
    interviews_df['added'] = interviews_df.progress_apply(lambda row:process_mcv(row), axis=1)
#     interviews_df.to_csv(tsv_file.replace('tsv','csv'))
    print("completed file {}".format(tsv_file))
