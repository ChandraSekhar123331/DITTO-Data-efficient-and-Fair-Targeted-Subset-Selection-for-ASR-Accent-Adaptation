import librosa, glob, sys, json, pickle, os
from tqdm import tqdm
import pandas as pd, numpy as np
from pydub import AudioSegment

base_dir = './validated/'
mp3base_dir = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mozilla/cv-corpus-7.0-2021-07-21/en/'
accent_files, filenames_in_accent = {}, {}
ac, tr, actr, cnt, voterr = 0, 0, 0, 0, 0

def process_mcv(row):
    global ac, tr, actr, cnt, voterr
    try:
        mp3name, transcript, up, down = row['path'], row['sentence'], int(row['up_votes']), int(row['down_votes'])
        gender, accent = row['gender'], row['accent']
        client_id = row['client_id']
        json_base = 'accent-trans'
    except:
        return 'false'
    
    try:
        transcript = transcript.strip()
    except:
        transcript = None
    
    # with open('./validated/validated.tsv') as mother:
    #     if mother.read().find(mp3name)!=-1:
    #         print("{} seen already".format(mp3name))
    #         print("STOP!")
    #         sys.exit()

    # if up>down:
    #     print(row)
    #     sys.exit()

    if accent!=accent:
        accent = 'unlabelled'
        json_base = 'trans'
    if accent=='unlabelled' and transcript!=transcript: 
        print(row)
        sys.exit()
        return 'false'
    elif transcript!=transcript:
        json_base = 'accent '
    if type(up)!=int or type(down)!=int: return 'false'
    if up<=0 or up<=down:
        if accent == 'unlabelled': return 'false'
        else:
            if json_base == 'accent-trans': voterr+=1 
            json_base = 'accent'

    if 'mp3' not in mp3name: return 'false'
    path_to_mp3 = mp3base_dir+'clips/'+mp3name

    if not(os.path.isfile(path_to_mp3)):
        print(path_to_mp3)
        return 'false'
    
    cnt += 1
    if json_base == 'accent': ac+=1
    elif json_base == 'accent-trans': actr+=1
    else: tr += 1 

    json_name = json_base + '/'+accent+'.json'
    json_path = base_dir + json_name
    json_file = open(json_path, 'a+')

    wav_file = path_to_mp3.replace('.mp3', '.wav').replace('/clips/', '/wav/')
    json_item = {'text': transcript, 'audio_filepath': wav_file, 'up':up, 'down':down, 'gender':gender,
                'client_id':client_id, 'accent':accent}
    
    json.dump(json_item, json_file)
    json_file.write('\n')
    json_file.close()
    
    return 'true'

total_processed = 0
tsvs = [f.name for f in os.scandir(base_dir) if 'tsv' in f.name]

tqdm.pandas()

for tsv_file in tsvs:
# for tsv_file in [1]:
    print("started file", tsv_file)
    total_processed, already_seen_files, no_upvotes, no_accent, no_transcript, no_file, other, valid = [0]*8
    tsv_file = base_dir+tsv_file
    # tsv_file = mp3base_dir+'train.tsv'
    interviews_df = pd.read_csv(tsv_file, sep='\t', dtype='unicode',
                                usecols = ['path','sentence','up_votes',
                                           'down_votes', 'gender', 'accent', 'client_id', 'age'])
    interviews_df['added'] = interviews_df.progress_apply(lambda row:process_mcv(row), axis=1)
#     interviews_df.to_csv(tsv_file.replace('tsv','csv'))
    print("completed file {}".format(tsv_file))

print("total output: {}, ac:{}, ac-tr:{}, tr:{}, voterr:{}".format(cnt, ac, actr, tr, voterr))