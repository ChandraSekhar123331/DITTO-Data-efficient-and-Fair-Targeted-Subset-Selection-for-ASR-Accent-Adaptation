from __future__ import print_function
import os
import json
import argparse
import statistics
import numpy as np
import random
from collections import Counter
import numpy as np

def budget(budget_size):
    return int(4.92*budget_size)


def round_robin_arrange(ground_list):
    cont = dict()

    for json_item in ground_list:
        speaker_id = json_item["client_id"]
        if speaker_id not in cont.keys():
            cont[speaker_id] = [json_item]
        else:
            cont[speaker_id].append(json_item)

    shuffled_users = list(cont.keys())
    random.shuffle(shuffled_users)
    inds = [0 for _ in range(len(shuffled_users))]

    result_list = []

    while shuffled_users:
        new_shuffled_users = []
        new_inds = []
        for user_id, index in zip(shuffled_users, inds):
            assert index < len(cont[user_id])
            result_list.append(cont[user_id][index])
            # json.dump(cont[user_id][index], outp_file)
            # outp_file.write("\n")

            if index + 1 < len(cont[user_id]):
                new_shuffled_users.append(user_id)
                new_inds.append(index + 1)

        shuffled_users = new_shuffled_users

        inds = new_inds

    assert len(ground_list) == len(result_list)
    return result_list


def select(base_dir, budget_size):
    accents = [
        'african', 'australia', 'bermuda', 
        'canada', 'england', 'hongkong', 
        'indian', 'ireland', 'malaysia', 
        'newzealand', 'philippines', 'scotland', 
        'singapore', 'southatlandtic', 'us', 
        'wales'
    ]
    
    json_file = os.path.join(base_dir, 'train.json')

    ground_list, ground_list_Y= [], []
    selection_file_list = [json.loads(line.strip()) for line in open(json_file)]
    ground_list.extend(selection_file_list)
    ground_list_Y.extend([0]*len(selection_file_list))

    print("ground len:{}".format(len(ground_list)))

    output_dir = 'budget_{}/diverse_speakers/'.format(budget_size)
    output_dir = os.path.join(base_dir, output_dir)
    
    list_total_selection, list_total_count, list_total_duration = [], [], []
    list_accent_sample_count, list_accent_sample_duration = {}, {}
    for accent in accents:
        list_accent_sample_count[accent] = []
        list_accent_sample_duration[accent] = []
    
    for i in [1, 2, 3]:
        random.seed(i)
        run = f'run_{i}'
        run_dir = os.path.join(output_dir, run)
        os.makedirs(run_dir, exist_ok=True)
        # for folder in ['train', 'output', 'plots']:
        #     os.makedirs(os.path.join(run_dir, folder))
        all_indices = list(range(len(ground_list)))
        random.seed(i)
        random.shuffle(all_indices)

        temp_list = [ground_list[ind] for ind in all_indices]

        round_robin_list = round_robin_arrange(temp_list)

        all_indices = list(range(len(round_robin_list)))


        total_duration, index = 0, 0
        while total_duration + round_robin_list[all_indices[index]]['duration'] <= budget(budget_size):
            total_duration += round_robin_list[all_indices[index]]['duration']
            index += 1
        list_total_count.append(index)
        list_total_duration.append(total_duration)
        selected_indices = all_indices[:index]
        selected_list = [round_robin_list[j] for j in selected_indices]

#        train_list = selected_list + query_list
        train_list = selected_list
        
        for accent in accents:
            accent_sample_count, accent_sample_duration = 0, 0
            for item in selected_list:
                if item['audio_filepath'].split('/')[-4] == accent:
                    accent_sample_count += 1
                    accent_sample_duration += item['duration']
            list_accent_sample_count[accent].append(accent_sample_count)
            list_accent_sample_duration[accent].append(accent_sample_duration)
        list_total_selection.append(Counter([item['audio_filepath'].split('/')[-4] for item in selected_list]))
        
#        with open(base_dir + query_dir + f'train/error_model/{budget_size}/seed_{i}/train.json', 'w') as f:
#            for line in train_list:
#                f.write('{}\n'.format(json.dumps(line)))
        with open(f'{run_dir}/train.json', 'w') as f:
            for line in train_list:
                f.write('{}\n'.format(json.dumps(line)))
                
#        plots(dirs, run_dir, ground_features, ground_features_Y, query_features, test_features, selected_indices)
    
    stats = 'total selection : ' + ' '.join(map(str, list_total_count)) + ' -> {0:.2f}\n'.format(statistics.mean(list_total_count))
    stats += 'total selection duration: ' + ' '.join(map(str, list_total_duration)) + ' -> {0:.2f}\n'.format(statistics.mean(list_total_duration))
    for accent in accents:
        stats += '\naccent: {}\n'.format(accent)
        stats += 'accented selection: ' + ' '.join(map(str, list_accent_sample_count[accent])) + ' -> {0:.2f}\n'.format(statistics.mean(list_accent_sample_count[accent]))
        stats += 'accented duration: ' + ' '.join(map(str, list_accent_sample_duration[accent])) + ' -> {0:.2f}\n'.format(statistics.mean(list_accent_sample_duration[accent]))
    stats += '\nall selections: ' + str(list_total_selection)
    
    with open(os.path.join(output_dir, 'stats.txt'), 'w') as f:
        f.write(stats)

parser = argparse.ArgumentParser(description ='TSS input')
parser.add_argument('--budget', type = int,  help ='budget')
parser.add_argument('--file_dir', type = str,  help ='dir where json file is')

args = parser.parse_args()
budget_size = args.budget
base_dir = args.file_dir

select(base_dir, budget_size)
