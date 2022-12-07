from __future__ import print_function
import os
import json
import argparse
import statistics
from collections import Counter
import numpy as np
import pickle
from submodlib import FacilityLocationFunction
from pprint import pprint


def budget(budget_size):
    return int(4.92 * budget_size)


def load_features(base_dir):
    feature_type = 'w2v2'
    feature_file = os.path.join(base_dir, "train_{}.file".format(feature_type))
    features = []
    with open(feature_file, 'rb') as f:
        while True:
            try:
                features.append(pickle.load(f))
            except EOFError:
                break
#             if features[-1].shape[1]<39:
#                 print(features[-1].shape)
#                 print(file_dir, "bad feature file")
#                 return np.array([[]])
#                 sys.exit()
    features = np.concatenate(features, axis=0)
    print(features.shape)
    return features


def calculate_SubMod_scores(data):
    objFL = FacilityLocationFunction(n = data.shape[0], data=data, mode="dense", metric = "euclidean")

    

    result = objFL.maximize(budget=data.shape[0] - 1, optimizer="NaiveGreedy")
    pprint(f"Printing the gains and ids of top {50} samples")
    pprint(f"The selections are based on W2V2 based features")
    pprint(result[:50])

    return result
    

def select(base_dir, budget_size):
    accents = [
        "kannada_male_english",
        "malayalam_male_english",
        "rajasthani_male_english",
        "hindi_male_english",
        "tamil_male_english",
        "gujarati_female_english",
        "manipuri_female_english",
        "assamese_female_english",
    ]

    json_file = os.path.join(base_dir, "train.json")
    

    train_features = load_features(base_dir)
        
    
    ground_list, ground_list_Y = [], []
    selection_file_list = [json.loads(line.strip()) for line in open(json_file)]
    ground_list.extend(selection_file_list)
    ground_list_Y.extend([0] * len(selection_file_list))

    print("ground len:{}".format(len(ground_list)))

    output_dir = "budget_{}/w2v2_avg/".format(budget_size)
    output_dir = os.path.join(base_dir, output_dir)

    list_total_selection, list_total_count, list_total_duration = [], [], []
    list_accent_sample_count, list_accent_sample_duration = {}, {}
    for accent in accents:
        list_accent_sample_count[accent] = []
        list_accent_sample_duration[accent] = []
    SM_result = calculate_SubMod_scores(train_features)

    for i in [1, 2, 3]:
        run = f"run_{i}"
        run_dir = os.path.join(output_dir, run)
        os.makedirs(run_dir, exist_ok=True)


        all_indices = list(range(len(ground_list)))
        total_duration = 0
        selected_indices = []
        for SM_index, SM_score in SM_result:
            total_duration += ground_list[SM_index]["duration"]
            selected_indices.append(SM_index)
        
            if total_duration >= budget(budget_size):
                break
        
        num_selections = len(selected_indices)
        list_total_count.append(num_selections)
        list_total_duration.append(total_duration)
        selected_list = [ground_list[j] for j in selected_indices]

        train_list = selected_list

        for accent in accents:
            accent_sample_count, accent_sample_duration = 0, 0
            for item in selected_list:
                if item["audio_filepath"].split("/")[-4] == accent:
                    accent_sample_count += 1
                    accent_sample_duration += item["duration"]
            list_accent_sample_count[accent].append(accent_sample_count)
            list_accent_sample_duration[accent].append(accent_sample_duration)
        list_total_selection.append(
            Counter([item["audio_filepath"].split("/")[-4] for item in selected_list])
        )

        with open(f"{run_dir}/train.json", "w") as f:
            for line in train_list:
                f.write("{}\n".format(json.dumps(line)))




    stats = (
        "total selection : "
        + " ".join(map(str, list_total_count))
        + " -> {0:.2f}\n".format(statistics.mean(list_total_count))
    )
    stats += (
        "total selection duration: "
        + " ".join(map(str, list_total_duration))
        + " -> {0:.2f}\n".format(statistics.mean(list_total_duration))
    )
    for accent in accents:
        stats += "\naccent: {}\n".format(accent)
        stats += (
            "accented selection: "
            + " ".join(map(str, list_accent_sample_count[accent]))
            + " -> {0:.2f}\n".format(statistics.mean(list_accent_sample_count[accent]))
        )
        stats += (
            "accented duration: "
            + " ".join(map(str, list_accent_sample_duration[accent]))
            + " -> {0:.2f}\n".format(
                statistics.mean(list_accent_sample_duration[accent])
            )
        )
    stats += "\nall selections: " + str(list_total_selection)

    with open(os.path.join(output_dir, "stats.txt"), "w") as f:
        f.write(stats)


parser = argparse.ArgumentParser(description="TSS input")
parser.add_argument("--budget", type=int, help="budget", required=True)
parser.add_argument(
    "--file_dir", type=str, help="dir where json file is", required=True
)

args = parser.parse_args()
budget_size = args.budget
base_dir = args.file_dir

select(base_dir, budget_size)
