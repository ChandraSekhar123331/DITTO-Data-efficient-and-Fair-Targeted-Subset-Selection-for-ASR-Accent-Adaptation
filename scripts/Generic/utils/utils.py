import json
import os
import pickle

import numpy as np
from submodlib import FacilityLocationFunction, LogDeterminantFunction
from submodlib.functions.facilityLocationMutualInformation import (
    FacilityLocationMutualInformationFunction,
)
from submodlib.functions.facilityLocationVariantMutualInformation import (
    FacilityLocationVariantMutualInformationFunction,
)
from submodlib.functions.graphCutMutualInformation import (
    GraphCutMutualInformationFunction,
)
from submodlib.functions.logDeterminantMutualInformation import (
    LogDeterminantMutualInformationFunction,
)
from submodlib.helper import create_kernel
from utils.dataset import get_accent


def sample_greedy(lines, required_duration):
    selected_duration = 0
    index = 0
    selected_lines = []

    while selected_duration < required_duration:
        selected_lines.append(lines[index])
        selected_duration += lines[index]["duration"]
        index += 1

    return selected_lines


def shuffle(lines, rng):
    indices = list(range(len(lines)))
    rng.shuffle(indices)  # shuffles indices in-place
    return [lines[index] for index in indices]


def sample_random(input_lines, required_duration, seed):

    rng = np.random.default_rng(seed=seed)
    shuffled_lines = shuffle(input_lines, rng)
    selected_duration = 0
    index = 0
    selected_lines = []

    while selected_duration < required_duration:
        selected_lines.append(shuffled_lines[index])
        selected_duration += shuffled_lines[index]["duration"]
        index += 1

    return selected_lines


def dump_lines(lines, OUTPUT_JSON_PATH):
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w") as out_file:
        for line in lines:
            out_file.write(json.dumps(line))
            out_file.write("\n")


def read_lines(INPUT_JSON_PATH):
    return [json.loads(line.strip()) for line in open(INPUT_JSON_PATH)]


def read_python_obj(file_path):
    with open(file_path, "rb") as file:
        obj = pickle.load(file)
    return obj


def read_python_list_obj(file_path):
    lst = []
    with open(file_path, "rb") as file:
        while True:
            try:
                obj = pickle.load(file)
                lst.append(obj)
            except:
                break
    return lst


def dump_python_obj(obj, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)
    return


def build_kernel(feat1, feat2, similarity):
    return create_kernel(X=feat1, X_rep=feat2, metric=similarity, mode="dense")


def maximise_SMI(SMI_obj, budget):
    return SMI_obj.maximize(
        budget=budget,
        optimizer="LazyGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        epsilon=0.1,
        verbose=False,
        show_progress=True,
    )


def build_SMI_OBJ(ground_ground, query_ground, query_query, fxn, eta):
    if fxn == "FL1MI":
        return FacilityLocationMutualInformationFunction(
            n=len(ground_ground),
            num_queries=query_ground.shape[1],
            query_sijs=query_ground,
            data_sijs=ground_ground,
            magnificationEta=eta,
        )
    elif fxn == "LogDMI":
        return LogDeterminantMutualInformationFunction(
            n=len(ground_ground),
            num_queries=query_ground.shape[1],
            lambdaVal=1,  # should this also be tuned?
            query_sijs=query_ground,
            data_sijs=ground_ground,
            query_query_sijs=query_query,
            magnificationEta=eta,
        )
    elif fxn == "GCMI":
        return GraphCutMutualInformationFunction(
            n=len(ground_ground),
            num_queries=query_ground.shape[1],
            query_sijs=query_ground,
        )
    elif fxn == "FL2MI":
        return FacilityLocationVariantMutualInformationFunction(
            n=len(ground_ground),
            num_queries=query_ground.shape[1],
            query_sijs=query_ground,
            queryDiversityEta=eta,
        )
    else:
        raise ValueError(
            "Given fxn not in {}".format(str(["FL1MI", "GCMI", "LogDMI", "FL2MI"]))
        )


def build_SM_OBJ(features, fxn, sim, lambdaVal, kernel):
    if fxn == "FacLoc":
        obj = FacilityLocationFunction(
            n=features.shape[0],
            data=features,
            mode="dense",
            metric=sim,
            sijs=kernel,
            separate_rep=False,
        )
    elif fxn == "LogDet":
        obj = LogDeterminantFunction(
            n=features.shape[0],
            data=features,
            mode="dense",
            metric=sim,
            lambdaVal=lambdaVal,
            sijs=kernel,
        )
    else:
        raise ValueError("Given fxn not in {}".format(str(["FacLoc", "LogDet"])))

    return obj


def maximise_SM(SM_obj, budget):
    return SM_obj.maximize(budget=budget, optimizer="NaiveGreedy", show_progress=True)


# def load_single_feature(line, dataset, FULL_DATASET_PATH, feature_type):
#     accent = get_accent(line, dataset)

#     FEATURE_PATH = os.path.join(
#         FULL_DATASET_PATH, accent, "features", feature_type, "all.file"
#     )
#     with open(FEATURE_PATH, "rb") as file:
#         feature_dict = pickle.load(file)
#     return feature_dict[line["audio_filepath"]]


def load_features(json_lst, dataset, FULL_DATASET_PATH, feature_type):
    all_accents = set([get_accent(line, dataset) for line in json_lst])
    accent_features_dict = {
        accent: pickle.load(
            open(
                os.path.join(
                    FULL_DATASET_PATH, accent, "features", feature_type, "all.file"
                ),
                "rb",
            )
        )
        for accent in all_accents
    }
    # print(accent_features_dict)
    # print("HELLO", flush=True)
    # print([get_accent(line, dataset) for line in json_lst])
    return np.concatenate(
        [
            accent_features_dict[get_accent(line, dataset)][line["audio_filepath"]]
            for line in json_lst
        ],
        axis=0,
    )
