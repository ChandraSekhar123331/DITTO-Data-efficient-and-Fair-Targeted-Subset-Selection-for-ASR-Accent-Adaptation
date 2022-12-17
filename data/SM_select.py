import json
import os
import argparse
import pickle
import numpy as np
from submodlib import FacilityLocationFunction
from submodlib import LogDeterminantFunction
from pprint import pprint
from math import floor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fxn",
        required=True,
        choices=[
            "FacLoc",
            # "DispMin",
            "LogDet",
            # "SetCover",
            # "ProbSetCover",
            # "GraphCut",
            # "FeatBased",
            # "DispSum",
            # "SetFunc",
        ],
        type=str,
    )

    parser.add_argument("--accent", type=str, required=True)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--similarity", type=str, choices=["cosine", "euclidean"], required=True)

    return vars(parser.parse_args())



def load_features(file_name, feature_type, accent):
    file_name = os.path.join(".", accent, feature_type, f"{file_name}_{feature_type}.file")
    features = []
    with open(file_name, "rb") as file:
        while True:
            try:
                features.append(pickle.load(file))
            except EOFError:
                break
    print(features[-1].shape)
    features = np.concatenate(features, axis=0)
    return features



def SM_select(config):
    input_json_path = os.path.join(".", config["accent"], "selection.json")
    lines = [json.loads(line.strip()) for line in open(input_json_path)]


    features = load_features("selection", config["features"], config["accent"])

    assert(len(features) == len(lines))


    fxn = config["fxn"]

    if(fxn == "FacLoc"):
        obj = FacilityLocationFunction(n = features.shape[0], data=features, mode="dense", metric = config["similarity"])
    elif(fxn == "LogDet"):
        obj = LogDeterminantFunction(n = features.shape[0], data=features, mode="dense", metric = config["similarity"], lambdaVal=1)
    else:
        print(f'\n\n\n ***************** fxn = {fxn} is not valid ********************* \n\n\n')
        exit(-1)

    result = obj.maximize(budget = 3 * config["budget"], optimizer="NaiveGreedy")
    pprint(result)


    for run in range(1, 4):

        output_dir = os.path.join(".", config["accent"], "all", "budget_{}".format(config["budget"]), "SM_select", config["fxn"], config["features"], config["similarity"], f"run_{run}")

        pprint("Started writing selections")
        pprint(f"Checking and creating directory = {output_dir}")
        os.makedirs(output_dir,exist_ok=True)
        pprint(f"Creating directory done")
        output_json_file = os.path.join(output_dir, "train.json")
        pprint(f"Writing the selections to {output_json_file}")
        taken_duration = 0
        budget_seconds = floor(4.92 * config["budget"])
        with open(output_json_file, 'w') as output_file:
            for sample in result:
                json_line = lines[sample[0]]
                taken_duration += json_line["duration"]
                output_file.write(json.dumps(json_line))
                output_file.write("\n")
                if(taken_duration >= budget_seconds):
                    break

    



    



if __name__ == "__main__":
    config = get_args()

    SM_select(config)


