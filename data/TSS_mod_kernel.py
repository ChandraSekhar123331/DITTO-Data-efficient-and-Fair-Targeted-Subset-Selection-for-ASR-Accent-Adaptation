from __future__ import print_function
import os, sys
from pprint import pprint
import json
import argparse
import numpy as np
import pickle
from collections import Counter
import time
import time
import numpy as np
from submodlib.helper import create_kernel
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


def budget(budget_size):
    return int(4.92 * budget_size)


def get_base_eta(query_accent_feat, query_content_feat):
    accent_norm = np.linalg.norm(query_accent_feat, axis=1)
    content_norm = np.linalg.norm(query_content_feat, axis=1)

    ratio = np.mean(accent_norm / content_norm)
    return ratio


def compute_subset(
    dirs,
    base_dir,
    query_dir,
    ground_list,
    ground_accent_features,
    ground_content_features,
    ground_features_Y,
    query_list,
    query_accent_features,
    test_accent_features,
    greedyList,
    budget_size,
    target_size,
    accent,
    fxn,
    accent_similarity,
    content_similarity,
    accent_feature_type,
    content_feature_type,
    query_content_features,
    base_eta,
    etaScale,
):
    output_dir = os.path.join(
        base_dir,
        query_dir,
        "all",
        f"budget_{budget_size}",
        f"target_{target_size}",
        f"{fxn}_etabase_{round(base_eta, 2)}_etaScale_{round(etaScale, 2)}",
        f"accent_{accent_feature_type}",
        f"content_{content_feature_type}",
        f"accent_{accent_similarity}",
        f"content_{content_similarity}",
    )
    os.makedirs(output_dir, exist_ok=True)

    all_indices = [j[0] for j in greedyList]
    all_gains = [j[1] for j in greedyList]

    print("*******all_indices ******")
    print(all_indices)

    print("******all_gains*********")
    print(all_gains)

    total_duration, index = 0, 0
    while total_duration + ground_list[all_indices[index]]["duration"] <= budget(
        budget_size
    ):
        total_duration += ground_list[all_indices[index]]["duration"]
        index += 1
    #         print(index, all_indices[index], len(ground_list))
    total_count = index
    selected_indices = all_indices[:index]
    selected_gains = all_gains[:index]
    print("*******selected_indices ******")
    print(selected_indices)

    print("******selected_gains*********")
    print(selected_gains)
    selected_list = [ground_list[j] for j in selected_indices]

    train_list = selected_list

    accent_sample_count, accent_sample_duration = 0, 0
    for item in selected_list:
        if item["audio_filepath"].split("/")[-4] == accent:
            accent_sample_count += 1
            accent_sample_duration += item["duration"]
    total_selections = Counter(
        [item["audio_filepath"].split("/")[-4] for item in selected_list]
    )

    #        with open(base_dir + query_dir + f'train/error_model/{budget_size}/seed_{i}/train.json', 'w') as f:
    #            for line in train_list:
    #                f.write('{}\n'.format(json.dumps(line)))
    with open(f"{output_dir}/train.json", "w") as f:
        for line in train_list:
            f.write("{}\n".format(json.dumps(line)))

    # plots(dirs, run_dir, ground_features, ground_features_Y, query_features, test_features, selected_indices, selected_gains, fxn)

    print("\n subset computed .... \n")
    stats = "total selections: " + str(total_count)
    stats += "\ntotal duration: " + str(total_duration)
    stats += "\naccented selections: " + str(accent_sample_count)
    stats += "\naccented duration: " + str(accent_sample_duration)
    stats += "\nall selections: " + str(total_selections)

    with open(output_dir + "/stats.txt", "w") as f:
        f.write(stats)


def generate_greedyList(
    ground_kernel,
    query_kernel,
    query_query_kernel,
    fxn,
    budget_size,
    etaScale,
    base_eta,
):
    print(f"\ncreating {fxn} object\n")
    if fxn == "FL1MI":
        obj1 = FacilityLocationMutualInformationFunction(
            n=len(ground_kernel),
            num_queries=query_kernel.shape[1],
            query_sijs=query_kernel,
            data_sijs=ground_kernel,
            magnificationEta=etaScale / base_eta,
        )
    # elif fxn == "FL2MI":
    #     obj1 = FacilityLocationVariantMutualInformationFunction(
    #         n=len(ground_kernel),
    #         num_queries=query_kernel.shape[1],
    #         query_sijs=query_kernel,
    #         queryDiversityEta=etaValue,
    #     )
    # elif fxn == "GCMI":
    #     obj1 = GraphCutMutualInformationFunction(
    #         n=len(ground_kernel),
    #         num_queries=query_kernel.shape[1],
    #         query_sijs=query_kernel,
    #     )
    elif fxn == "LogDMI":
        obj1 = LogDeterminantMutualInformationFunction(
            n=len(ground_kernel),
            num_queries=query_kernel.shape[1],
            lambdaVal=1,  # should this also be tuned?
            query_sijs=query_kernel,
            data_sijs=ground_kernel,
            query_query_sijs=query_query_kernel,
            magnificationEta=etaScale / base_eta,
        )
    else:
        print("\n\n\n............... ERROR not a valid FUNCTION ............\n\n\n")
        exit()
    print(f"\n{fxn} object created\n")
    print("\ngenerating greedyList...\n")
    greedyList = obj1.maximize(
        budget=3 * budget_size,
        optimizer="LazyGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        epsilon=0.1,
        verbose=False,
        show_progress=True,
    )
    print("\n.... greedyList generated ... \n")
    return greedyList


def load_features(file_dir, feature_type):
    features = []
    file_name = file_dir.split("/")[-1]
    par_dir = "/".join(file_dir.split("/")[:-1]) + f"/{feature_type}/"
    with open(par_dir + file_name.replace(".json", f"_{feature_type}.file"), "rb") as f:
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


def preprocess(
    base_dir,
    target_size,
    budget_size,
    accent,
    accent_similarity,
    content_similarity,
    accent_feature_type,
    content_feature_type,
):
    dirs = [
        "kannada_male_english",
        "malayalam_male_english",
        "rajasthani_male_english",
        "hindi_male_english",
        "tamil_male_english",
        "gujarati_female_english",
        "manipuri_female_english",
        "assamese_female_english",
    ]

    query_dir = f"{accent}/"

    # Query List and # Query Features
    query_file_path = base_dir + query_dir + "seed.json"
    query_list = [json.loads(line.strip()) for line in open(query_file_path)]

    query_accent_features = load_features(query_file_path, accent_feature_type)
    query_content_features = load_features(query_file_path, content_feature_type)
    query_list, query_accent_features, query_content_features = (
        query_list[:target_size],
        query_accent_features[:target_size],
        query_content_features[:target_size],
    )

    # Ground List and Ground Accent Features, Ground Content Features
    ground_list, ground_list_Y, ground_accent_features, ground_content_features = (
        [],
        [],
        [],
        [],
    )
    for i, _dir in enumerate(dirs):
        selection_file_path = os.path.join(base_dir, _dir, "selection.json")
        selection_file_list = [
            json.loads(line.strip()) for line in open(selection_file_path)
        ]
        ground_list.extend(selection_file_list)

        ground_accent_features.append(
            load_features(selection_file_path, accent_feature_type)
        )
        ground_content_features.append(
            load_features(selection_file_path, content_feature_type)
        )

        ground_list_Y.extend([i] * len(selection_file_list))

    ground_accent_features = np.concatenate(ground_accent_features, axis=0)
    ground_content_features = np.concatenate(ground_content_features, axis=0)
    ground_features_Y = np.asarray(ground_list_Y).reshape(-1, 1)

    ### test file
    ## We only need test_accent_features
    test_file_path = base_dir + query_dir + "test.json"
    test_list = [json.loads(line.strip()) for line in open(test_file_path)]
    test_accent_features = load_features(test_file_path, accent_feature_type)

    print(
        f"Ground stats",
        f"Length = {len(ground_list)}",
        f"Accent features shape = {ground_accent_features.shape}",
        f"Content features shape = {ground_content_features.shape}",
    )
    print(
        f"Query stats",
        f"Length = {len(query_list)}",
        f"Accent features shape = {query_accent_features.shape}",
        f"Content features shape = {query_content_features.shape}",
    )
    print(
        f"Test stats",
        f"Length = {len(test_list)}",
        f"Accent features shape = {test_accent_features.shape}",
    )
    # print(len(query_list), query_features.shape)
    # print(len(test_list), test_features.shape)

    print("creating kernels ....")
    t1 = time.time()
    print(
        create_kernel(
            np.array([[1, 1], [1, 5]]), metric=content_similarity, mode="dense"
        )
    )
    print(f"content_similarity is {content_similarity}")
    # exit(1)
    ground_kernel = create_kernel(
        ground_content_features, metric=content_similarity, mode="dense"
    )
    # ground_kernel = create_kernel(
    #     ground_accent_features, metric=accent_similarity, mode="dense"
    # )
    query_kernel = create_kernel(
        query_accent_features,
        metric=accent_similarity,
        mode="dense",
        X_rep=ground_accent_features,
    )
    query_query_kernel = create_kernel(
        query_content_features,
        metric=content_similarity,
        mode="dense",
        X_rep=query_content_features,
    )

    print("************ ground kernel*************")
    print(ground_kernel)
    print("************ query kernel*************")
    print(query_kernel)
    print("************ query query kernel*************")
    print(query_query_kernel)
    t2 = time.time()
    print("kernel creation done ....", t2 - t1)

    print("ground_kernel: ", ground_kernel.shape)
    print("query_kernel: ", query_kernel.shape)
    print("query_query_kernel: ", query_query_kernel.shape)

    base_eta = get_base_eta(
        query_accent_feat=query_accent_features,
        query_content_feat=query_content_features,
    )

    return (
        dirs,
        query_dir,
        ground_list,
        ground_accent_features,
        ground_content_features,
        ground_features_Y,
        ground_kernel,
        query_list,
        query_accent_features,
        query_kernel,
        query_query_kernel,
        test_accent_features,
        query_content_features,
        base_eta,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSS input")
    parser.add_argument("--budget", type=int, help="budget")
    parser.add_argument("--target", type=int, help="target")
    parser.add_argument("--eta_scale", type=float, help="scale on value")
    parser.add_argument(
        "--accent_similarity", type=str, help="accent_similarity metric"
    )
    parser.add_argument(
        "--content_similarity", type=str, help="content_similarity_metric"
    )
    parser.add_argument("--fxn", type=str, help="function")
    parser.add_argument("--accent", type=str, help="query set")
    parser.add_argument(
        "--accent_feature_type", type=str, help="which accent feature space"
    )
    parser.add_argument(
        "--content_feature_type", type=str, help="which content diversity feature space"
    )

    args = parser.parse_args()

    pprint(vars(args))

    budget_size = args.budget
    target = args.target
    eta_scale = args.eta_scale
    accent_similarity = args.accent_similarity
    content_similarity = args.content_similarity
    fxn = args.fxn
    accent = args.accent
    accent_feature_type = args.accent_feature_type
    content_feature_type = args.content_feature_type

    # parser.add_argument('--base_dir_data', type = str, help = '')

    #     args = parser.parse_args()
    budget_size = budget_size  # args.budget
    target_size = target  # args.target
    etaScale = eta_scale  # args.eta
    accent_similarity = accent_similarity  # args.accent_similarity
    content_similarity = content_similarity  # args.content_similarity
    fxn = fxn  # args.fxn
    accent = accent  # args.accent
    accent_feature_type = accent_feature_type  # args.accent_feature_type
    content_feature_type = content_feature_type  # args.content_feature_type
    base_dir = ""

    (
        dirs,
        query_dir,
        ground_list,
        ground_accent_features,
        ground_content_features,
        ground_features_Y,
        ground_kernel,
        query_list,
        query_accent_features,
        query_kernel,
        query_query_kernel,
        test_accent_features,
        query_content_features,
        base_eta,
    ) = preprocess(
        base_dir,
        target_size,
        budget_size,
        accent,
        accent_similarity,
        content_similarity,
        accent_feature_type,
        content_feature_type,
    )

    greedyList = generate_greedyList(
        ground_kernel,
        query_kernel,
        query_query_kernel,
        fxn,
        budget_size,
        etaScale,
        base_eta,
    )
    compute_subset(
        dirs,
        base_dir,
        query_dir,
        ground_list,
        ground_accent_features,
        ground_content_features,
        ground_features_Y,
        query_list,
        query_accent_features,
        test_accent_features,
        greedyList,
        budget_size,
        target_size,
        accent,
        fxn,
        accent_similarity,
        content_similarity,
        accent_feature_type,
        content_feature_type,
        query_content_features,
        base_eta,
        etaScale,
    )
