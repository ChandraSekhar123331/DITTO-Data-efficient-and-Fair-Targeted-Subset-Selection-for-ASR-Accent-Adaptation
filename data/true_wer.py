import argparse
import json
import os
import random
from pprint import pprint

import numpy as np


def get_budget(budget_size):
    return int(4.92 * budget_size)


def parse_args():
    parser = argparse.ArgumentParser(description="True wer based diversity")
    parser.add_argument(
        "--inp_json",
        type=str,
        required=True,
        help="input json file from which we subset select",
    )
    parser.add_argument(
        "--budget", type=int, required=True, help="budget(i.e. number of selections)"
    )
    parser.add_argument(
        "--exp_id",
        type=int,
        required=True,
        help="experiment id. The output path depends on this",
    )
    args = parser.parse_args()
    return args


def main(args):

    inp_json_file = args.inp_json
    budget = args.budget
    total_budget = get_budget(args.budget)
    exp_id = args.exp_id
    seed = args.exp_id

    random.seed(seed)
    np.random.seed(seed)

    inp_json_dir = "/".join(inp_json_file.split("/")[:-1])
    output_json_dir = os.path.join(
        inp_json_dir, "budget_{}".format(budget), "true_wer", "run_" + str(exp_id)
    )
    output_json_file = os.path.join(output_json_dir, "train.json")

    with open(inp_json_file) as inp_file:
        data = [json.loads(line.strip()) for line in inp_file.readlines()]
        data.sort(key=lambda x: x["pretrained_wer"], reverse=True)

    pprint("Printing Sample WER data in sorted order")
    pprint(data[:5])

    pprint(f"Started writing to file: {output_json_file}")
    os.makedirs(output_json_dir, exist_ok=True)
    with open(output_json_file, "w") as out_file:
        budget_taken = 0
        for line in data:
            budget_taken += line["duration"]

            out_file.write(json.dumps(line))
            out_file.write("\n")

            if budget_taken >= total_budget:
                break

    pprint(f"Writing to file: {output_json_file} is done")


#     print("loading data....")
#     data, input_file_lines = load_phoneme_seq_from_pseudo_transcripts(selection_json_file, remove_duplicates=False)


#     print("**********Sample phoneme data******")
#     print(data[0])
#     print("data_size: {}".format(len(data)))

#     print("**** Converting phonemes to ids*****")

#     corpus = []
#     for phoneme_seq in data:
#         corpus.append(" ".join(list(map(str, convert_phonemes_to_ids(phoneme_seq)))))

#     print("Sample phonemes converted to ids")
#     print(corpus[0])

#     vectorizer = TfidfVectorizer(lowercase=False, token_pattern='(?u)\\b\\w+\\b', ngram_range=(1,3))
#     data = vectorizer.fit_transform(corpus).toarray()
#     # print(vectorizer.get_feature_names_out())
#     # print(len(vectorizer.get_feature_names_out()))

#     print("TF-IDF vectorised data")
#     print(data[0])
#     print(f"shape of TF-IDF data is = {data.shape}")

#     objFL = FacilityLocationFunction(n = data.shape[0], data=data, mode="dense", metric = "euclidean")
#     pprint(f"Printing the gains and ids of top {budget} samples")
#     pprint(f"The selections are based on TF-IDF based features")

#     result = objFL.maximize(budget=budget, optimizer="NaiveGreedy")
#     pprint(result)

#     pprint("Started writing selections")
#     output_dir = os.path.split(output_json_file)[0]
#     pprint(f"Checking and creating directory = {output_dir}")
#     os.makedirs(output_dir,exist_ok=True)
#     pprint(f"Creating directory done")

#     pprint("Writing the selections to {output_json_file}")
#     with open(output_json_file, 'w') as output_file:
#             for sample in result:
#                 output_file.write(input_file_lines[sample[0]])


if __name__ == "__main__":
    args = parse_args()
    pprint(vars(args))
    main(args)
