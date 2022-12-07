import pickle
import numpy as np
from scipy.stats import entropy
from scipy.special import softmax
import argparse

def read_pickle_file(file_name):
    pickle_file = open(file_name, "rb")
    objects = []
    while True:
        try:
            objects.append(pickle.load(pickle_file))
        except EOFError:
            break
    # each element in objects is a list of numpy arrays
    result = []
    for obj in objects:
        result.extend([softmax(logit_array.T, axis = 0) for logit_array in obj])
    pickle_file.close()
    return result

def calculate_entropy(logit_array, method = "mean"):
    # Assumes the logit array is of the form (time * vocab_size)
    vocab_size = 29
    prob = logit_array
    assert(prob.shape[0] == vocab_size)
    entr = entropy(prob)
    if method == "mean":
        return np.mean(entr)
    elif method == "median":
        return np.median(entr)
    else:
        print(f"Unknown Method = {method}")

    
    
def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_json", type=str, required=True,
                        help="path to the json which yielded logits_file")
    parser.add_argument("--logits_file", type=str, required=True,
                        help="path to the logits file dumped by inference.py")
    parser.add_argument("--selected_json", type=str, required=True, help="path where the entropy selections has to be dumped")
    parser.add_argument("--selection_count", type=int, required=True,
                        help="maximum number of samples that entropy model should select")

    return vars(parser.parse_args())


if __name__ == "__main__":
#     selection_budget = 100
    args = get_args()
    selection_budget = args["selection_count"]
    logits_file = args["logits_file"]
    train_json = args["train_json"]
    selected_json = args["selected_json"]
    
    
    logit_ls = read_pickle_file(logits_file)
    entropies = [calculate_entropy(_, "mean") for _ in logit_ls]
    sorted_idx = np.argsort(entropies)
    
    selected_idx = list(sorted_idx[-min(selection_budget, len(sorted_idx)):])
    selected_idx.reverse()
    selected_entropies = [entropies[x] for x in selected_idx]
    print(selected_entropies)
    with open(train_json) as json_file:
        lines = json_file.readlines()
        selected_lines = [lines[idx] for idx in selected_idx]
        with open(selected_json, 'w') as entropy_file:
            entropy_file.writelines(selected_lines)