# This code is 99% same as entropy.py file

import pickle
import numpy as np
from scipy.stats import entropy
from scipy.special import softmax
import argparse
import matplotlib.pyplot as plt

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

def calculate_entropy(logit_array, method = "mean", k = 100):
    # Assumes the logit array is of the form (time * vocab_size)
    vocab_size = 29
    prob = logit_array
    assert(prob.shape[0] == vocab_size)
    entr = entropy(prob)
        
    selected_entropies = np.sort(entr)[-int(k*entr.size):]
    if method == "mean":
        return np.mean(selected_entropies)
    elif method == "median":
        return np.median(selected_entropies)        
    else:
        print(f"Unknown Method = {method}")

def calculate_error_rates(wers_file):
    with open(wers_file) as f:
        lines = [line.strip() for line in f.readlines()]
        wer_ls = []
        cer_ls = []
        for line in lines:
            if line.startswith("WER:"):
                wer_ls.append(float(line.split(":")[1].strip()))
            elif line.startswith("CER:"):
                cer_ls.append(float(line.split(":")[1].strip()))
            else:
                continue
        return {
            "WER": wer_ls,
            "CER": cer_ls
        }
    
def get_args():
    
    parser = argparse.ArgumentParser()
    
#     parser.add_argument("--train_json", type=str, required=True,
#                         help="path to the json which yielded logits_file")
    parser.add_argument("--method" , type=str , choices=["mean", "median"], help="Method to use to calculate aggreg entropy")
    parser.add_argument("--top_k_percentile", type=int, help = "k in top-k_percentile used to calculate the aggreg entropy")
    parser.add_argument("--logits_file", type=str, required=True,
                        help="path to the logits file dumped by inference.py")
    parser.add_argument("--wers_file", type=str, required=True,
                        help="path to the wers file dumped by inference.py")
#     parser.add_argument("--selected_json", type=str, required=True, help="path where the entropy selections has to be dumped")
#     parser.add_argument("--selection_count", type=int, required=True,
#                         help="maximum number of samples that entropy model should select")

    return vars(parser.parse_args())


if __name__ == "__main__":
#     selection_budget = 100
    args = get_args()
    print(args)
#     selection_budget = args["selection_count"]
    logits_file = args["logits_file"]
    wers_file = args["wers_file"]
    method = args["method"]
    top_k_percentile = args["top_k_percentile"]
#     train_json = args["train_json"]
#     selected_json = args["selected_json"]
    
    
    
    logit_ls = read_pickle_file(logits_file)
    entropies = [calculate_entropy(_, method = method, k = top_k_percentile) for _ in logit_ls]
    wer_ls = calculate_error_rates(wers_file)["WER"]
    print(f"Argmax of wer_ls is = {np.argmax(wer_ls)}")
    cer_ls = calculate_error_rates(wers_file)["CER"]
    plt.scatter(x=wer_ls, y=entropies)
    plt.xlabel("WER")
    plt.ylabel("entropy")
    plt.savefig(wers_file.split('.')[-2]+f"_{method}_{top_k_percentile}_wer_entropy_scatter.png")
    plt.clf()
    plt.scatter(x=cer_ls, y=entropies)
    plt.xlabel("CER")
    plt.ylabel("entropy")
    plt.savefig(wers_file.split('.')[-2]+f"_{method}_{top_k_percentile}_cer_entropy_scatter.png")
    plt.clf()
#     print(np.corrcoef(entropies, wer_ls))
    print(f"WER corrln is: {np.corrcoef(entropies, wer_ls)[0][1]}")
#     print(np.corrcoef(entropies, cer_ls))
    print(f"CER corrln is: {np.corrcoef(entropies, cer_ls)[0][1]}")
#     sorted_idx = np.argsort(entropies)
    
#     selected_idx = list(sorted_idx[-min(selection_budget, len(sorted_idx)):])
#     selected_idx.reverse()
#     selected_entropies = [entropies[x] for x in selected_idx]
#     print(selected_entropies)
#     with open(train_json) as json_file:
#         lines = json_file.readlines()
#         selected_lines = [lines[idx] for idx in selected_idx]
#         with open(selected_json, 'w') as entropy_file:
#             entropy_file.writelines(selected_lines)