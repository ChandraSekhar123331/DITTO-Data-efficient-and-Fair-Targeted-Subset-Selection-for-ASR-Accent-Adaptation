import argparse
from cmath import isclose
import numpy as np
import pickle
from pprint import pprint
import os


def load_features(file_name):
    features = []
    with open(file_name, "rb") as f:
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
    pprint(
        {
            "file_name": file_name,
            "total_features_shape": features.shape,
        }
    )
    return features


def normalise(inp_features):
    norm_vals = np.linalg.norm(inp_features, axis=1).reshape(-1, 1)
    return inp_features / norm_vals


def write_features(outp_file, data):
    os.makedirs(name=os.path.dirname(outp_file), exist_ok=True)
    with open(outp_file, "wb") as file:
        pickle.dump(data, file=file)
    return


def check_features_norm(outp_file):
    with open(outp_file, "rb") as file:
        data = pickle.load(file)

    norm = np.linalg.norm(data, axis=1)
    if np.all(np.isclose(norm, 1)):
        print("Test passed. Normalised features indeed have norm = 1")
        return
    else:
        print("Test failed. normalised features doesn't have norm close to 1")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Normalise MFCC features")

    parser.add_argument(
        "--input_file_name", help="name of input file", type=str, required=True
    )

    parser.add_argument(
        "--input_dir", help="path to input dir", type=str, required=True
    )

    parser.add_argument(
        "--output_dir", help="path to output dir", type=str, required=True
    )

    parser.add_argument(
        "--inp_feature_type", help="inp_feature_type", type=str, required=True
    )

    parser.add_argument(
        "--outp_feature_type", help="outp_feature_type", type=str, required=True
    )

    args = parser.parse_args()

    inp_file_name = args.input_file_name
    inp_dir = args.input_dir
    outp_dir = args.output_dir
    outp_feature_type = args.outp_feature_type
    inp_feature_type = args.inp_feature_type

    inp_file = f"{inp_dir}/{inp_file_name}_{inp_feature_type}.file"

    outp_file = f"{outp_dir}/{inp_file_name}_{outp_feature_type}.file"

    unnorm_features = load_features(inp_file)

    norm_features = normalise(unnorm_features)

    write_features(outp_file, norm_features)

    check_features_norm(outp_file)
