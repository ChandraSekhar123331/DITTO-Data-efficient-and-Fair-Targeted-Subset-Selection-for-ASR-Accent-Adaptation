import argparse
import os
import json
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_file", type=str, required=True)
    parser.add_argument("--outp_file", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()

    return vars(args)


def validate_arguments(args):
    inp_file = args["inp_file"]
    outp_file = args["outp_file"]
    seed = args["seed"]

    if not os.path.exists(inp_file):
        raise FileNotFoundError(f"inp_file argument == {inp_file} is invalid it is not found")

    ret = os.path.split(outp_file)
    if len(ret) != 2:
        raise Exception(f"Unable to properly parse outp_file path = {outp_file}")

    outp_dir_name, outp_file_name = ret
    if not os.path.exists(outp_dir_name):
        raise NotADirectoryError(f"head of outp_file_path = {outp_file} is outp_dir = {outp_dir_name} and this doesnot exist")
    
    if not outp_file_name:
        raise Exception(f"file_name in outp_file_path = {outp_file} is empty. file_name should not be empty.")

    if seed<0:
        raise ValueError(f"seed value can't be < 0. But the given value is = {seed}")    

    return







if __name__ == "__main__":
    args = parse_arguments()

    validate_arguments(args)

    inp_path = args["inp_file"]

    outp_path = args["outp_file"]

    seed = args["seed"]

    rgen = np.random.default_rng(seed)

    cont = dict()

    with open(inp_path) as inp_file:
        for line in inp_file.readlines():
            row = json.loads(line.strip())
            if "client_id" not in row.keys():
                print(row, "------------")
            else:
                speaker_id = row["client_id"]
                ## gender = row["gender"]
                ## TODO: We can try div wrt to gender also maybe

                if speaker_id not in cont.keys():
                    cont[speaker_id] = [row]

                else:
                    cont[speaker_id].append(row)

    for key in cont.keys():
        rgen.shuffle(cont[key])


    
    shuffled_users = list(cont.keys())
    rgen.shuffle(shuffled_users)
    inds = [0 for _ in range(len(shuffled_users))]

    with open(outp_path, 'w') as outp_file:

        while shuffled_users:
            new_shuffled_users = []
            new_inds = []
            for user_id, index in zip(shuffled_users, inds):
                assert index < len(cont[user_id])
                json.dump(cont[user_id][index], outp_file)
                outp_file.write("\n")

                if index + 1 < len(cont[user_id]):
                    new_shuffled_users.append(user_id)
                    new_inds.append(index + 1)

            shuffled_users = new_shuffled_users

            inds = new_inds






