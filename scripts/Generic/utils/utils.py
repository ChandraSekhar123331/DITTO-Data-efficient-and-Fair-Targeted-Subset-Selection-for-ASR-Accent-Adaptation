import json
import os

import numpy as np


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
