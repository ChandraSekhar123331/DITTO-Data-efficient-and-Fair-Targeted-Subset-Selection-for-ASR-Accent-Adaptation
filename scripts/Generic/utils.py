import os
import json


def sample_greedy(lines, required_duration):
    selected_duration = 0
    index = 0
    selected_lines = []


    while selected_duration < required_duration:
        selected_lines.append(lines[index])
        selected_duration += lines[index]["duration"]
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
    