import argparse
import json
from pprint import pprint

def parse_args():
    parser = argparse.ArgumentParser(description="Append wers to train.json and creates a new file")
    parser.add_argument("--file_dir", type=str, required=True, help="Path to the directory where train.json, pretrained_test_out.txt files are present. The output file = train_wers_appended.json will also be created in the same directory")
    args = parser.parse_args()
    return args

def main(args):
    file_dir = args.file_dir
    
    with open(f"{file_dir}/train.json") as train_file:
        data = [json.loads(line.strip()) for line in train_file.readlines()]
        
    with open(f"{file_dir}/pretrained_test_out.txt") as res_file:
        wers = []
        cers = []
        while True:
            line = res_file.readline()
            if not line:
                break
            line = line.strip()
            wer_pattern = "WER: "
            if line.startswith(wer_pattern):
                wers.append(float(line[len(wer_pattern):]))
                
            cer_pattern = "CER: "
            if line.startswith(cer_pattern):
                cers.append(float(line[len(cer_pattern):]))
        
    assert(len(data) == len(wers) == len(cers))
    output_file = f"{file_dir}/train_wers_appended.json"
    with open(output_file, 'w') as out_file:
        for line, wer, cer in zip(data, wers, cers):
            line["pretrained_wer"] = wer
            line["pretrained_cer"] = cer
            
            out_file.write(json.dumps(line))
            out_file.write("\n")

if __name__ == "__main__":
    args = parse_args()
    pprint(vars(args))
    main(args)