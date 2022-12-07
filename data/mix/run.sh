accent=$1
budget=$2
out_dir=$3
flag=$4
python TSS.py --budget "$budget" --target 10 --eta 1.0 --similarity euclidean --fxn FL2MI --accent "$accent" --feature_type 39 --out_dir "$out_dir" --flag "$flag"


