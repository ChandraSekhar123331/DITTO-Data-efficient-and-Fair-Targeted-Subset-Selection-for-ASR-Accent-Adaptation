accent=$1
budget=$2
out_dir=$3

flag=1
python TSS.py --budget "$budget" --target 10 --eta 1.0 --similarity euclidean --fxn FL2MI --accent "$accent" --feature_type 39 --out_dir "$out_dir" --flag "$flag"
mv "$1"/"$3"/train.json "$1"/"$3"/train_temp.json
mv "$1"/"$3"/stats.txt "$1"/"$3"/stats_temp.json

flag=2
python TSS.py --budget "$budget" --target 10 --eta 1.0 --similarity euclidean --fxn FL2MI --accent "$accent" --feature_type 39 --out_dir "$out_dir" --flag "$flag"

cat "$1"/"$3"/train_temp.json >> "$1"/"$3"/train.json
cat "$1"/"$3"/stats_temp.json >> "$1"/"$3"/stats.json

rm "$1"/"$3"/train_temp.json
rm "$1"/"$3"/stats_temp.txt


